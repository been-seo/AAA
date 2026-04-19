"""
Dreamer-style MBPO: World Model 위에서 관제를 "꿈꾸며" 학습

흐름:
1. World Model로 환경 시뮬 (실제 비행 없이 imagination rollout)
2. Actor(Policy)가 관제 행동 선택
3. Critic(Value Function)이 "이 상태가 얼마나 위험한지" 학습
4. 사고(분리 위반, 공역 침범 등) 발생 시 큰 negative reward
   → Value Function이 "이 패턴은 위험하다" 기억
5. 학습된 Value Function = Safety Advisor로 위험도 판진

v2: World context (웨이포인트/항로 피처) 지원
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .dataset import (STATE_DIM, NORM_MEAN, NORM_STD, MAX_NEIGHBORS, CONTEXT_DIM,
                      WORLD_DIM, WP_FEAT_DIM, MAX_NEAREST_WP, WP_TYPE_MAP,
                      _load_waypoints, _get_nearest_waypoints)


# === 행동 공간 정의 ===
HDG_DELTAS = [-30, -15, -5, 0, 5, 15, 30]
ALT_DELTAS = [-2000, -1000, 0, 1000, 2000]
SPD_DELTAS = [-50, 0, 50]
NUM_ACTIONS = len(HDG_DELTAS) * len(ALT_DELTAS) * len(SPD_DELTAS) + 1  # +1 = HOLD

def action_to_deltas(action_idx):
    """이산 행동 → (delta_hdg, delta_alt, delta_spd)"""
    if action_idx == NUM_ACTIONS - 1:
        return (0, 0, 0)  # HOLD
    n_spd = len(SPD_DELTAS)
    n_alt = len(ALT_DELTAS)
    spd_i = action_idx % n_spd
    alt_i = (action_idx // n_spd) % n_alt
    hdg_i = action_idx // (n_spd * n_alt)
    return (HDG_DELTAS[hdg_i], ALT_DELTAS[alt_i], SPD_DELTAS[spd_i])


# === 보상 함수 ===

def compute_reward(state, all_states, action_idx, airspace_polygons=None):
    """
    상태 + 행동 → 보상.
    negative = 사고/위험 → Value Function이 "이건 위험" 학습.
    """
    lat, lon, alt, gs, track = state[0], state[1], state[2], state[3], state[4]
    reward = 0.0

    # 1. 분리 위반
    for i in range(len(all_states)):
        o_lat, o_lon, o_alt = all_states[i, 0], all_states[i, 1], all_states[i, 2]
        dlat = (lat - o_lat) * 60.0
        dlon = (lon - o_lon) * 60.0 * math.cos(math.radians(float(lat)))
        h_dist = math.sqrt(dlat**2 + dlon**2)
        v_dist = abs(float(alt - o_alt))

        if h_dist < 5.0 and v_dist < 1000:
            reward -= 10000
        elif h_dist < 10.0 and v_dist < 2000:
            reward -= 1000
        elif h_dist < 15.0:
            reward -= 100

    # 2. 고도 위반
    if alt < 2000:
        reward -= 5000
    elif alt < 5000:
        reward -= 500

    # 3. 속도 이상
    if gs < 150:
        reward -= 2000
    if gs > 600:
        reward -= 500

    # 4. 생존 보너스
    reward += 10

    # 5. HOLD 보너스
    if action_idx == NUM_ACTIONS - 1:
        reward += 5

    return reward


# === Actor ===

class Actor(nn.Module):
    """관제 정책: 상태 → 행동 확률분포"""

    def __init__(self, state_dim=STATE_DIM, hidden_dim=256, world_dim=WORLD_DIM):
        super().__init__()
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
        )
        self.traffic_enc = nn.Sequential(
            nn.Linear(state_dim * 5, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        )
        # World encoder for Actor
        self.world_dim = world_dim
        if world_dim > 0:
            self.world_enc = nn.Sequential(
                nn.Linear(world_dim, 64), nn.ELU(),
                nn.Linear(64, 32), nn.ELU(),
            )
            policy_in = 128 + 64 + 32
        else:
            self.world_enc = None
            policy_in = 128 + 64

        self.policy_head = nn.Sequential(
            nn.Linear(policy_in, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 128), nn.ELU(),
            nn.Linear(128, NUM_ACTIONS),
        )

    def forward(self, own_state, traffic_states, world_feat=None):
        B = own_state.shape[0]
        s = self.state_enc(own_state)
        t = self.traffic_enc(traffic_states.reshape(B, -1))
        parts = [s, t]
        if self.world_enc is not None and world_feat is not None:
            w = self.world_enc(world_feat)
            parts.append(w)
        return self.policy_head(torch.cat(parts, dim=-1))

    def get_action(self, own_state, traffic_states, world_feat=None,
                   deterministic=False):
        logits = self.forward(own_state, traffic_states, world_feat)
        if deterministic:
            return logits.argmax(dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()


# === Critic ===

class Critic(nn.Module):
    """
    상태 → 가치(Value).
    V(s)가 낮음 = "이 상황은 위험하다"
    V(s)가 높다 = "이 상황은 안전하다"
    """

    def __init__(self, state_dim=STATE_DIM, hidden_dim=256, world_dim=WORLD_DIM):
        super().__init__()
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
        )
        self.traffic_enc = nn.Sequential(
            nn.Linear(state_dim * 5, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        )
        self.world_dim = world_dim
        if world_dim > 0:
            self.world_enc = nn.Sequential(
                nn.Linear(world_dim, 64), nn.ELU(),
                nn.Linear(64, 32), nn.ELU(),
            )
            value_in = 128 + 64 + 32
        else:
            self.world_enc = None
            value_in = 128 + 64

        self.value_head = nn.Sequential(
            nn.Linear(value_in, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, own_state, traffic_states, world_feat=None):
        B = own_state.shape[0]
        s = self.state_enc(own_state)
        t = self.traffic_enc(traffic_states.reshape(B, -1))
        parts = [s, t]
        if self.world_enc is not None and world_feat is not None:
            w = self.world_enc(world_feat)
            parts.append(w)
        return self.value_head(torch.cat(parts, dim=-1)).squeeze(-1)

    def risk_score(self, own_state, traffic_states, world_feat=None):
        with torch.no_grad():
            v = self.forward(own_state, traffic_states, world_feat)
            risk = torch.sigmoid(-v / 2000.0)
            return risk.clamp(0, 1)


# === 돌발 이벤트 주입기 ===

import random

class EventInjector:
    """
    Imagination rollout 중 돌발 이벤트를 랜덤 주입.

    - RTB_FIGHTER: 군 공역에서 임무 끝난 전투기가 갑자기 나타남
    - WX_DEVIATION: 기상 회피로 민항기가 공역 진입
    - SQUAWK_7700: 근처 항공기 비상 선언
    - POP_UP_TRAFFIC: 레이더에 갑자기 잡히는 저고도 트래픽
    - AIRSPACE_HOT: 인접 공역이 갑자기 HOT으로 전환
    """

    EVENT_TYPES = [
        ('RTB_FIGHTER',   0.03),
        ('WX_DEVIATION',  0.02),
        ('SQUAWK_7700',   0.005),
        ('POP_UP_TRAFFIC', 0.04),
        ('AIRSPACE_HOT',  0.01),
    ]

    @staticmethod
    def inject(all_states_raw, own_idx, device):
        for event_name, prob in EventInjector.EVENT_TYPES:
            if random.random() > prob:
                continue

            own = all_states_raw[own_idx]
            lat, lon, alt = float(own[0]), float(own[1]), float(own[2])

            if event_name == 'RTB_FIGHTER':
                bearing = random.uniform(0, 360)
                dist_nm = random.uniform(10, 20)
                rad = math.radians(bearing)
                new_lat = lat + dist_nm * math.cos(rad) / 60.0
                cos_lat = math.cos(math.radians(lat))
                new_lon = lon + dist_nm * math.sin(rad) / (60.0 * max(cos_lat, 0.01))
                inbound_hdg = (bearing + 180) % 360
                new_state = np.array([
                    new_lat, new_lon,
                    alt + random.uniform(-2000, 2000),
                    random.uniform(350, 500),
                    inbound_hdg,
                    random.uniform(-500, 500),
                    random.uniform(300, 450), 0.8, 0, 0,
                ], dtype=np.float32)
                return event_name, torch.from_numpy(new_state).to(device)

            elif event_name == 'WX_DEVIATION':
                bearing = random.uniform(0, 360)
                dist_nm = random.uniform(15, 30)
                rad = math.radians(bearing)
                new_lat = lat + dist_nm * math.cos(rad) / 60.0
                cos_lat = math.cos(math.radians(lat))
                new_lon = lon + dist_nm * math.sin(rad) / (60.0 * max(cos_lat, 0.01))
                deviated_hdg = random.uniform(0, 360)
                new_state = np.array([
                    new_lat, new_lon,
                    random.uniform(25000, 38000),
                    random.uniform(400, 500),
                    deviated_hdg, 0,
                    random.uniform(350, 450), 0.78, 0, 0,
                ], dtype=np.float32)
                return event_name, torch.from_numpy(new_state).to(device)

            elif event_name == 'SQUAWK_7700':
                bearing = random.uniform(0, 360)
                dist_nm = random.uniform(5, 15)
                rad = math.radians(bearing)
                new_lat = lat + dist_nm * math.cos(rad) / 60.0
                cos_lat = math.cos(math.radians(lat))
                new_lon = lon + dist_nm * math.sin(rad) / (60.0 * max(cos_lat, 0.01))
                new_state = np.array([
                    new_lat, new_lon,
                    random.uniform(10000, 30000),
                    random.uniform(200, 400),
                    random.uniform(0, 360),
                    random.uniform(-2000, 0),
                    random.uniform(200, 350), 0.6, 0, 0,
                ], dtype=np.float32)
                return event_name, torch.from_numpy(new_state).to(device)

            elif event_name == 'POP_UP_TRAFFIC':
                bearing = random.uniform(0, 360)
                dist_nm = random.uniform(5, 10)
                rad = math.radians(bearing)
                new_lat = lat + dist_nm * math.cos(rad) / 60.0
                cos_lat = math.cos(math.radians(lat))
                new_lon = lon + dist_nm * math.sin(rad) / (60.0 * max(cos_lat, 0.01))
                new_state = np.array([
                    new_lat, new_lon,
                    alt + random.uniform(-3000, 3000),
                    random.uniform(200, 450),
                    random.uniform(0, 360), 0,
                    random.uniform(200, 400), 0.7, 0, 0,
                ], dtype=np.float32)
                return event_name, torch.from_numpy(new_state).to(device)

            elif event_name == 'AIRSPACE_HOT':
                bearing = random.uniform(0, 360)
                dist_nm = random.uniform(8, 15)
                rad = math.radians(bearing)
                new_lat = lat + dist_nm * math.cos(rad) / 60.0
                cos_lat = math.cos(math.radians(lat))
                new_lon = lon + dist_nm * math.sin(rad) / (60.0 * max(cos_lat, 0.01))
                new_state = np.array([
                    new_lat, new_lon,
                    random.uniform(8000, 15000),
                    random.uniform(350, 500),
                    random.uniform(0, 360), 0,
                    random.uniform(300, 450), 0.8, 0, 0,
                ], dtype=np.float32)
                return event_name, torch.from_numpy(new_state).to(device)

        return None, None


# === Imagination Rollout ===

class DreamerTrainer:
    """
    World Model + Actor + Critic 동시 학습.

    v2: World context 지원 — imagination 중 매 스텝 웨이포인트 피처 계산
    """

    def __init__(self, world_model, data_dir='data', device='cuda',
                 imagination_horizon=15, gamma=0.99, actor_lr=3e-4, critic_lr=3e-4):
        self.world_model = world_model
        self.device = torch.device(device)
        self.world_model.to(self.device)
        self.world_model.eval()

        # 웨이포인트 데이터 로드
        wp_data = _load_waypoints(data_dir)
        if wp_data is not None:
            self._wp_array, self._wp_types, self._wp_names = wp_data
            self._has_world = True
            world_dim = WORLD_DIM
        else:
            self._has_world = False
            world_dim = 0

        self.actor = Actor(world_dim=world_dim).to(self.device)
        self.critic = Critic(world_dim=world_dim).to(self.device)
        self.target_critic = Critic(world_dim=world_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.horizon = imagination_horizon
        self.gamma = gamma
        self.norm_mean = torch.from_numpy(NORM_MEAN).to(self.device)
        self.norm_std = torch.from_numpy(NORM_STD).to(self.device)

        self.total_episodes = 0
        self.total_crashes = 0
        self.total_safe = 0

    def _denormalize(self, state_norm):
        return state_norm * self.norm_std + self.norm_mean

    def _normalize(self, state_raw):
        return (state_raw - self.norm_mean) / self.norm_std

    def _compute_world_feat(self, states_raw):
        """비정규화 상태에서 world 피처 계산.

        :param states_raw: (B, STATE_DIM) 비정규화 상태
        :return: (B, WORLD_DIM) or None
        """
        if not self._has_world:
            return None

        B = states_raw.shape[0]
        world = np.zeros((B, WORLD_DIM), dtype=np.float32)
        states_np = states_raw.detach().cpu().numpy()

        for i in range(B):
            lat, lon, track = states_np[i, 0], states_np[i, 1], states_np[i, 4]
            world[i] = _get_nearest_waypoints(
                lat, lon, track, self._wp_array, self._wp_types)

        return torch.from_numpy(world).to(self.device)

    def _get_nearby_traffic(self, all_states_raw, own_idx, k=5):
        own = all_states_raw[own_idx]
        B = all_states_raw.shape[0]
        if B <= 1:
            return torch.zeros(1, k, STATE_DIM, device=self.device)

        lat, lon = own[0], own[1]
        dlats = (all_states_raw[:, 0] - lat) * 60.0
        dlons = (all_states_raw[:, 1] - lon) * 60.0 * torch.cos(
            torch.deg2rad(lat))
        dists = torch.sqrt(dlats**2 + dlons**2)
        dists[own_idx] = 1e9

        _, indices = dists.topk(min(k, B - 1), largest=False)
        traffic = all_states_raw[indices]

        if traffic.shape[0] < k:
            pad = torch.zeros(k - traffic.shape[0], STATE_DIM, device=self.device)
            traffic = torch.cat([traffic, pad], dim=0)

        return traffic.unsqueeze(0)

    def imagine_rollout(self, initial_states_norm, initial_contexts,
                        initial_world=None, batch_size=32):
        """
        World Model 위에서 imagination rollout.

        v2: 매 스텝 world context를 동적으로 계산해서 WM에 전달
        """
        B, K, D = initial_states_norm.shape

        # WM warm-up
        h = torch.zeros(self.world_model.gru.num_layers, B, self.world_model.hidden_dim,
                        device=self.device)
        z = torch.zeros(B, self.world_model.latent_dim, device=self.device)

        with torch.no_grad():
            for t in range(K):
                state_t = initial_states_norm[:, t]
                ctx_t = initial_contexts[:, t]
                state_emb = self.world_model._encode_state(state_t)
                interaction = self.world_model.neighbor_attn(state_emb, ctx_t)

                world_emb = None
                if initial_world is not None:
                    world_emb = self.world_model._encode_world(initial_world[:, t])

                gru_in = self.world_model._build_gru_input(
                    state_emb, interaction, z, world_emb)
                gru_out, h = self.world_model.gru(gru_in, h)
                h_t = gru_out.squeeze(1)
                post_mean, post_logstd = self.world_model._posterior(h_t, state_emb)
                z = self.world_model._sample_z(post_mean, post_logstd)

        current_state_raw = self._denormalize(initial_states_norm[:, -1])
        empty_ctx = torch.zeros(B, MAX_NEIGHBORS, CONTEXT_DIM, device=self.device)

        all_states = [current_state_raw]
        all_actions = []
        all_rewards = []
        all_values = []

        for step in range(self.horizon):
            own_state = current_state_raw

            traffic = torch.zeros(B, 5, STATE_DIM, device=self.device)
            if B > 1:
                for i in range(min(B, 5)):
                    idx = (torch.arange(B, device=self.device) + i + 1) % B
                    traffic[:, i] = current_state_raw[idx]

            # World features (동적 계산)
            world_feat = self._compute_world_feat(current_state_raw)

            # Critic value
            with torch.no_grad():
                value = self.target_critic(own_state, traffic, world_feat)
            all_values.append(value)

            # Actor action
            action = self.actor.get_action(own_state, traffic, world_feat)
            all_actions.append(action)

            # 행동을 상태에 반영
            deltas = torch.zeros(B, STATE_DIM, device=self.device)
            for i in range(B):
                dh, da, ds = action_to_deltas(action[i].item())
                deltas[i, 4] = dh
                deltas[i, 2] = da
                deltas[i, 3] = ds

            modified_state_raw = current_state_raw.clone()
            modified_state_raw[:, 4] = (modified_state_raw[:, 4] + deltas[:, 4]) % 360
            modified_state_raw[:, 2] = (modified_state_raw[:, 2] + deltas[:, 2]).clamp(2000, 45000)
            modified_state_raw[:, 3] = (modified_state_raw[:, 3] + deltas[:, 3]).clamp(200, 600)

            # WM prior로 다음 상태 예측
            modified_norm = self._normalize(modified_state_raw)

            # 수정된 상태에서 world 피처 재계산
            world_feat_wm = self._compute_world_feat(modified_state_raw)

            with torch.no_grad():
                state_emb = self.world_model._encode_state(modified_norm)
                interaction = self.world_model.neighbor_attn(state_emb, empty_ctx)

                world_emb = None
                if world_feat_wm is not None:
                    world_emb = self.world_model._encode_world(world_feat_wm)

                gru_in = self.world_model._build_gru_input(
                    state_emb, interaction, z, world_emb)
                gru_out, h = self.world_model.gru(gru_in, h)
                h_t = gru_out.squeeze(1)
                prior_mean, prior_logstd = self.world_model._prior(h_t)
                z = self.world_model._sample_z(prior_mean, prior_logstd)
                pred_mean, pred_logstd = self.world_model._decode_state(h_t, z)
                next_state_norm = self.world_model._sample_z(pred_mean, pred_logstd)

            next_state_raw = self._denormalize(next_state_norm)

            # 돌발 이벤트 주입
            injected_traffic = []
            for i in range(B):
                evt, new_ac = EventInjector.inject(
                    next_state_raw.detach(), i, self.device)
                if new_ac is not None:
                    injected_traffic.append(new_ac)

            extra_traffic = (torch.stack(injected_traffic, dim=0)
                             if injected_traffic else None)

            # 보상 계산
            rewards = torch.zeros(B, device=self.device)
            for i in range(B):
                others = torch.cat([next_state_raw[:i], next_state_raw[i+1:]], dim=0)
                if extra_traffic is not None:
                    others = torch.cat([others, extra_traffic], dim=0)
                if others.shape[0] == 0:
                    others = torch.zeros(1, STATE_DIM, device=self.device)
                rewards[i] = compute_reward(
                    next_state_raw[i].cpu().numpy(),
                    others.cpu().numpy(),
                    action[i].item()
                )
            all_rewards.append(rewards)

            current_state_raw = next_state_raw
            all_states.append(current_state_raw)

        # 마지막 스텝 value (bootstrap)
        traffic = torch.zeros(B, 5, STATE_DIM, device=self.device)
        world_feat_final = self._compute_world_feat(current_state_raw)
        with torch.no_grad():
            final_value = self.target_critic(current_state_raw, traffic, world_feat_final)

        return {
            'states': torch.stack(all_states, dim=1),
            'actions': torch.stack(all_actions, dim=1),
            'rewards': torch.stack(all_rewards, dim=1),
            'values': torch.stack(all_values, dim=1),
            'final_value': final_value,
        }

    def compute_returns(self, rewards, values, final_value, lam=0.95):
        H = rewards.shape[1]
        returns = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(H)):
            next_v = final_value if t == H - 1 else values[:, t + 1]
            delta = rewards[:, t] + self.gamma * next_v - values[:, t]
            last_gae = delta + self.gamma * lam * last_gae
            returns[:, t] = last_gae + values[:, t]

        return returns

    def train_step(self, initial_states_norm, initial_contexts, initial_world=None):
        """
        한 번의 학습 스텝.
        """
        rollout = self.imagine_rollout(
            initial_states_norm, initial_contexts, initial_world)

        returns = self.compute_returns(
            rollout['rewards'], rollout['values'], rollout['final_value'])

        # === Critic 업데이트 ===
        states_flat = rollout['states'][:, :-1].reshape(-1, STATE_DIM)
        traffic_flat = torch.zeros(states_flat.shape[0], 5, STATE_DIM,
                                   device=self.device)
        returns_flat = returns.reshape(-1)

        # world feat for critic
        world_flat = self._compute_world_feat(states_flat) if self._has_world else None

        critic_pred = self.critic(states_flat, traffic_flat, world_flat)
        critic_loss = F.mse_loss(critic_pred, returns_flat.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_opt.step()

        # === Actor 업데이트 ===
        own_states = rollout['states'][:, :-1]
        B, H, D = own_states.shape
        own_flat = own_states.reshape(B * H, D)
        traffic_flat_a = torch.zeros(B * H, 5, STATE_DIM, device=self.device)
        world_flat_a = self._compute_world_feat(own_flat) if self._has_world else None

        logits = self.actor(own_flat, traffic_flat_a, world_flat_a)
        dist = torch.distributions.Categorical(logits=logits)
        actions_flat = rollout['actions'].reshape(B * H)
        log_probs = dist.log_prob(actions_flat)

        with torch.no_grad():
            values_flat = self.critic(own_flat, traffic_flat_a, world_flat_a)
        advantages = (returns_flat - values_flat).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs * advantages).mean()
        entropy = dist.entropy().mean()
        actor_loss = actor_loss - 0.01 * entropy

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()

        # Target critic soft update
        tau = 0.005
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        total_reward = rollout['rewards'].sum(dim=1).mean().item()
        crashes = (rollout['rewards'] < -5000).any(dim=1).sum().item()
        self.total_episodes += rollout['rewards'].shape[0]
        self.total_crashes += crashes
        self.total_safe += rollout['rewards'].shape[0] - crashes

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_reward': total_reward,
            'mean_value': rollout['values'].mean().item(),
            'crashes': crashes,
            'entropy': entropy.item(),
        }

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'total_episodes': self.total_episodes,
            'total_crashes': self.total_crashes,
            'total_safe': self.total_safe,
            'has_world': self._has_world,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_critic.load_state_dict(ckpt['target_critic'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.total_episodes = ckpt.get('total_episodes', 0)
        self.total_crashes = ckpt.get('total_crashes', 0)
        self.total_safe = ckpt.get('total_safe', 0)

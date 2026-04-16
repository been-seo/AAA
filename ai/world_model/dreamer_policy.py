"""
Dreamer-style MBPO: World Model 안에서 관제를 "꿈꾸며" 학습

흐름:
1. World Model이 환경 역할 (실제 비행 없이 imagination rollout)
2. Actor(Policy)가 관제 행동 선택
3. Critic(Value Function)이 "이 상태가 얼마나 위험한지" 학습
4. 사고(분리 위반, 공역 침범 등) 발생 시 큰 negative reward
   → Value Function이 "이 패턴은 위험하다" 기억
5. 학습된 Value Function = Safety Advisor의 위험도 평가 엔진

"400kill 2death" — 수만 번 깨지면서 뭐가 진짜 위험한지 배움.
tight vectoring은 안전하다는 것도 경험으로 학습.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .dataset import STATE_DIM, NORM_MEAN, NORM_STD, MAX_NEIGHBORS, CONTEXT_DIM


# ── Symlog Transform (DreamerV3) ──
# 큰 보상/가치를 압축하여 학습 안정화
def symlog(x):
    """sign(x) * ln(1 + |x|) — 큰 값을 로그 스케일로 압축"""
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    """symlog의 역변환"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# ── 행동 공간 ──
# 관제 지시: (delta_hdg, delta_alt, delta_spd)
# delta_hdg: -30, -15, -5, 0, +5, +15, +30 (7)
# delta_alt: -2000, -1000, 0, +1000, +2000 (5)
# delta_spd: -50, 0, +50 (3)
# 총: 7 * 5 * 3 = 105 + 1(유지) = 106
HDG_DELTAS = [-30, -15, -5, 0, 5, 15, 30]
ALT_DELTAS = [-2000, -1000, 0, 1000, 2000]
SPD_DELTAS = [-50, 0, 50]
NUM_ACTIONS = len(HDG_DELTAS) * len(ALT_DELTAS) * len(SPD_DELTAS) + 1  # +1 = HOLD

def action_to_deltas(action_idx):
    """이산 행동 → (delta_hdg, delta_alt, delta_spd) — 단건 조회용"""
    if action_idx == NUM_ACTIONS - 1:
        return (0, 0, 0)  # HOLD
    n_spd = len(SPD_DELTAS)
    n_alt = len(ALT_DELTAS)
    spd_i = action_idx % n_spd
    alt_i = (action_idx // n_spd) % n_alt
    hdg_i = action_idx // (n_spd * n_alt)
    return (HDG_DELTAS[hdg_i], ALT_DELTAS[alt_i], SPD_DELTAS[spd_i])


# ── 배치 행동 룩업 테이블 (GPU 벡터화) ──
def _build_action_table(device):
    """모든 행동의 (delta_hdg, delta_alt, delta_spd) 테이블. (NUM_ACTIONS, 3)"""
    table = []
    for h in HDG_DELTAS:
        for a in ALT_DELTAS:
            for s in SPD_DELTAS:
                table.append([h, a, s])
    table.append([0, 0, 0])  # HOLD
    return torch.tensor(table, dtype=torch.float32, device=device)


# ── 벡터화 보상 함수 (전체 배치를 GPU에서 한번에) ──

def compute_reward_batch(states_raw, actions, device):
    """
    배치 전체 보상을 텐서 연산으로 계산.

    states_raw: (B, STATE_DIM) 비정규화 상태
    actions: (B,) 행동 인덱스
    return: (B,) 보상
    """
    B = states_raw.shape[0]
    lat = states_raw[:, 0]   # (B,)
    lon = states_raw[:, 1]
    alt = states_raw[:, 2]
    gs  = states_raw[:, 3]

    rewards = torch.ones(B, device=device)  # 생존 보너스 1.0

    # HOLD 보너스
    rewards += (actions == NUM_ACTIONS - 1).float() * 0.5

    # 분리 위반: 모든 쌍의 거리 계산 (B x B)
    dlat = (lat.unsqueeze(1) - lat.unsqueeze(0)) * 60.0          # (B, B) NM
    cos_lat = torch.cos(torch.deg2rad(lat)).unsqueeze(1)          # (B, 1)
    dlon = (lon.unsqueeze(1) - lon.unsqueeze(0)) * 60.0 * cos_lat  # (B, B) NM
    h_dist = torch.sqrt(dlat**2 + dlon**2)                        # (B, B)
    v_dist = torch.abs(alt.unsqueeze(1) - alt.unsqueeze(0))       # (B, B)

    # 자기 자신 제외
    eye_mask = torch.eye(B, device=device, dtype=torch.bool)
    h_dist = h_dist.masked_fill(eye_mask, 1e9)

    # RA급: h<5 & v<1000 → -50
    ra = (h_dist < 5.0) & (v_dist < 1000)
    rewards -= ra.float().sum(dim=1) * 50

    # TA급: h<10 & v<2000 (RA 제외) → -10
    ta = (h_dist < 10.0) & (v_dist < 2000) & ~ra
    rewards -= ta.float().sum(dim=1) * 10

    # 접근: h<15 (RA/TA 제외) → -2
    approach = (h_dist < 15.0) & ~ra & ~ta
    rewards -= approach.float().sum(dim=1) * 2

    # 고도 위반
    rewards -= (alt < 2000).float() * 30
    rewards -= ((alt >= 2000) & (alt < 5000)).float() * 5

    # 속도 이상
    rewards -= (gs < 150).float() * 20
    rewards -= (gs > 600).float() * 5

    return rewards


# ── Actor (Policy Network) ──

class Actor(nn.Module):
    """관제 정책: 상태 → 행동 확률분포"""

    def __init__(self, state_dim=STATE_DIM, hidden_dim=256):
        super().__init__()
        # 자기 상태 + 주변 요약 = state_dim + 64
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
        )
        self.traffic_enc = nn.Sequential(
            nn.Linear(state_dim * 5, 128), nn.ELU(),  # 주변 5대 상태 concat
            nn.Linear(128, 64), nn.ELU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(128 + 64, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 128), nn.ELU(),
            nn.Linear(128, NUM_ACTIONS),
        )

    def forward(self, own_state, traffic_states):
        """
        :param own_state: (B, STATE_DIM)
        :param traffic_states: (B, 5, STATE_DIM) 가까운 5대
        :return: (B, NUM_ACTIONS) logits
        """
        B = own_state.shape[0]
        s = self.state_enc(own_state)
        t = self.traffic_enc(traffic_states.reshape(B, -1))
        return self.policy_head(torch.cat([s, t], dim=-1))

    def get_action(self, own_state, traffic_states, deterministic=False):
        logits = self.forward(own_state, traffic_states)
        if deterministic:
            return logits.argmax(dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()


# ── Critic (Value Function = 위험도 평가 엔진) ──

class Critic(nn.Module):
    """
    상태 → 가치(Value).
    V(s)가 낮다 = "이 상황은 위험하다" (미래에 큰 negative reward 예상).
    V(s)가 높다 = "이 상황은 안전하다".

    학습된 후 Safety Advisor가 이걸 호출해서 위험도를 산출.
    """

    def __init__(self, state_dim=STATE_DIM, hidden_dim=256):
        super().__init__()
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
        )
        self.traffic_enc = nn.Sequential(
            nn.Linear(state_dim * 5, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128 + 64, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, own_state, traffic_states):
        B = own_state.shape[0]
        s = self.state_enc(own_state)
        t = self.traffic_enc(traffic_states.reshape(B, -1))
        return self.value_head(torch.cat([s, t], dim=-1)).squeeze(-1)

    def risk_score(self, own_state, traffic_states):
        """
        Safety Advisor용: V(s) → 위험도 [0, 1].
        Critic은 symlog space로 학습됨.

        현재 체크포인트(1.3M ep)는 대부분의 상태에 음수 V를 출력하므로,
        symlog 값을 직접 sigmoid로 변환하되 스케일을 넓게 잡는다.
        학습이 진행되면 안전 상태의 V가 양수로 올라가면서 자연스럽게 보정됨.

        risk = sigmoid(-V_symlog * 0.5)
          V_sym=+3  → risk=0.18 (안전)
          V_sym= 0  → risk=0.50
          V_sym=-1  → risk=0.62
          V_sym=-3  → risk=0.82 (주의)
          V_sym=-5  → risk=0.92 (위험)
          V_sym=-10 → risk=0.99 (critical)
        """
        with torch.no_grad():
            v_symlog = self.forward(own_state, traffic_states)
            risk = torch.sigmoid(-v_symlog * 0.5)
            return risk.clamp(0, 1)


# ── 돌발 이벤트 주입 (배치 벡터화) ──

def inject_events_batch(states_raw, device):
    """
    배치 전체에 돌발 이벤트를 GPU 텐서 연산으로 주입.
    각 배치 항목에 ~10.5% 확률로 이벤트 항공기 1대 추가.

    states_raw: (B, STATE_DIM) 비정규화 상태
    return: (B, STATE_DIM) 주입된 항공기 상태, (B,) 주입 여부 마스크
    """
    B = states_raw.shape[0]
    # 통합 확률: 0.03+0.02+0.005+0.04+0.01 ≈ 10.5%
    inject_mask = torch.rand(B, device=device) < 0.105

    lat = states_raw[:, 0]
    lon = states_raw[:, 1]
    alt = states_raw[:, 2]

    # 랜덤 방위/거리 (5~20NM)
    bearing = torch.rand(B, device=device) * 360.0
    dist_nm = torch.rand(B, device=device) * 15.0 + 5.0
    rad = bearing * (math.pi / 180.0)
    cos_lat = torch.cos(lat * (math.pi / 180.0)).clamp(min=0.01)

    new_lat = lat + dist_nm * torch.cos(rad) / 60.0
    new_lon = lon + dist_nm * torch.sin(rad) / (60.0 * cos_lat)
    new_alt = alt + (torch.rand(B, device=device) - 0.5) * 6000  # ±3000ft
    new_gs = torch.rand(B, device=device) * 300 + 200             # 200~500kt
    new_hdg = torch.rand(B, device=device) * 360
    new_vrate = (torch.rand(B, device=device) - 0.5) * 2000       # ±1000fpm
    new_ias = new_gs - torch.rand(B, device=device) * 50
    new_mach = torch.full((B,), 0.75, device=device)
    zeros = torch.zeros(B, device=device)

    injected = torch.stack([
        new_lat, new_lon, new_alt, new_gs, new_hdg,
        new_vrate, new_ias, new_mach, zeros, zeros
    ], dim=1)  # (B, STATE_DIM)

    return injected, inject_mask


# ── Imagination Rollout (World Model 안에서 꿈꾸기) ──

class DreamerTrainer:
    """
    World Model + Actor + Critic 동시 학습.

    매 iteration:
    1. World Model로 imagination rollout (H 스텝)
    2. Actor가 각 스텝에서 행동 선택
    3. 보상 계산 (충돌/위반 = 대형 negative)
    4. Critic이 returns 학습 (어디가 위험한지)
    5. Actor가 Critic을 maximize하도록 학습 (위험 회피)
    """

    def __init__(self, world_model, device='cuda',
                 imagination_horizon=15, gamma=0.99, actor_lr=1e-4, critic_lr=1e-4):
        self.world_model = world_model
        self.device = torch.device(device)
        self.world_model.to(self.device)
        self.world_model.eval()  # WM은 별도 학습

        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.target_critic = Critic().to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

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

    def _get_nearby_traffic(self, all_states_raw, own_idx, k=5):
        """주변 항공기 5대 추출 (거리순)"""
        own = all_states_raw[own_idx]
        B = all_states_raw.shape[0]
        if B <= 1:
            return torch.zeros(1, k, STATE_DIM, device=self.device)

        lat, lon = own[0], own[1]
        dlats = (all_states_raw[:, 0] - lat) * 60.0
        dlons = (all_states_raw[:, 1] - lon) * 60.0 * torch.cos(
            torch.deg2rad(lat))
        dists = torch.sqrt(dlats**2 + dlons**2)
        dists[own_idx] = 1e9  # 자기 자신 제외

        _, indices = dists.topk(min(k, B - 1), largest=False)
        traffic = all_states_raw[indices]  # (k', STATE_DIM)

        # 패딩
        if traffic.shape[0] < k:
            pad = torch.zeros(k - traffic.shape[0], STATE_DIM, device=self.device)
            traffic = torch.cat([traffic, pad], dim=0)

        return traffic.unsqueeze(0)  # (1, k, STATE_DIM)

    def imagine_rollout(self, initial_states_norm, initial_contexts, batch_size=32):
        """
        World Model 안에서 imagination rollout (GPU 벡터화).

        Returns: (states, actions, rewards, values) 시퀀스
        """
        B, K, D = initial_states_norm.shape

        # 행동 룩업 테이블 (lazy init)
        if not hasattr(self, '_action_table'):
            self._action_table = _build_action_table(self.device)

        # WM warm-up: 과거 시퀀스로 hidden state 초기화
        h = torch.zeros(self.world_model.gru.num_layers, B, self.world_model.hidden_dim,
                        device=self.device)
        z = torch.zeros(B, self.world_model.latent_dim, device=self.device)

        with torch.no_grad():
            for t in range(K):
                state_t = initial_states_norm[:, t]
                ctx_t = initial_contexts[:, t]
                state_emb = self.world_model._encode_state(state_t)
                interaction = self.world_model.neighbor_attn(state_emb, ctx_t)
                gru_in = torch.cat([state_emb, interaction, z], dim=-1).unsqueeze(1)
                gru_out, h = self.world_model.gru(gru_in, h)
                h_t = gru_out.squeeze(1)
                post_mean, post_logstd = self.world_model._posterior(h_t, state_emb)
                z = self.world_model._sample_z(post_mean, post_logstd)

        # 현재 상태 (비정규화)
        current_state_raw = self._denormalize(initial_states_norm[:, -1])  # (B, D)
        empty_ctx = torch.zeros(B, MAX_NEIGHBORS, CONTEXT_DIM, device=self.device)

        # Imagination rollout
        all_states = [current_state_raw]
        all_actions = []
        all_rewards = []
        all_values = []

        # traffic 인덱스 (배치 내 순환 시프트) — 미리 계산
        shift_indices = torch.stack([
            (torch.arange(B, device=self.device) + i + 1) % B
            for i in range(min(B - 1, 5))
        ], dim=1)  # (B, 5)

        for step in range(self.horizon):
            # Actor/Critic은 정규화된 상태를 입력으로 받음
            own_state_norm = self._normalize(current_state_raw)  # (B, D)

            # 주변 항공기 = 배치 내 다른 항공기 (정규화, 벡터화)
            traffic_norm = own_state_norm[shift_indices]  # (B, 5, D)

            # Critic value
            with torch.no_grad():
                value = self.target_critic(own_state_norm, traffic_norm)
            all_values.append(value)

            # Actor action
            action = self.actor.get_action(own_state_norm, traffic_norm)
            all_actions.append(action)

            # 행동을 상태에 반영 (룩업 테이블, GPU 벡터화)
            action_deltas = self._action_table[action]  # (B, 3) = [hdg, alt, spd]

            modified_state_raw = current_state_raw.clone()
            modified_state_raw[:, 4] = (modified_state_raw[:, 4] + action_deltas[:, 0]) % 360
            modified_state_raw[:, 2] = (modified_state_raw[:, 2] + action_deltas[:, 1]).clamp(2000, 45000)
            modified_state_raw[:, 3] = (modified_state_raw[:, 3] + action_deltas[:, 2]).clamp(200, 600)

            # WM prior로 다음 상태 예측
            modified_norm = self._normalize(modified_state_raw)
            with torch.no_grad():
                state_emb = self.world_model._encode_state(modified_norm)
                interaction = self.world_model.neighbor_attn(state_emb, empty_ctx)
                gru_in = torch.cat([state_emb, interaction, z], dim=-1).unsqueeze(1)
                gru_out, h = self.world_model.gru(gru_in, h)
                h_t = gru_out.squeeze(1)
                prior_mean, prior_logstd = self.world_model._prior(h_t)
                z = self.world_model._sample_z(prior_mean, prior_logstd)
                pred_mean, pred_logstd = self.world_model._decode_state(h_t, z)
                next_state_norm = self.world_model._sample_z(pred_mean, pred_logstd)

            next_state_raw = self._denormalize(next_state_norm)

            # ── 돌발 이벤트 주입 (배치 벡터화) ──
            # 주입된 항공기 위치를 배치 내 일부 항목에 덮어씌움
            # → 갑자기 위험한 위치에 항공기가 나타난 효과
            injected, inject_mask = inject_events_batch(next_state_raw, self.device)
            reward_states = next_state_raw.clone()
            reward_states[inject_mask] = injected[inject_mask]

            # 보상 계산 (GPU 벡터화)
            rewards = compute_reward_batch(reward_states, action, self.device)
            all_rewards.append(rewards)

            current_state_raw = next_state_raw
            all_states.append(current_state_raw)

        # 마지막 스텝 value (bootstrap)
        own_norm_final = self._normalize(current_state_raw)
        traffic_norm_final = own_norm_final[shift_indices]
        with torch.no_grad():
            final_value = self.target_critic(own_norm_final, traffic_norm_final)

        return {
            'states': torch.stack(all_states, dim=1),      # (B, H+1, D)
            'actions': torch.stack(all_actions, dim=1),     # (B, H)
            'rewards': torch.stack(all_rewards, dim=1),     # (B, H)
            'values': torch.stack(all_values, dim=1),       # (B, H)
            'final_value': final_value,                     # (B,)
        }

    def compute_returns(self, rewards, values_symlog, final_value_symlog, lam=0.95):
        """
        GAE-style lambda returns — symlog space에서 직접 계산.
        rewards는 raw space이므로 symlog 변환 후 계산.
        이렇게 하면 symexp 폭발 문제를 방지.
        """
        H = rewards.shape[1]
        # rewards를 symlog space로 변환
        rewards_sl = symlog(rewards)
        values = values_symlog
        final_value = final_value_symlog

        returns = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(H)):
            next_v = final_value if t == H - 1 else values[:, t + 1]
            delta = rewards_sl[:, t] + self.gamma * next_v - values[:, t]
            last_gae = delta + self.gamma * lam * last_gae
            returns[:, t] = last_gae + values[:, t]

        # returns는 이미 symlog space — 클램프로 안전 보장
        return returns.clamp(-10, 10)

    def train_step(self, initial_states_norm, initial_contexts):
        """
        한 번의 학습 스텝:
        1. Imagination rollout
        2. Critic 업데이트 (returns 학습)
        3. Actor 업데이트 (value 최대화)
        """
        rollout = self.imagine_rollout(initial_states_norm, initial_contexts)

        returns = self.compute_returns(
            rollout['rewards'], rollout['values'], rollout['final_value'])

        # ── Critic 업데이트 (symlog space) ──
        # rollout['states']는 raw space → 정규화 필요
        states_raw_flat = rollout['states'][:, :-1].reshape(-1, STATE_DIM)  # (B*H, D)
        states_flat = (states_raw_flat - self.norm_mean) / self.norm_std   # 정규화
        traffic_flat = torch.zeros(states_flat.shape[0], 5, STATE_DIM,
                                   device=self.device)
        returns_flat = returns.reshape(-1)

        # returns는 이미 symlog space (compute_returns에서 변환됨)
        symlog_targets = torch.nan_to_num(returns_flat.detach(), nan=0.0).clamp(-10, 10)

        critic_pred = self.critic(states_flat, traffic_flat)
        critic_loss = F.mse_loss(critic_pred, symlog_targets)

        # NaN 감지 시 critic 업데이트 스킵
        if torch.isfinite(critic_loss):
            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_opt.step()

        # ── Actor 업데이트 ──
        own_states = rollout['states'][:, :-1]  # (B, H, D) raw
        B, H, D = own_states.shape
        own_flat = ((own_states - self.norm_mean) / self.norm_std).reshape(B * H, D)  # 정규화
        traffic_flat_a = torch.zeros(B * H, 5, STATE_DIM, device=self.device)

        logits = self.actor(own_flat, traffic_flat_a)  # (B*H, A)
        # NaN 방지: logits 클램프
        logits = torch.nan_to_num(logits, nan=0.0).clamp(-20, 20)
        dist = torch.distributions.Categorical(logits=logits)
        actions_flat = rollout['actions'].reshape(B * H)
        log_probs = dist.log_prob(actions_flat)

        # Advantage (symlog space에서 비교)
        with torch.no_grad():
            values_flat = self.critic(own_flat, traffic_flat_a)
        advantages = (symlog_targets - values_flat).detach()
        advantages = torch.nan_to_num(advantages, nan=0.0)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.clamp(-5, 5)  # 극단적 advantage 클램프

        actor_loss = -(log_probs * advantages).mean()
        entropy = dist.entropy().mean()
        actor_loss = actor_loss - 0.03 * entropy  # entropy bonus (강화)

        # NaN 감지 시 actor 업데이트 스킵
        if torch.isfinite(actor_loss):
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

        # Target critic soft update
        tau = 0.02  # 빠른 추적 (발산 방지)
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        # 통계
        total_reward = rollout['rewards'].sum(dim=1).mean().item()
        crashes = (rollout['rewards'] < -30).any(dim=1).sum().item()
        self.total_episodes += rollout['rewards'].shape[0]
        self.total_crashes += crashes
        self.total_safe += rollout['rewards'].shape[0] - crashes

        # mean_value: symlog space의 평균을 직접 표시 (안정적)
        mean_v_real = rollout['values'].mean().item()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_reward': total_reward,
            'mean_value': mean_v_real,
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
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if 'actor' in ckpt:
            self.actor.load_state_dict(ckpt['actor'])
            self.critic.load_state_dict(ckpt['critic'])
            self.target_critic.load_state_dict(ckpt['target_critic'])
            self.actor_opt.load_state_dict(ckpt['actor_opt'])
            self.critic_opt.load_state_dict(ckpt['critic_opt'])
            print(f"[Dreamer] Loaded actor/critic from checkpoint")
        else:
            print(f"[Dreamer] No actor/critic in checkpoint - starting fresh")
        self.total_episodes = ckpt.get('total_episodes', 0)
        self.total_crashes = ckpt.get('total_crashes', 0)
        self.total_safe = ckpt.get('total_safe', 0)

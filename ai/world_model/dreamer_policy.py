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
    """이산 행동 → (delta_hdg, delta_alt, delta_spd)"""
    if action_idx == NUM_ACTIONS - 1:
        return (0, 0, 0)  # HOLD
    n_spd = len(SPD_DELTAS)
    n_alt = len(ALT_DELTAS)
    spd_i = action_idx % n_spd
    alt_i = (action_idx // n_spd) % n_alt
    hdg_i = action_idx // (n_spd * n_alt)
    return (HDG_DELTAS[hdg_i], ALT_DELTAS[alt_i], SPD_DELTAS[spd_i])


# ── 보상 함수 (경험으로 배울 교훈) ──

def compute_reward(state, all_states, action_idx, airspace_polygons=None):
    """
    상태 + 행동 → 보상.
    큰 negative = 사고/위험 → Value Function이 "이건 위험" 학습.

    state: (STATE_DIM,) 비정규화 — [lat, lon, alt, gs, track, vrate, ...]
    all_states: (N, STATE_DIM) 주변 항공기 비정규화 상태
    """
    lat, lon, alt, gs, track = state[0], state[1], state[2], state[3], state[4]
    reward = 0.0

    # 1. 분리 위반 — 핵심 교훈: 가까이 가면 죽는다
    for i in range(len(all_states)):
        o_lat, o_lon, o_alt = all_states[i, 0], all_states[i, 1], all_states[i, 2]
        # 수평 거리 (NM)
        dlat = (lat - o_lat) * 60.0
        dlon = (lon - o_lon) * 60.0 * math.cos(math.radians(float(lat)))
        h_dist = math.sqrt(dlat**2 + dlon**2)
        v_dist = abs(float(alt - o_alt))

        if h_dist < 5.0 and v_dist < 1000:
            # RA급 — 대참사
            reward -= 10000
        elif h_dist < 10.0 and v_dist < 2000:
            # TA급 — 위험
            reward -= 1000
        elif h_dist < 15.0:
            # 접근 — 주의 필요하지만 tight vectoring일 수도 있음
            # 경험이 쌓이면 이 수준은 "별 문제 없다"로 학습됨
            reward -= 100

    # 2. 고도 위반
    if alt < 2000:
        reward -= 5000  # MSA 하회 = 지형 충돌
    elif alt < 5000:
        reward -= 500

    # 3. 속도 이상
    if gs < 150:
        reward -= 2000  # 실속
    if gs > 600:
        reward -= 500  # 과속

    # 4. 생존 보너스 — 아무 일 없이 비행하면 보상
    reward += 10

    # 5. HOLD 보너스 — 불필요한 기동 지양
    if action_idx == NUM_ACTIONS - 1:
        reward += 5

    return reward


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
        V(s)가 낮을수록 위험 → risk 높음.
        sigmoid(-V(s) / scale)로 변환.
        """
        with torch.no_grad():
            v = self.forward(own_state, traffic_states)
            # V(s) 범위: 대략 -10000(최악) ~ +500(안전)
            # risk = sigmoid(-V / 2000) → 위험할수록 1에 가까움
            risk = torch.sigmoid(-v / 2000.0)
            return risk.clamp(0, 1)


# ── 돌발 이벤트 주입기 ──

import random

class EventInjector:
    """
    Imagination rollout 중 돌발 이벤트를 랜덤 주입.
    실제 관제에서 일어나는 예측 불가능한 상황들:

    - RTB_FIGHTER: 옆 공역에서 임무 끝난 전투기가 갑자기 튀어나옴
    - WX_DEVIATION: 기상 회피(weather deviation)로 민항기가 공역 진입
    - SQUAWK_7700: 근처 항공기 비상 선언 → 우선권 양보 필요
    - POP_UP_TRAFFIC: 레이더에 갑자기 잡히는 저고도 트래픽
    - AIRSPACE_HOT: 인접 공역이 갑자기 HOT으로 전환 (항공기 유입)
    """

    # (이벤트 이름, 발생 확률/스텝, 설명)
    EVENT_TYPES = [
        ('RTB_FIGHTER',   0.03),  # 스텝당 3% — 임무 끝난 전투기 공역 이탈
        ('WX_DEVIATION',  0.02),  # 스텝당 2% — 기상 회피 민항기
        ('SQUAWK_7700',   0.005), # 스텝당 0.5% — 비상 선언 (드문 일)
        ('POP_UP_TRAFFIC', 0.04), # 스텝당 4% — 갑자기 나타나는 트래픽
        ('AIRSPACE_HOT',  0.01),  # 스텝당 1% — 공역 HOT 전환
    ]

    @staticmethod
    def inject(all_states_raw, own_idx, device):
        """
        현재 상태에 돌발 이벤트를 주입. 새 항공기 상태를 반환.

        :param all_states_raw: (N, STATE_DIM) 현재 모든 항공기 비정규화 상태
        :param own_idx: 관제 대상 인덱스
        :param device: torch device
        :return: (event_name, new_aircraft_state) or (None, None)
        """
        for event_name, prob in EventInjector.EVENT_TYPES:
            if random.random() > prob:
                continue

            own = all_states_raw[own_idx]
            lat, lon, alt = float(own[0]), float(own[1]), float(own[2])

            if event_name == 'RTB_FIGHTER':
                # 10~20NM 떨어진 곳에서 자기 방향으로 날아오는 전투기
                bearing = random.uniform(0, 360)
                dist_nm = random.uniform(10, 20)
                rad = math.radians(bearing)
                new_lat = lat + dist_nm * math.cos(rad) / 60.0
                cos_lat = math.cos(math.radians(lat))
                new_lon = lon + dist_nm * math.sin(rad) / (60.0 * max(cos_lat, 0.01))
                # 자기 쪽으로 향하는 heading
                inbound_hdg = (bearing + 180) % 360
                new_state = np.array([
                    new_lat, new_lon,
                    alt + random.uniform(-2000, 2000),  # 비슷한 고도
                    random.uniform(350, 500),  # 전투기 속도
                    inbound_hdg,
                    random.uniform(-500, 500),  # 약간의 상승/하강
                    random.uniform(300, 450), 0.8, 0, 0,
                ], dtype=np.float32)
                return event_name, torch.from_numpy(new_state).to(device)

            elif event_name == 'WX_DEVIATION':
                # 15~30NM에서 공역으로 진입하는 민항기 (기상 회피)
                bearing = random.uniform(0, 360)
                dist_nm = random.uniform(15, 30)
                rad = math.radians(bearing)
                new_lat = lat + dist_nm * math.cos(rad) / 60.0
                cos_lat = math.cos(math.radians(lat))
                new_lon = lon + dist_nm * math.sin(rad) / (60.0 * max(cos_lat, 0.01))
                # 원래 항로에서 20~40도 벗어난 heading
                deviated_hdg = random.uniform(0, 360)
                new_state = np.array([
                    new_lat, new_lon,
                    random.uniform(25000, 38000),  # 민항기 순항 고도
                    random.uniform(400, 500),  # 민항기 속도
                    deviated_hdg,
                    0,
                    random.uniform(350, 450), 0.78, 0, 0,
                ], dtype=np.float32)
                return event_name, torch.from_numpy(new_state).to(device)

            elif event_name == 'SQUAWK_7700':
                # 5~15NM 거리에 비상 항공기 출현
                bearing = random.uniform(0, 360)
                dist_nm = random.uniform(5, 15)
                rad = math.radians(bearing)
                new_lat = lat + dist_nm * math.cos(rad) / 60.0
                cos_lat = math.cos(math.radians(lat))
                new_lon = lon + dist_nm * math.sin(rad) / (60.0 * max(cos_lat, 0.01))
                # 비상 항공기: 예측 불가능한 방향, 하강 중일 수 있음
                new_state = np.array([
                    new_lat, new_lon,
                    random.uniform(10000, 30000),
                    random.uniform(200, 400),
                    random.uniform(0, 360),
                    random.uniform(-2000, 0),  # 하강 경향
                    random.uniform(200, 350), 0.6, 0, 0,
                ], dtype=np.float32)
                return event_name, torch.from_numpy(new_state).to(device)

            elif event_name == 'POP_UP_TRAFFIC':
                # 가까이에서 갑자기 나타남 (5~10NM)
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
                    random.uniform(0, 360),
                    0,
                    random.uniform(200, 400), 0.7, 0, 0,
                ], dtype=np.float32)
                return event_name, torch.from_numpy(new_state).to(device)

            elif event_name == 'AIRSPACE_HOT':
                # 공역 전환으로 2~4대가 한꺼번에 나타남
                # 단일 항공기로 표현 (대표)
                bearing = random.uniform(0, 360)
                dist_nm = random.uniform(8, 15)
                rad = math.radians(bearing)
                new_lat = lat + dist_nm * math.cos(rad) / 60.0
                cos_lat = math.cos(math.radians(lat))
                new_lon = lon + dist_nm * math.sin(rad) / (60.0 * max(cos_lat, 0.01))
                new_state = np.array([
                    new_lat, new_lon,
                    random.uniform(8000, 15000),  # 군용 고도
                    random.uniform(350, 500),
                    random.uniform(0, 360),
                    0,
                    random.uniform(300, 450), 0.8, 0, 0,
                ], dtype=np.float32)
                return event_name, torch.from_numpy(new_state).to(device)

        return None, None


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
                 imagination_horizon=15, gamma=0.99, actor_lr=3e-4, critic_lr=3e-4):
        self.world_model = world_model
        self.device = torch.device(device)
        self.world_model.to(self.device)
        self.world_model.eval()  # WM은 별도 학습

        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.target_critic = Critic().to(self.device)
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
        World Model 안에서 imagination rollout.

        1. 초기 상태에서 WM의 hidden state를 warm up
        2. horizon 스텝 동안:
           - Actor가 행동 선택
           - 행동을 상태에 반영 (delta_hdg/alt/spd)
           - WM prior로 다음 상태 예측
           - 보상 계산
        3. 사고 발생 시 큰 negative → Critic이 학습

        Returns: (states, actions, rewards, values) 시퀀스
        """
        B, K, D = initial_states_norm.shape

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

        for step in range(self.horizon):
            # 각 항공기에 대해 traffic 추출 + 행동 선택
            # 간소화: 배치 내 첫 번째를 "관제 대상"으로
            own_state = current_state_raw  # (B, D)

            # 주변 항공기 = 배치 내 다른 항공기 (간소화)
            # 실제로는 all_states에서 가까운 5대를 추출해야 하지만,
            # 학습 효율을 위해 빈 traffic으로 시작 후 개선
            traffic = torch.zeros(B, 5, STATE_DIM, device=self.device)
            if B > 1:
                for i in range(min(B, 5)):
                    idx = (torch.arange(B, device=self.device) + i + 1) % B
                    traffic[:, i] = current_state_raw[idx]

            # Critic value
            with torch.no_grad():
                value = self.target_critic(own_state, traffic)
            all_values.append(value)

            # Actor action
            action = self.actor.get_action(own_state, traffic)
            all_actions.append(action)

            # 행동을 상태에 반영
            deltas = torch.zeros(B, STATE_DIM, device=self.device)
            for i in range(B):
                dh, da, ds = action_to_deltas(action[i].item())
                deltas[i, 4] = dh    # track delta
                deltas[i, 2] = da    # alt delta
                deltas[i, 3] = ds    # gs delta

            modified_state_raw = current_state_raw.clone()
            modified_state_raw[:, 4] = (modified_state_raw[:, 4] + deltas[:, 4]) % 360
            modified_state_raw[:, 2] = (modified_state_raw[:, 2] + deltas[:, 2]).clamp(2000, 45000)
            modified_state_raw[:, 3] = (modified_state_raw[:, 3] + deltas[:, 3]).clamp(200, 600)

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

            # ── 돌발 이벤트 주입 ──
            # 각 배치 항목에 대해 독립적으로 이벤트 발생 여부 판단
            injected_traffic = []
            for i in range(B):
                evt, new_ac = EventInjector.inject(
                    next_state_raw.detach(), i, self.device)
                if new_ac is not None:
                    injected_traffic.append(new_ac)

            # 주입된 항공기를 보상 계산에 포함
            extra_traffic = (torch.stack(injected_traffic, dim=0)
                             if injected_traffic else None)

            # 보상 계산
            rewards = torch.zeros(B, device=self.device)
            for i in range(B):
                others = torch.cat([next_state_raw[:i], next_state_raw[i+1:]], dim=0)
                # 돌발 이벤트 항공기도 포함
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
        with torch.no_grad():
            final_value = self.target_critic(current_state_raw, traffic)

        return {
            'states': torch.stack(all_states, dim=1),      # (B, H+1, D)
            'actions': torch.stack(all_actions, dim=1),     # (B, H)
            'rewards': torch.stack(all_rewards, dim=1),     # (B, H)
            'values': torch.stack(all_values, dim=1),       # (B, H)
            'final_value': final_value,                     # (B,)
        }

    def compute_returns(self, rewards, values, final_value, lam=0.95):
        """GAE-style lambda returns"""
        H = rewards.shape[1]
        returns = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(H)):
            next_v = final_value if t == H - 1 else values[:, t + 1]
            delta = rewards[:, t] + self.gamma * next_v - values[:, t]
            last_gae = delta + self.gamma * lam * last_gae
            returns[:, t] = last_gae + values[:, t]

        return returns

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

        # ── Critic 업데이트 ──
        states_flat = rollout['states'][:, :-1].reshape(-1, STATE_DIM)  # (B*H, D)
        traffic_flat = torch.zeros(states_flat.shape[0], 5, STATE_DIM,
                                   device=self.device)
        returns_flat = returns.reshape(-1)

        critic_pred = self.critic(states_flat, traffic_flat)
        critic_loss = F.mse_loss(critic_pred, returns_flat.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_opt.step()

        # ── Actor 업데이트 ──
        # Actor가 Critic의 value를 최대화하도록
        own_states = rollout['states'][:, :-1]  # (B, H, D)
        B, H, D = own_states.shape
        own_flat = own_states.reshape(B * H, D)
        traffic_flat_a = torch.zeros(B * H, 5, STATE_DIM, device=self.device)

        logits = self.actor(own_flat, traffic_flat_a)  # (B*H, A)
        dist = torch.distributions.Categorical(logits=logits)
        actions_flat = rollout['actions'].reshape(B * H)
        log_probs = dist.log_prob(actions_flat)

        # Advantage
        with torch.no_grad():
            values_flat = self.critic(own_flat, traffic_flat_a)
        advantages = (returns_flat - values_flat).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs * advantages).mean()
        entropy = dist.entropy().mean()
        actor_loss = actor_loss - 0.01 * entropy  # entropy bonus

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()

        # Target critic soft update
        tau = 0.005
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        # 통계
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

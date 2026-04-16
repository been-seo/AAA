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

# 군용 기지 좌표 (목적지 할당용) — config.py에서 가져올 수 없으므로 직접 정의
_DEST_COORDS = [
    (35.1795, 128.9382),  # Gimhae
    (35.8941, 128.6586),  # Daegu
    (35.1264, 126.8089),  # Gwangju
    (36.7220, 127.4987),  # Cheongju
    (35.0886, 128.0703),  # Sacheon
    (37.4381, 127.9604),  # Wonju
    (35.9879, 129.4204),  # Pohang
    (37.4449, 127.1140),  # Seoul AB
    (35.9038, 126.6158),  # Gunsan AB
    (37.0906, 127.0296),  # Osan AB
    (37.2393, 127.0078),  # Suwon AB
    (36.9965, 127.8849),  # Jungwon AB
    (37.7536, 128.9440),  # Gangneung AB
    (36.6319, 128.3553),  # Yecheon AB
    (36.7039, 126.4864),  # Seosan AB
]

# APP 인계 조건
_APP_HANDOFF_DIST_NM = 30.0
_APP_HANDOFF_ALT_MIN = 7500.0
_APP_HANDOFF_ALT_MAX = 12500.0

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

def compute_reward_batch(states_raw, actions, device,
                         dest_lat=None, dest_lon=None, dest_alt=None,
                         prev_dist=None, is_arrival=None,
                         injected=None, prev_in_danger=None):
    """
    배치 전체 보상을 텐서 연산으로 계산.

    보상 철학:
    - 비행 = 연료 비용 (항상 패널티)
    - 안전 비행 = 연료비 상쇄
    - 자기 잘못으로 위험에 빠짐 = 큰 패널티
    - 돌발 이벤트(inject) 회피 성공 = 보너스
    - 무사 도달 = 강한 보너스

    injected: (B,) bool — 이번 스텝에 돌발 이벤트가 주입된 항목
    prev_in_danger: (B,) bool — 이전 스텝에서 위험 상태였던 항목
    return: (B,) 보상, (B,) 현재 거리, (B,) 현재 위험 상태
    """
    B = states_raw.shape[0]
    lat = states_raw[:, 0]
    lon = states_raw[:, 1]
    alt = states_raw[:, 2]
    gs  = states_raw[:, 3]

    if injected is None:
        injected = torch.zeros(B, dtype=torch.bool, device=device)
    if prev_in_danger is None:
        prev_in_danger = torch.zeros(B, dtype=torch.bool, device=device)

    # ── 공통: 분리 계산 ──
    dlat = (lat.unsqueeze(1) - lat.unsqueeze(0)) * 60.0
    cos_lat = torch.cos(torch.deg2rad(lat)).unsqueeze(1)
    dlon = (lon.unsqueeze(1) - lon.unsqueeze(0)) * 60.0 * cos_lat
    h_dist = torch.sqrt(dlat**2 + dlon**2)
    v_dist = torch.abs(alt.unsqueeze(1) - alt.unsqueeze(0))
    eye_mask = torch.eye(B, device=device, dtype=torch.bool)
    h_dist = h_dist.masked_fill(eye_mask, 1e9)

    ra = (h_dist < 5.0) & (v_dist < 1000)
    ta = (h_dist < 10.0) & (v_dist < 2000) & ~ra
    has_ra = ra.any(dim=1)
    has_ta = ta.any(dim=1)
    in_danger = has_ra | has_ta

    # ══════════════════════════════════
    #   축 1: SAFETY (안전)
    # ══════════════════════════════════
    r_safety = torch.zeros(B, device=device)
    r_safety += 1.0  # 안전 기본 보너스

    # RA 충돌 = 큰 패널티
    r_safety -= has_ra.float() * 50
    # TA 접근 = 중간 패널티
    r_safety -= has_ta.float() * 10
    # 자초한 위험 (inject 아닌데 위험)
    self_caused = in_danger & ~injected
    r_safety -= self_caused.float() * 30
    # 돌발 회피 성공
    evaded = prev_in_danger & injected & ~in_danger
    r_safety += evaded.float() * 40
    # 돌발 사전 대응
    preemptive = injected & ~in_danger & ~prev_in_danger
    r_safety += preemptive.float() * 10
    # 고도 위반
    r_safety -= (alt < 2000).float() * 30
    r_safety -= ((alt >= 2000) & (alt < 5000)).float() * 5

    # ══════════════════════════════════
    #   축 2: EFFICIENCY (효율)
    # ══════════════════════════════════
    r_efficiency = torch.zeros(B, device=device)
    # 연료 소모 (속도 비례)
    r_efficiency -= gs / 600.0
    # 속도 이상
    r_efficiency -= (gs < 150).float() * 10
    r_efficiency -= (gs > 600).float() * 5

    # ══════════════════════════════════
    #   축 3: MISSION (임무 완수)
    # ══════════════════════════════════
    r_mission = torch.zeros(B, device=device)

    # ── 목적지 (mission 축) ──
    cur_dist = None
    if dest_lat is not None:
        d_dlat = (lat - dest_lat) * 60.0
        d_cos = torch.cos(torch.deg2rad(lat)).clamp(min=0.01)
        d_dlon = (lon - dest_lon) * 60.0 * d_cos
        cur_dist = torch.sqrt(d_dlat**2 + d_dlon**2)

        # 멀어지면 mission 패널티, efficiency에도 연료 낭비 반영
        if prev_dist is not None:
            regress = (cur_dist - prev_dist).clamp(min=0)
            r_mission -= regress * 0.5
            r_efficiency -= regress * 0.2

        # 무사 도달 = mission 대보너스
        if is_arrival is not None:
            arr_complete = is_arrival & (cur_dist < _APP_HANDOFF_DIST_NM) & \
                           (alt >= _APP_HANDOFF_ALT_MIN) & (alt <= _APP_HANDOFF_ALT_MAX)
            r_mission += arr_complete.float() * 100.0

            arr_close = is_arrival & (cur_dist < _APP_HANDOFF_DIST_NM)
            arr_alt_bad = arr_close & ((alt < _APP_HANDOFF_ALT_MIN) | (alt > _APP_HANDOFF_ALT_MAX))
            r_mission -= arr_alt_bad.float() * 3.0

            dep_complete = ~is_arrival & (cur_dist < 10.0)
            r_mission += dep_complete.float() * 80.0

    rewards = {'safety': r_safety, 'efficiency': r_efficiency, 'mission': r_mission}
    return rewards, cur_dist, in_danger


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


# ── Multi-Axis Critic (GAN-style 경쟁 학습) ──
#
# 3개 축이 각자 Actor를 자기 방향으로 당기며 경쟁:
#   Safety:     "이 상태가 얼마나 위험한가"
#   Efficiency: "연료를 얼마나 낭비하는가"
#   Mission:    "임무 완수에 얼마나 가까운가"
#
# Actor는 세 축의 균형점(파레토 최적)을 찾아야 함.
# 각 Critic의 출력 = 해당 축의 점수 → demo에서 세부 요인 표시에 직접 사용.

REWARD_AXES = ['safety', 'efficiency', 'mission']

class _CriticHead(nn.Module):
    """단일 축의 Value Function"""
    def __init__(self, state_dim=STATE_DIM, hidden_dim=192):
        super().__init__()
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, 96), nn.ELU(),
            nn.Linear(96, 96), nn.ELU(),
        )
        self.traffic_enc = nn.Sequential(
            nn.Linear(state_dim * 5, 96), nn.ELU(),
            nn.Linear(96, 48), nn.ELU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(96 + 48, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 96), nn.ELU(),
            nn.Linear(96, 1),
        )

    def forward(self, own_state, traffic_states):
        B = own_state.shape[0]
        s = self.state_enc(own_state)
        t = self.traffic_enc(traffic_states.reshape(B, -1))
        return self.value_head(torch.cat([s, t], dim=-1)).squeeze(-1)


class Critic(nn.Module):
    """
    3-Axis Critic: safety / efficiency / mission.

    각 head가 해당 축의 Value를 학습하고, Actor 업데이트 시
    가중합으로 advantage를 계산하되, 가중치가 GAN식으로 경쟁.

    Safety Advisor에서는 각 축의 출력을 직접 위험도/효율/임무 점수로 사용.
    """

    def __init__(self, state_dim=STATE_DIM):
        super().__init__()
        self.safety = _CriticHead(state_dim)
        self.efficiency = _CriticHead(state_dim)
        self.mission = _CriticHead(state_dim)
        self.heads = {'safety': self.safety, 'efficiency': self.efficiency, 'mission': self.mission}

    def forward(self, own_state, traffic_states, axis=None):
        """axis=None이면 3축 가중합 (기본 가중치), axis 지정 시 해당 축만"""
        if axis:
            return self.heads[axis](own_state, traffic_states)
        # 기본: safety 우선 가중합
        s = self.safety(own_state, traffic_states)
        e = self.efficiency(own_state, traffic_states)
        m = self.mission(own_state, traffic_states)
        return s * 0.5 + e * 0.2 + m * 0.3

    def forward_all(self, own_state, traffic_states):
        """3축 모두 반환: dict of (B,) tensors"""
        return {
            'safety': self.safety(own_state, traffic_states),
            'efficiency': self.efficiency(own_state, traffic_states),
            'mission': self.mission(own_state, traffic_states),
        }

    def risk_score(self, own_state, traffic_states):
        """Safety Advisor용: safety V → 위험도 [0,1]"""
        with torch.no_grad():
            v = self.safety(own_state, traffic_states)
            return torch.sigmoid(-v * 0.5).clamp(0, 1)

    def axis_scores(self, own_state, traffic_states):
        """Safety Advisor용: 3축 점수 dict"""
        with torch.no_grad():
            vals = self.forward_all(own_state, traffic_states)
            return {
                'safety': torch.sigmoid(-vals['safety'] * 0.5).clamp(0, 1),
                'efficiency': torch.sigmoid(-vals['efficiency'] * 0.5).clamp(0, 1),
                'mission': torch.sigmoid(-vals['mission'] * 0.5).clamp(0, 1),
            }


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

        # ── 목적지 할당 (배치마다 랜덤 기지) ──
        dest_coords = torch.tensor(_DEST_COORDS, dtype=torch.float32, device=self.device)
        n_dest = dest_coords.shape[0]
        dest_idx = torch.randint(0, n_dest, (B,), device=self.device)
        dest_lat = dest_coords[dest_idx, 0]
        dest_lon = dest_coords[dest_idx, 1]
        # 50% arrival, 50% departure
        is_arrival = torch.rand(B, device=self.device) < 0.5
        # arrival: 접근 고도, departure: 순항 고도
        dest_alt = torch.where(is_arrival,
            torch.tensor(10000.0, device=self.device),
            torch.tensor(20000.0, device=self.device))
        # 초기 거리 계산
        init_dlat = (current_state_raw[:, 0] - dest_lat) * 60.0
        init_cos = torch.cos(torch.deg2rad(current_state_raw[:, 0])).clamp(min=0.01)
        init_dlon = (current_state_raw[:, 1] - dest_lon) * 60.0 * init_cos
        prev_dist = torch.sqrt(init_dlat**2 + init_dlon**2)

        # Imagination rollout
        all_states = [current_state_raw]
        all_actions = []
        all_rewards = {ax: [] for ax in REWARD_AXES}
        all_values = {ax: [] for ax in REWARD_AXES}
        prev_in_danger = torch.zeros(B, dtype=torch.bool, device=self.device)

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

            # Critic value (축별)
            with torch.no_grad():
                vals = self.target_critic.forward_all(own_state_norm, traffic_norm)
            for ax in REWARD_AXES:
                all_values[ax].append(vals[ax])

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

            # 보상 계산 (GPU 벡터화, 목적지 + 돌발 이벤트 회피)
            rewards, cur_dist, in_danger = compute_reward_batch(
                reward_states, action, self.device,
                dest_lat=dest_lat, dest_lon=dest_lon, dest_alt=dest_alt,
                prev_dist=prev_dist, is_arrival=is_arrival,
                injected=inject_mask, prev_in_danger=prev_in_danger)
            if cur_dist is not None:
                prev_dist = cur_dist
            prev_in_danger = in_danger
            for ax in REWARD_AXES:
                all_rewards[ax].append(rewards[ax])

            current_state_raw = next_state_raw
            all_states.append(current_state_raw)

        # 마지막 스텝 value (bootstrap, 축별)
        own_norm_final = self._normalize(current_state_raw)
        traffic_norm_final = own_norm_final[shift_indices]
        with torch.no_grad():
            final_vals = self.target_critic.forward_all(own_norm_final, traffic_norm_final)

        return {
            'states': torch.stack(all_states, dim=1),      # (B, H+1, D)
            'actions': torch.stack(all_actions, dim=1),     # (B, H)
            'rewards': {ax: torch.stack(all_rewards[ax], dim=1) for ax in REWARD_AXES},  # {ax: (B,H)}
            'values': {ax: torch.stack(all_values[ax], dim=1) for ax in REWARD_AXES},    # {ax: (B,H)}
            'final_value': final_vals,                      # {ax: (B,)}
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
        3-Axis GAN-style 학습:
        1. Imagination rollout → 3축 보상
        2. 각 Critic head 독립 학습 (자기 축의 returns)
        3. Actor: 3축 advantage 가중합으로 학습
           가중치는 각 축의 성능(loss)에 반비례 → 못하는 축이 더 강하게 당김
        """
        rollout = self.imagine_rollout(initial_states_norm, initial_contexts)

        # 축별 returns 계산
        returns_per_axis = {}
        for ax in REWARD_AXES:
            returns_per_axis[ax] = self.compute_returns(
                rollout['rewards'][ax], rollout['values'][ax], rollout['final_value'][ax])

        # 공통 준비
        states_raw_flat = rollout['states'][:, :-1].reshape(-1, STATE_DIM)
        states_flat = (states_raw_flat - self.norm_mean) / self.norm_std
        traffic_flat = torch.zeros(states_flat.shape[0], 5, STATE_DIM, device=self.device)

        own_states = rollout['states'][:, :-1]
        B, H, D = own_states.shape
        own_flat = ((own_states - self.norm_mean) / self.norm_std).reshape(B * H, D)
        traffic_flat_a = torch.zeros(B * H, 5, STATE_DIM, device=self.device)

        # ── 각 Critic head 독립 학습 ──
        critic_losses = {}
        symlog_targets_per_axis = {}
        for ax in REWARD_AXES:
            targets = torch.nan_to_num(returns_per_axis[ax].reshape(-1).detach(), nan=0.0).clamp(-10, 10)
            symlog_targets_per_axis[ax] = targets
            pred = self.critic.heads[ax](states_flat, traffic_flat)
            loss = F.mse_loss(pred, targets)
            critic_losses[ax] = loss.item()

            if torch.isfinite(loss):
                self.critic_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_opt.step()

        # ── GAN-style 가중치: 못하는 축이 더 강하게 당김 ──
        with torch.no_grad():
            losses = torch.tensor([critic_losses[ax] for ax in REWARD_AXES], device=self.device)
            losses = losses.clamp(min=0.1)
            # 높은 loss = 못하는 축 = 더 높은 가중치
            weights = losses / losses.sum()
            # safety는 최소 보장 (하한 40%)
            w_dict = {ax: weights[i].item() for i, ax in enumerate(REWARD_AXES)}
            if w_dict['safety'] < 0.4:
                deficit = 0.4 - w_dict['safety']
                w_dict['safety'] = 0.4
                # 다른 축에서 비례 감소
                other_sum = sum(w_dict[a] for a in REWARD_AXES if a != 'safety')
                if other_sum > 0:
                    for a in REWARD_AXES:
                        if a != 'safety':
                            w_dict[a] -= deficit * (w_dict[a] / other_sum)

        # ── Actor 업데이트: 3축 가중합 advantage ──
        logits = self.actor(own_flat, traffic_flat_a)
        logits = torch.nan_to_num(logits, nan=0.0).clamp(-20, 20)
        dist = torch.distributions.Categorical(logits=logits)
        actions_flat = rollout['actions'].reshape(B * H)
        log_probs = dist.log_prob(actions_flat)

        # 축별 advantage 계산 후 가중합
        total_advantage = torch.zeros(B * H, device=self.device)
        for ax in REWARD_AXES:
            with torch.no_grad():
                v = self.critic.heads[ax](own_flat, traffic_flat_a)
            adv = (symlog_targets_per_axis[ax] - v).detach()
            adv = torch.nan_to_num(adv, nan=0.0)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv = adv.clamp(-5, 5)
            total_advantage += adv * w_dict[ax]

        actor_loss = -(log_probs * total_advantage).mean()
        entropy = dist.entropy().mean()
        actor_loss = actor_loss - 0.03 * entropy

        if torch.isfinite(actor_loss):
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

        # Target critic soft update
        tau = 0.02
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        # 통계
        safety_r = rollout['rewards']['safety'].sum(dim=1).mean().item()
        crashes = (rollout['rewards']['safety'] < -30).any(dim=1).sum().item()
        self.total_episodes += B
        self.total_crashes += crashes
        self.total_safe += B - crashes

        mean_v = {ax: rollout['values'][ax].mean().item() for ax in REWARD_AXES}

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': sum(critic_losses.values()) / 3,
            'c_safety': critic_losses['safety'],
            'c_efficiency': critic_losses['efficiency'],
            'c_mission': critic_losses['mission'],
            'mean_reward': safety_r,
            'mean_value': mean_v['safety'],
            'v_safety': mean_v['safety'],
            'v_efficiency': mean_v['efficiency'],
            'v_mission': mean_v['mission'],
            'w_safety': w_dict['safety'],
            'w_efficiency': w_dict['efficiency'],
            'w_mission': w_dict['mission'],
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

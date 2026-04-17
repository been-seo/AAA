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
# HDG: 0, 10, 20, ..., 350 (36개) — 목표 방위 직접 지정
# ALT: 7500, 8500, ..., 44500 (38개) — 목표 고도 직접 지정
# SPD: -50, 0, +50 (3개) — 속도 변경
# RATE: normal, quick (2개) — 상승/하강률
# HOLD: 1개
# 총: 36 * 38 * 3 * 2 + 1 = 8209
HDG_TARGETS = list(range(0, 360, 10))       # [0, 10, 20, ..., 350]
ALT_TARGETS = list(range(7500, 45000, 1000)) # [7500, 8500, ..., 44500]
SPD_DELTAS = [-50, 0, 50]
RATE_OPTIONS = [0, 1]  # 0=normal, 1=quick
NUM_ACTIONS = len(HDG_TARGETS) * len(ALT_TARGETS) * len(SPD_DELTAS) * len(RATE_OPTIONS) + 1

def action_to_instruction(action_idx):
    """이산 행동 → (hdg_target, alt_target, delta_spd, quick) — 단건 조회용"""
    if action_idx == NUM_ACTIONS - 1:
        return None  # HOLD
    n_rate = len(RATE_OPTIONS)
    n_spd = len(SPD_DELTAS)
    n_alt = len(ALT_TARGETS)
    rate_i = action_idx % n_rate
    spd_i = (action_idx // n_rate) % n_spd
    alt_i = (action_idx // (n_rate * n_spd)) % n_alt
    hdg_i = action_idx // (n_rate * n_spd * n_alt)
    return (HDG_TARGETS[hdg_i], ALT_TARGETS[alt_i], SPD_DELTAS[spd_i], RATE_OPTIONS[rate_i])


# ── 배치 행동 룩업 테이블 (GPU 벡터화) ──
def _build_action_table(device):
    """모든 행동의 (hdg_target, alt_target, delta_spd, quick) 테이블. (NUM_ACTIONS, 4)"""
    table = []
    for h in HDG_TARGETS:
        for a in ALT_TARGETS:
            for s in SPD_DELTAS:
                for r in RATE_OPTIONS:
                    table.append([h, a, s, r])
    table.append([0, 0, 0, 0])  # HOLD (현재 유지)
    return torch.tensor(table, dtype=torch.float32, device=device)


# ── 벡터화 보상 함수 (전체 배치를 GPU에서 한번에) ──

def compute_reward_episode(own_state, traffic_states, action, device,
                           dest_lat=None, dest_lon=None, dest_alt=None,
                           prev_dist=None, is_arrival=None,
                           injected=None, prev_in_danger=None):
    """
    에피소드 기반 보상 계산: 내 항공기(own) vs 주변 트래픽(traffic).

    own_state: (B, STATE_DIM) 내 항공기 상태
    traffic_states: (B, N_traffic, STATE_DIM) 주변 트래픽 상태
    action: (B,) 행동 인덱스
    injected: (B,) 돌발 이벤트 주입 여부
    prev_in_danger: (B,) 이전 위험 상태
    return: rewards dict, cur_dist, in_danger
    """
    B = own_state.shape[0]
    my_lat = own_state[:, 0]
    my_lon = own_state[:, 1]
    my_alt = own_state[:, 2]
    my_gs  = own_state[:, 3]

    if injected is None:
        injected = torch.zeros(B, dtype=torch.bool, device=device)
    if prev_in_danger is None:
        prev_in_danger = torch.zeros(B, dtype=torch.bool, device=device)

    N_t = traffic_states.shape[1]

    # ── 내 항공기 vs 각 트래픽 거리 계산: (B, N_traffic) ──
    t_lat = traffic_states[:, :, 0]  # (B, N)
    t_lon = traffic_states[:, :, 1]
    t_alt = traffic_states[:, :, 2]

    dlat = (my_lat.unsqueeze(1) - t_lat) * 60.0           # (B, N)
    cos_lat = torch.cos(torch.deg2rad(my_lat)).unsqueeze(1).clamp(min=0.01)
    dlon = (my_lon.unsqueeze(1) - t_lon) * 60.0 * cos_lat  # (B, N)
    h_dist = torch.sqrt(dlat**2 + dlon**2)                  # (B, N) NM
    v_dist = torch.abs(my_alt.unsqueeze(1) - t_alt)         # (B, N) ft

    # 유효 트래픽 마스크 (lat=0, lon=0인 빈 슬롯 제외)
    valid = (t_lat.abs() > 0.1) & (t_lon.abs() > 0.1)
    h_dist = h_dist.masked_fill(~valid, 1e9)

    ra = (h_dist < 5.0) & (v_dist < 1000)    # RA급
    ta = (h_dist < 10.0) & (v_dist < 2000) & ~ra  # TA급
    has_ra = ra.any(dim=1)   # (B,)
    has_ta = ta.any(dim=1)
    in_danger = has_ra | has_ta

    # ══════════════════════════════════
    #   축 1: SAFETY
    # ══════════════════════════════════
    r_safety = torch.zeros(B, device=device)
    r_safety -= has_ra.float() * 100
    r_safety -= has_ta.float() * 30
    self_caused = in_danger & ~injected
    r_safety -= self_caused.float() * 60
    evaded = prev_in_danger & injected & ~in_danger
    r_safety += evaded.float() * 40
    preemptive = injected & ~in_danger & ~prev_in_danger
    r_safety += preemptive.float() * 10
    r_safety -= (my_alt < 2000).float() * 60
    r_safety -= ((my_alt >= 2000) & (my_alt < 5000)).float() * 15

    # ══════════════════════════════════
    #   축 2: EFFICIENCY
    # ══════════════════════════════════
    r_efficiency = torch.zeros(B, device=device)
    r_efficiency -= my_gs / 600.0
    r_efficiency -= (my_gs < 150).float() * 10
    r_efficiency -= (my_gs > 600).float() * 5

    # ══════════════════════════════════
    #   축 3: MISSION
    # ══════════════════════════════════
    r_mission = torch.zeros(B, device=device)

    # ── 목적지 (mission 축) ──
    cur_dist = None
    if dest_lat is not None:
        d_dlat = (my_lat - dest_lat) * 60.0
        d_cos = torch.cos(torch.deg2rad(my_lat)).clamp(min=0.01)
        d_dlon = (my_lon - dest_lon) * 60.0 * d_cos
        cur_dist = torch.sqrt(d_dlat**2 + d_dlon**2)

        if prev_dist is not None:
            regress = (cur_dist - prev_dist).clamp(min=0)
            r_mission -= regress * 0.5
            r_efficiency -= regress * 0.2

        if is_arrival is not None:
            arr_complete = is_arrival & (cur_dist < _APP_HANDOFF_DIST_NM) & \
                           (my_alt >= _APP_HANDOFF_ALT_MIN) & (my_alt <= _APP_HANDOFF_ALT_MAX)
            r_mission += arr_complete.float() * 100.0

            arr_close = is_arrival & (cur_dist < _APP_HANDOFF_DIST_NM)
            arr_alt_bad = arr_close & ((my_alt < _APP_HANDOFF_ALT_MIN) | (my_alt > _APP_HANDOFF_ALT_MAX))
            r_mission -= arr_alt_bad.float() * 3.0

            dep_complete = ~is_arrival & (cur_dist < 10.0)
            r_mission += dep_complete.float() * 80.0

    rewards = {'safety': r_safety, 'efficiency': r_efficiency, 'mission': r_mission}
    return rewards, cur_dist, in_danger


# ── Actor (Policy Network) ──

class Actor(nn.Module):
    """관제 정책: 상태 → 행동 확률분포"""

    def __init__(self, state_dim=STATE_DIM, hidden_dim=512):
        super().__init__()
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
        )
        self.traffic_enc = nn.Sequential(
            nn.Linear(state_dim * 5, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(256 + 128, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, NUM_ACTIONS),
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
        """Safety Advisor용: safety V → 위험도 [0,1].
        safety는 raw space: 0=안전, -100=RA충돌. sigmoid(-V * 0.05)로 변환.
          V=0 → 0.50, V=-20 → 0.73, V=-50 → 0.92, V=-100 → 0.99
        """
        with torch.no_grad():
            v = self.safety(own_state, traffic_states)
            return torch.sigmoid(-v * 0.05).clamp(0, 1)

    def axis_scores(self, own_state, traffic_states):
        """Safety Advisor용: 3축 점수 dict"""
        with torch.no_grad():
            vals = self.forward_all(own_state, traffic_states)
            return {
                'safety': torch.sigmoid(-vals['safety'] * 0.05).clamp(0, 1),
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
        # 각 Critic head에 독립 optimizer (gradient 간섭 방지)
        self.critic_opts = {
            ax: optim.Adam(self.critic.heads[ax].parameters(), lr=critic_lr, eps=1e-5)
            for ax in REWARD_AXES
        }

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

    def imagine_rollout(self, initial_states_norm, initial_contexts, traffic_env=None):
        """
        에피소드 기반 imagination rollout.

        각 배치 항목 = 1개 에피소드:
          - initial_states_norm[:, -1] = 내 항공기 초기 상태
          - traffic_env: (B, H, N_traffic, STATE_DIM) 각 스텝별 주변 트래픽 상태
            None이면 initial_contexts에서 트래픽을 추출하여 고정 사용

        내 항공기만 Actor가 조종. 트래픽은 환경 데이터대로 움직임.
        충돌 = 내 항공기 vs 트래픽.
        """
        B, K, D = initial_states_norm.shape

        if not hasattr(self, '_action_table'):
            self._action_table = _build_action_table(self.device)

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
                gru_in = torch.cat([state_emb, interaction, z], dim=-1).unsqueeze(1)
                gru_out, h = self.world_model.gru(gru_in, h)
                h_t = gru_out.squeeze(1)
                post_mean, post_logstd = self.world_model._posterior(h_t, state_emb)
                z = self.world_model._sample_z(post_mean, post_logstd)

        # 내 항공기 초기 상태 (비정규화)
        current_own_raw = self._denormalize(initial_states_norm[:, -1])  # (B, D)

        # 트래픽 환경 구성
        # traffic_env가 없으면 initial_contexts의 마지막 스텝에서 트래픽 추출하여 고정
        N_traffic = 5
        if traffic_env is not None:
            # (B, H, N_traffic, STATE_DIM) — 스텝별로 다른 트래픽
            pass
        else:
            # context (B, total_steps, MAX_NEIGHBORS, CONTEXT_DIM)에서 트래픽 상태 추정
            # context는 상대 좌표이므로, 절대 좌표로 복원
            # 간단히: 내 위치 + context 상대좌표 → 트래픽 절대 좌표
            last_ctx = initial_contexts[:, -1]  # (B, MAX_NEIGHBORS, CONTEXT_DIM)
            # CONTEXT_DIM = [dx_nm, dy_nm, dalt_kft, dgs_100, dtrack_180, dvrate_1000]
            my_lat = current_own_raw[:, 0]  # (B,)
            my_lon = current_own_raw[:, 1]
            my_alt = current_own_raw[:, 2]
            my_gs  = current_own_raw[:, 3]
            my_trk = current_own_raw[:, 4]

            cos_lat = torch.cos(torch.deg2rad(my_lat)).clamp(min=0.01)
            t_lat = my_lat.unsqueeze(1) + last_ctx[:, :N_traffic, 1] / 60.0   # dy_nm → lat
            t_lon = my_lon.unsqueeze(1) + last_ctx[:, :N_traffic, 0] / (60.0 * cos_lat.unsqueeze(1))  # dx_nm → lon
            t_alt = my_alt.unsqueeze(1) + last_ctx[:, :N_traffic, 2] * 1000.0  # dalt_kft → ft
            t_gs  = my_gs.unsqueeze(1) + last_ctx[:, :N_traffic, 3] * 100.0    # dgs_100
            t_trk = my_trk.unsqueeze(1) + last_ctx[:, :N_traffic, 4] * 180.0   # dtrack_180
            t_vr  = last_ctx[:, :N_traffic, 5] * 1000.0

            # 안전 거리 필터: 15NM 미만 트래픽 제거 (안전한 상태에서 시작)
            init_dlat_t = (my_lat.unsqueeze(1) - t_lat) * 60.0
            init_dlon_t = (my_lon.unsqueeze(1) - t_lon) * 60.0 * cos_lat.unsqueeze(1)
            init_dist_t = torch.sqrt(init_dlat_t**2 + init_dlon_t**2)  # (B, N_traffic)
            too_close = init_dist_t < 15.0  # 15NM 미만은 제거
            t_lat = t_lat.masked_fill(too_close, 0.0)
            t_lon = t_lon.masked_fill(too_close, 0.0)
            t_alt = t_alt.masked_fill(too_close, 0.0)
            t_gs  = t_gs.masked_fill(too_close, 0.0)
            t_trk = t_trk.masked_fill(too_close, 0.0)
            t_vr  = t_vr.masked_fill(too_close, 0.0)

            # (B, N_traffic, STATE_DIM) 트래픽 상태 텐서
            zeros = torch.zeros(B, N_traffic, device=self.device)
            static_traffic = torch.stack([
                t_lat, t_lon, t_alt, t_gs, t_trk % 360, t_vr,
                t_gs - 30,  # IAS 근사
                torch.full_like(zeros, 0.75),  # mach 근사
                zeros, zeros  # wind
            ], dim=-1)  # (B, N_traffic, STATE_DIM)

            # 매 스텝 트래픽 = 직선 외삽 (10초 간격)
            traffic_env = torch.zeros(B, self.horizon, N_traffic, STATE_DIM, device=self.device)
            for step in range(self.horizon):
                dt = (step + 1) * 10.0  # 초
                moved = static_traffic.clone()
                spd_nm_s = moved[:, :, 3] * 0.000514444 * 1.852 / 1000  # kt → NM/s 근사
                hdg_rad = torch.deg2rad(moved[:, :, 4])
                moved[:, :, 0] += spd_nm_s * dt * torch.cos(hdg_rad) / 60.0
                c_lat = torch.cos(torch.deg2rad(moved[:, :, 0])).clamp(min=0.01)
                moved[:, :, 1] += spd_nm_s * dt * torch.sin(hdg_rad) / (60.0 * c_lat)
                moved[:, :, 2] += moved[:, :, 5] / 60.0 * dt  # vrate
                traffic_env[:, step] = moved

        # ── 목적지 할당 ──
        dest_coords = torch.tensor(_DEST_COORDS, dtype=torch.float32, device=self.device)
        n_dest = dest_coords.shape[0]
        dest_idx = torch.randint(0, n_dest, (B,), device=self.device)
        dest_lat = dest_coords[dest_idx, 0]
        dest_lon = dest_coords[dest_idx, 1]
        is_arrival = torch.rand(B, device=self.device) < 0.5
        dest_alt = torch.where(is_arrival,
            torch.tensor(10000.0, device=self.device),
            torch.tensor(20000.0, device=self.device))
        init_dlat = (current_own_raw[:, 0] - dest_lat) * 60.0
        init_cos = torch.cos(torch.deg2rad(current_own_raw[:, 0])).clamp(min=0.01)
        init_dlon = (current_own_raw[:, 1] - dest_lon) * 60.0 * init_cos
        prev_dist = torch.sqrt(init_dlat**2 + init_dlon**2)

        # Imagination rollout
        all_states = [current_own_raw]
        all_actions = []
        all_traffic = []  # 매 스텝 트래픽 저장 (Critic 학습용)
        all_rewards = {ax: [] for ax in REWARD_AXES}
        all_values = {ax: [] for ax in REWARD_AXES}
        prev_in_danger = torch.zeros(B, dtype=torch.bool, device=self.device)

        empty_ctx = torch.zeros(B, MAX_NEIGHBORS, CONTEXT_DIM, device=self.device)

        for step in range(self.horizon):
            own_norm = self._normalize(current_own_raw)  # (B, D)

            # 이 스텝의 트래픽 (정규화)
            step_traffic_raw = traffic_env[:, step]  # (B, N_traffic, D)
            step_traffic_norm = (step_traffic_raw - self.norm_mean) / self.norm_std
            all_traffic.append(step_traffic_norm)

            # Critic value
            with torch.no_grad():
                vals = self.target_critic.forward_all(own_norm, step_traffic_norm)
            for ax in REWARD_AXES:
                all_values[ax].append(vals[ax])

            # Actor action (내 항공기 기준, 트래픽 참조)
            action = self.actor.get_action(own_norm, step_traffic_norm)
            all_actions.append(action)

            # 내 항공기에 행동 반영 (target 방식)
            # action_table: (hdg_target, alt_target, delta_spd, quick)
            act_params = self._action_table[action]  # (B, 4)
            is_hold = (action == NUM_ACTIONS - 1)
            modified_raw = current_own_raw.clone()
            # HDG: 목표 방위로 설정 (HOLD면 현재 유지)
            modified_raw[:, 4] = torch.where(is_hold, modified_raw[:, 4], act_params[:, 0])
            # ALT: 목표 고도로 설정
            modified_raw[:, 2] = torch.where(is_hold, modified_raw[:, 2], act_params[:, 1].clamp(2000, 45000))
            # SPD: delta 적용
            modified_raw[:, 3] = (modified_raw[:, 3] + act_params[:, 2]).clamp(200, 600)

            # WM으로 내 항공기 다음 상태 예측
            modified_norm = self._normalize(modified_raw)
            with torch.no_grad():
                state_emb = self.world_model._encode_state(modified_norm)
                interaction = self.world_model.neighbor_attn(state_emb, empty_ctx)
                gru_in = torch.cat([state_emb, interaction, z], dim=-1).unsqueeze(1)
                gru_out, h = self.world_model.gru(gru_in, h)
                h_t = gru_out.squeeze(1)
                prior_mean, prior_logstd = self.world_model._prior(h_t)
                z = self.world_model._sample_z(prior_mean, prior_logstd)
                pred_mean, pred_logstd = self.world_model._decode_state(h_t, z)
                next_own_norm = self.world_model._sample_z(pred_mean, pred_logstd)

            next_own_raw = self._denormalize(next_own_norm)

            # 돌발 이벤트: 트래픽에 추가 (내 항공기 근처에 갑자기 나타남)
            injected_traffic, inject_mask = inject_events_batch(next_own_raw, self.device)
            # inject된 항공기를 트래픽 마지막 슬롯에 추가
            reward_traffic = step_traffic_raw.clone()
            # inject_mask가 True인 배치에 대해 첫 번째 트래픽 슬롯을 주입 항공기로 교체
            if inject_mask.any():
                injected_state = torch.stack([
                    injected_traffic[:, 0], injected_traffic[:, 1], injected_traffic[:, 2],
                    injected_traffic[:, 3], injected_traffic[:, 4], injected_traffic[:, 5],
                    injected_traffic[:, 3] - 30, torch.full((B,), 0.75, device=self.device),
                    torch.zeros(B, device=self.device), torch.zeros(B, device=self.device),
                ], dim=-1)  # (B, STATE_DIM)
                reward_traffic[inject_mask, 0] = injected_state[inject_mask]

            # 보상: 내 항공기 vs 트래픽
            rewards, cur_dist, in_danger = compute_reward_episode(
                next_own_raw, reward_traffic, action, self.device,
                dest_lat=dest_lat, dest_lon=dest_lon, dest_alt=dest_alt,
                prev_dist=prev_dist, is_arrival=is_arrival,
                injected=inject_mask, prev_in_danger=prev_in_danger)
            if cur_dist is not None:
                prev_dist = cur_dist
            prev_in_danger = in_danger
            for ax in REWARD_AXES:
                all_rewards[ax].append(rewards[ax])

            current_own_raw = next_own_raw
            all_states.append(current_own_raw)

        # 마지막 value (bootstrap)
        own_norm_final = self._normalize(current_own_raw)
        final_traffic_norm = (traffic_env[:, -1] - self.norm_mean) / self.norm_std
        with torch.no_grad():
            final_vals = self.target_critic.forward_all(own_norm_final, final_traffic_norm)

        return {
            'states': torch.stack(all_states, dim=1),
            'actions': torch.stack(all_actions, dim=1),
            'traffic': torch.stack(all_traffic, dim=1),  # (B, H, N_traffic, D) 정규화됨
            'rewards': {ax: torch.stack(all_rewards[ax], dim=1) for ax in REWARD_AXES},
            'values': {ax: torch.stack(all_values[ax], dim=1) for ax in REWARD_AXES},
            'final_value': final_vals,
        }

    def compute_returns(self, rewards, values, final_value, lam=0.95, use_symlog=True):
        """
        GAE-style lambda returns.
        use_symlog=True: symlog space에서 계산 (efficiency, mission)
        use_symlog=False: raw space에서 직접 계산 (safety — 패널티 크기 보존)
        """
        H = rewards.shape[1]

        if use_symlog:
            rewards_t = symlog(rewards)
        else:
            rewards_t = rewards

        returns = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(H)):
            next_v = final_value if t == H - 1 else values[:, t + 1]
            delta = rewards_t[:, t] + self.gamma * next_v - values[:, t]
            last_gae = delta + self.gamma * lam * last_gae
            returns[:, t] = last_gae + values[:, t]

        if use_symlog:
            return returns.clamp(-10, 10)
        else:
            return returns.clamp(-200, 200)

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
        # safety: raw space (패널티 크기를 압축 없이 직접 반영)
        # efficiency/mission: symlog space (기존)
        returns_per_axis = {}
        for ax in REWARD_AXES:
            returns_per_axis[ax] = self.compute_returns(
                rollout['rewards'][ax], rollout['values'][ax], rollout['final_value'][ax],
                use_symlog=(ax != 'safety'))

        # 공통 준비
        own_states = rollout['states'][:, :-1]  # (B, H, D) raw
        B, H, D = own_states.shape
        own_flat = ((own_states - self.norm_mean) / self.norm_std).reshape(B * H, D)

        # 실제 트래픽 (rollout에서 저장된 정규화 텐서)
        traffic_all = rollout['traffic']  # (B, H, N_traffic, STATE_DIM) 정규화됨
        N_t = traffic_all.shape[2]
        traffic_flat = traffic_all.reshape(B * H, N_t, STATE_DIM)  # (B*H, N_traffic, D)

        # ── 각 Critic head 독립 학습 (독립 optimizer) ──
        critic_losses = {}
        targets_per_axis = {}
        for ax in REWARD_AXES:
            raw_targets = returns_per_axis[ax].reshape(-1).detach()
            if ax == 'safety':
                # raw space: 클램프만 적용 (symlog 없음)
                targets = torch.nan_to_num(raw_targets, nan=0.0).clamp(-200, 200)
            else:
                # symlog space
                targets = torch.nan_to_num(raw_targets, nan=0.0).clamp(-10, 10)
            targets_per_axis[ax] = targets
            pred = self.critic.heads[ax](own_flat, traffic_flat)
            loss = F.mse_loss(pred, targets)
            critic_losses[ax] = loss.item()

            if torch.isfinite(loss):
                self.critic_opts[ax].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.heads[ax].parameters(), 1.0)
                self.critic_opts[ax].step()

        # ── GAN-style 가중치: 못하는 축이 더 강하게 당김 ──
        with torch.no_grad():
            losses = torch.tensor([critic_losses[ax] for ax in REWARD_AXES], device=self.device)
            losses = losses.clamp(min=0.1)
            # 높은 loss = 못하는 축 = 더 높은 가중치
            weights = losses / losses.sum()
            # 가중치 하한/상한: safety 40~80%, efficiency/mission 각 최소 10%
            w_dict = {ax: weights[i].item() for i, ax in enumerate(REWARD_AXES)}
            w_dict['safety'] = max(0.4, min(0.8, w_dict['safety']))
            w_dict['efficiency'] = max(0.1, w_dict['efficiency'])
            w_dict['mission'] = max(0.1, w_dict['mission'])
            # 정규화
            total = sum(w_dict.values())
            w_dict = {ax: v / total for ax, v in w_dict.items()}

        # ── Actor 업데이트: 3축 가중합 advantage ──
        logits = self.actor(own_flat, traffic_flat)
        logits = torch.nan_to_num(logits, nan=0.0).clamp(-20, 20)
        dist = torch.distributions.Categorical(logits=logits)
        actions_flat = rollout['actions'].reshape(B * H)
        log_probs = dist.log_prob(actions_flat)

        # 축별 advantage 계산 후 가중합
        total_advantage = torch.zeros(B * H, device=self.device)
        for ax in REWARD_AXES:
            with torch.no_grad():
                v = self.critic.heads[ax](own_flat, traffic_flat)
            adv = (targets_per_axis[ax] - v).detach()
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
            'critic_opts': {ax: opt.state_dict() for ax, opt in self.critic_opts.items()},
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
            if 'critic_opts' in ckpt:
                for ax, state in ckpt['critic_opts'].items():
                    self.critic_opts[ax].load_state_dict(state)
            elif 'critic_opt' in ckpt:
                pass  # 이전 형식 호환 — 무시하고 새 optimizer 사용
            print(f"[Dreamer] Loaded actor/critic from checkpoint")
        else:
            print(f"[Dreamer] No actor/critic in checkpoint - starting fresh")
        self.total_episodes = ckpt.get('total_episodes', 0)
        self.total_crashes = ckpt.get('total_crashes', 0)
        self.total_safe = ckpt.get('total_safe', 0)

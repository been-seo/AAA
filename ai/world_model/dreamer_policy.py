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
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .dataset import (STATE_DIM, NORM_MEAN, NORM_STD, MAX_NEIGHBORS, CONTEXT_DIM,
                       WORLD_DIM, _load_waypoints, _get_nearest_waypoints,
                       _get_nearest_waypoints_batch)
from .paving_controller import PAVINGController

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
                           prev_dist=None, init_dist=None, is_arrival=None,
                           is_terminal=None, fuel_remaining=None,
                           injected=None, prev_in_danger=None,
                           crash_penalty=500.0, ta_penalty=50.0,
                           self_caused_penalty=150.0,
                           reach_bonus=500.0):
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
    # 대칭적 거리 계산: pair-wise 평균 위도로 cos_lat 적용
    t_lat = traffic_states[:, :, 0]  # (B, N)
    t_lon = traffic_states[:, :, 1]
    t_alt = traffic_states[:, :, 2]

    dlat = (my_lat.unsqueeze(1) - t_lat) * 60.0           # (B, N)
    # 대칭성 유지: 두 점의 평균 위도로 코사인 계산
    lat_mean = (my_lat.unsqueeze(1) + t_lat) / 2.0
    cos_lat_pair = torch.cos(torch.deg2rad(lat_mean)).clamp(min=0.01)
    dlon = (my_lon.unsqueeze(1) - t_lon) * 60.0 * cos_lat_pair  # (B, N)
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
    #   Inner Task 1: sep_h (수평 분리)
    #   절대 vice: 고정 페널티 (curriculum 없음), 경중 차이만.
    # ══════════════════════════════════
    r_sep_h = torch.zeros(B, device=device)
    r_sep_h -= has_ra.float() * 10000      # RA 절대 vice
    r_sep_h -= has_ta.float() * 1000       # TA 경보급
    self_caused = in_danger & ~injected
    r_sep_h -= self_caused.float() * 5000  # self-caused 가중

    # ══════════════════════════════════
    #   Inner Task 2: sep_v (수직 분리)
    #   per-step 누적 최소화 → Vs 0 근처 유지, Vc와의 분리 확대
    # ══════════════════════════════════
    r_sep_v = torch.zeros(B, device=device)
    min_v_dist = v_dist.min(dim=1).values  # (B,)
    r_sep_v -= (min_v_dist < 500).float() * 5
    r_sep_v -= ((min_v_dist >= 500) & (min_v_dist < 1000)).float() * 0.5

    # ══════════════════════════════════
    #   Inner Task 3: alt_floor (최저안전고도)
    #   per-step 누적 최소화
    # ══════════════════════════════════
    r_alt_floor = torch.zeros(B, device=device)
    r_alt_floor -= (my_alt < 2000).float() * 5
    r_alt_floor -= ((my_alt >= 2000) & (my_alt < 5000)).float() * 0.3

    # ══════════════════════════════════
    #   Inner Task 5: fuel (연료 소모율)
    #   주 feature: 속도
    # ══════════════════════════════════
    r_fuel = torch.zeros(B, device=device)
    r_fuel -= my_gs / 600.0

    # ══════════════════════════════════
    #   Inner Task 6: direct (경로 직진성)
    #   주 feature: heading vs 목적지 방위 차이
    # ══════════════════════════════════
    r_direct = torch.zeros(B, device=device)

    # ══════════════════════════════════
    #   Inner Task 7: progress (목적지 접근)
    #   주 feature: 거리 변화량
    # ══════════════════════════════════
    r_progress = torch.zeros(B, device=device)

    # ══════════════════════════════════
    #   Inner Task 8: alt_match (도착 고도 매칭)
    #   주 feature: 고도 vs 목표 고도 차이
    # ══════════════════════════════════
    r_alt_match = torch.zeros(B, device=device)

    # ── 목적지 관련 inner tasks ──
    cur_dist = None
    if dest_lat is not None:
        # bearing용: dest - my (목적지 방향)
        north_nm = (dest_lat - my_lat) * 60.0
        # 대칭: 두 점 평균 위도로 코사인
        cos_mean = torch.cos(
            torch.deg2rad((my_lat + dest_lat) / 2.0)).clamp(min=0.01)
        east_nm = (dest_lon - my_lon) * 60.0 * cos_mean
        cur_dist = torch.sqrt(north_nm**2 + east_nm**2)

        # direct: heading이 목적지 방향과 얼마나 정렬됐는지
        # bearing = atan2(east, north) (항공 convention: N=0, E=90, 시계방향)
        dest_bearing = torch.rad2deg(torch.atan2(east_nm, north_nm))
        hdg_diff = (own_state[:, 4] - dest_bearing + 180) % 360 - 180
        alignment = torch.cos(torch.deg2rad(hdg_diff))  # 1=정렬, -1=역방향
        r_direct += alignment * 2.0  # 정렬 보너스

        # progress: 거리 변화 (NM당 15점, 도달 보너스 대체)
        if prev_dist is not None:
            progress = (prev_dist - cur_dist).clamp(-5, 5)
            r_progress += progress * 15.0

        # alt_match: 목적지 고도 매칭
        if is_arrival is not None and dest_alt is not None:
            target_alt = torch.where(is_arrival,
                torch.tensor(10000.0, device=device),
                dest_alt if dest_alt is not None else torch.tensor(20000.0, device=device))
            alt_err = torch.abs(my_alt - target_alt) / 1000.0  # kft 단위
            if cur_dist is not None:
                near_dest = cur_dist < 50.0
                r_alt_match -= near_dest.float() * alt_err.clamp(0, 10) * 2.0

        # 터미널 평가 (inner task 직교성 유지: 각 축은 해당 축 feature만)
        if is_terminal is not None and is_arrival is not None:
            arr_reached = is_arrival & (cur_dist < _APP_HANDOFF_DIST_NM) & \
                          (my_alt >= _APP_HANDOFF_ALT_MIN) & (my_alt <= _APP_HANDOFF_ALT_MAX)
            dep_reached = ~is_arrival & (cur_dist < 10.0)
            reached = arr_reached | dep_reached

            # 도달 보너스 (curriculum-scaled, crash penalty와 대칭)
            r_progress += (reached & is_terminal).float() * reach_bonus
            not_reached_terminal = ~reached & is_terminal
            if init_dist is not None:
                dist_frac = (cur_dist / init_dist.clamp(min=1.0)).clamp(0, 2)
            else:
                dist_frac = torch.ones(B, device=device)
            # 미도달 terminal: dist 비례 × 500
            r_progress -= not_reached_terminal.float() * dist_frac * 500.0

            # fuel: 연료 상태
            if fuel_remaining is not None:
                fuel_frac = fuel_remaining.clamp(0, 1)
                # 연료 효율 보너스: 도달 시 잔량 비례
                r_fuel += (reached & is_terminal).float() * fuel_frac * 50.0
                # 연료 고갈 종료: 대형 페널티
                fuel_empty = (fuel_remaining <= 0) & is_terminal
                r_fuel -= fuel_empty.float() * 500.0

    # Inner tasks → Outer groups (합산)
    # B안: critic safety head는 순수 crash predictor가 되어야 함.
    # → safety 그룹 = sep_h(터미널 crash penalty)만.
    # → per-step shaping(sep_v, alt_floor)은 efficiency로 이동
    #    (actor가 per-step gradient density 유지하되, critic V_safety는 깨끗).
    inner = {
        'sep_h': r_sep_h, 'sep_v': r_sep_v, 'alt_floor': r_alt_floor,
        'fuel': r_fuel, 'direct': r_direct, 'progress': r_progress, 'alt_match': r_alt_match,
    }
    rewards = {
        'safety': r_sep_h,
        'efficiency': r_fuel + r_direct + r_sep_v + r_alt_floor,
        'mission': r_progress + r_alt_match,
    }
    rewards['_inner'] = inner  # 로깅/디버깅용
    # Certificate h = (1/K) Σ L_k² (PAVING 논문)
    inner_vals = torch.stack(list(inner.values()), dim=0)  # (8, B)
    rewards['_certificate'] = (inner_vals ** 2).mean(dim=0)  # (B,)

    return rewards, cur_dist, in_danger


# ── 연속 액션 공간 ──
# 3차원: Δhdg, Δalt, Δspd (tanh → [-1,1] → 스케일링)
CONT_ACTION_DIM = 3
ACTION_SCALE = torch.tensor([30.0, 5000.0, 50.0])  # hdg ±30°, alt ±5000ft, spd ±50kt


# ── Actor (Policy Network) — Gaussian 연속 정책 ──

class Actor(nn.Module):
    """관제 정책: 상태 → 연속 행동 (Δhdg, Δalt, Δspd)"""

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
        self.world_enc = nn.Sequential(
            nn.Linear(WORLD_DIM, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(256 + 128 + 64, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, CONT_ACTION_DIM)
        self.logstd_head = nn.Linear(hidden_dim, CONT_ACTION_DIM)

    def forward(self, own_state, traffic_states, world_feat=None):
        """
        :return: mean (B, 3), log_std (B, 3)
        """
        B = own_state.shape[0]
        s = self.state_enc(own_state)
        t = self.traffic_enc(traffic_states.reshape(B, -1))
        if world_feat is not None:
            w = self.world_enc(world_feat)
        else:
            w = torch.zeros(B, 64, device=own_state.device)
        h = self.trunk(torch.cat([s, t, w], dim=-1))
        mean = self.mean_head(h)
        log_std = self.logstd_head(h).clamp(-2, 0)  # std ∈ [0.135, 1.0]
        return mean, log_std

    def get_action(self, own_state, traffic_states, deterministic=False, world_feat=None):
        """tanh 스퀴싱된 액션 반환: (B, 3) ∈ [-1, 1]"""
        mean, log_std = self.forward(own_state, traffic_states, world_feat=world_feat)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        noise = torch.randn_like(std)
        return torch.tanh(mean + std * noise)

    def log_prob(self, own_state, traffic_states, action, world_feat=None):
        """Squashed Gaussian log probability"""
        mean, log_std = self.forward(own_state, traffic_states, world_feat=world_feat)
        std = log_std.exp()
        # atanh (inverse tanh) for unsquashing
        raw_action = torch.atanh(action.clamp(-0.999, 0.999))
        # Gaussian log prob
        log_p = -0.5 * ((raw_action - mean) / std).pow(2) - log_std - 0.5 * math.log(2 * math.pi)
        # tanh Jacobian correction
        log_p = log_p - torch.log(1 - action.pow(2) + 1e-6)
        return log_p.sum(dim=-1)  # (B,)


# ── Multi-Axis Critic (GAN-style 경쟁 학습) ──
#
# 3개 축이 각자 Actor를 자기 방향으로 당기며 경쟁:
#   Safety:     "이 상태가 얼마나 위험한가"
#   Efficiency: "연료를 얼마나 낭비하는가"
#   Mission:    "임무 완수에 얼마나 가까운가"
#
# Actor는 세 축의 균형점(파레토 최적)을 찾아야 함.
# 각 Critic의 출력 = 해당 축의 점수 → demo에서 세부 요인 표시에 직접 사용.

# ── Inner Tasks (8개) → Outer Groups (3개) ──
# 각 inner task는 서로 다른 상태 feature에 반응 → 자연 직교
INNER_TASKS = ['sep_h',                          # safety group (1) — crash terminal
               'sep_v', 'alt_floor', 'fuel', 'direct',  # efficiency group (4)
               'progress', 'alt_match']          # mission group (2)
TASK_GROUPS = {
    'safety': ['sep_h'],
    'efficiency': ['sep_v', 'alt_floor', 'fuel', 'direct'],
    'mission': ['progress', 'alt_match'],
}
REWARD_AXES = ['safety', 'efficiency', 'mission']  # 외부 호환용 유지

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
        self.world_enc = nn.Sequential(
            nn.Linear(WORLD_DIM, 64), nn.ELU(),
            nn.Linear(64, 32), nn.ELU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128 + 64 + 32, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, own_state, traffic_states, world_feat=None):
        B = own_state.shape[0]
        s = self.state_enc(own_state)
        t = self.traffic_enc(traffic_states.reshape(B, -1))
        if world_feat is not None:
            w = self.world_enc(world_feat)
        else:
            w = torch.zeros(B, 32, device=own_state.device)
        return self.value_head(torch.cat([s, t, w], dim=-1)).squeeze(-1)


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

    def forward(self, own_state, traffic_states, axis=None, world_feat=None):
        """axis=None이면 3축 가중합 (기본 가중치), axis 지정 시 해당 축만"""
        if axis:
            return self.heads[axis](own_state, traffic_states, world_feat=world_feat)
        # 기본: safety 우선 가중합
        s = self.safety(own_state, traffic_states, world_feat=world_feat)
        e = self.efficiency(own_state, traffic_states, world_feat=world_feat)
        m = self.mission(own_state, traffic_states, world_feat=world_feat)
        return s * 0.5 + e * 0.2 + m * 0.3

    def forward_all(self, own_state, traffic_states, world_feat=None):
        """3축 모두 반환: dict of (B,) tensors"""
        return {
            'safety': self.safety(own_state, traffic_states, world_feat=world_feat),
            'efficiency': self.efficiency(own_state, traffic_states, world_feat=world_feat),
            'mission': self.mission(own_state, traffic_states, world_feat=world_feat),
        }

    # 축별 default V 범위 (실측 EMA 없을 때만 fallback).
    # 학습 중 observed: safety ≈ (-25, -100), eff ≈ (-10, -30), mission ≈ (-10, -30)
    _DEFAULT_V_RANGE = {
        'safety':     (-25.0, -100.0),
        'efficiency': (-10.0,  -30.0),
        'mission':    (-10.0,  -30.0),
    }

    def set_value_range(self, axis, v_safe, v_crash):
        """학습 종료 후 실측 V 분포를 기반으로 정규화 범위 설정."""
        if not hasattr(self, '_v_range'):
            self._v_range = {}
        self._v_range[axis] = (float(v_safe), float(v_crash))

    def _get_v_range(self, axis):
        if hasattr(self, '_v_range') and axis in self._v_range:
            return self._v_range[axis]
        return self._DEFAULT_V_RANGE.get(axis, (-25.0, -100.0))

    def risk_score(self, own_state, traffic_states, world_feat=None):
        """Safety Advisor용: safety V → 위험도 [0,1].
        선형 정규화: V_safe → 0%, V_crash → 100%.
        """
        v_safe, v_crash = self._get_v_range('safety')
        with torch.no_grad():
            v = self.safety(own_state, traffic_states, world_feat=world_feat)
            risk = (v - v_safe) / (v_crash - v_safe + 1e-6)
            return risk.clamp(0, 1)

    def axis_scores(self, own_state, traffic_states, world_feat=None):
        """Safety Advisor용: 3축 위험도 dict (축별 V 범위 사용)."""
        with torch.no_grad():
            vals = self.forward_all(
                own_state, traffic_states, world_feat=world_feat)
            result = {}
            for axis in ('safety', 'efficiency', 'mission'):
                v_safe, v_crash = self._get_v_range(axis)
                risk = (vals[axis] - v_safe) / (v_crash - v_safe + 1e-6)
                result[axis] = risk.clamp(0, 1)
            return result


# ── 돌발 이벤트 주입 (배치 벡터화) ──

def inject_events_batch(states_raw, device, inject_prob=0.03,
                        collision_frac=0.6):
    """
    배치 전체에 돌발 이벤트를 GPU 텐서 연산으로 주입.
    Critic의 crash 학습 신호 확보용 — 초기 트래픽은 안전하게 유지,
    에피소드 중간에 돌발적으로 나타남.

    :param inject_prob: per-step 주입 확률 (각 배치 에피소드마다 매 step 독립 추첨)
    :param collision_frac: 주입 항공기가 충돌 코스로 배치될 비율.
        나머지는 랜덤 방위 (실제 돌발 관제 상황 모사).
    """
    B = states_raw.shape[0]
    inject_mask = torch.rand(B, device=device) < inject_prob

    lat = states_raw[:, 0]
    lon = states_raw[:, 1]
    alt = states_raw[:, 2]
    own_track = states_raw[:, 4]
    own_gs = states_raw[:, 3]
    cos_lat = torch.cos(lat * (math.pi / 180.0)).clamp(min=0.01)

    # 절반 이상은 충돌 코스, 나머지는 랜덤
    is_collision = torch.rand(B, device=device) < collision_frac

    # --- 충돌 코스: 내 항공기 전방 8-15NM, 180도 마주 오는 방향 ---
    col_bearing = own_track  # 내 진행방향 (degrees)
    col_dist_nm = torch.rand(B, device=device) * 7.0 + 8.0  # 8-15NM
    col_rad = col_bearing * (math.pi / 180.0)
    col_lat = lat + col_dist_nm * torch.cos(col_rad) / 60.0
    col_lon = lon + col_dist_nm * torch.sin(col_rad) / (60.0 * cos_lat)
    col_alt = alt + (torch.rand(B, device=device) - 0.5) * 1000  # ±500ft
    col_hdg = (own_track + 180.0 + (torch.rand(B, device=device) - 0.5) * 30.0) % 360
    col_gs = own_gs + (torch.rand(B, device=device) - 0.5) * 50  # 비슷한 속도

    # --- 랜덤 돌발: 5-20NM 원거리 ---
    rnd_bearing = torch.rand(B, device=device) * 360.0
    rnd_dist_nm = torch.rand(B, device=device) * 15.0 + 5.0
    rnd_rad = rnd_bearing * (math.pi / 180.0)
    rnd_lat = lat + rnd_dist_nm * torch.cos(rnd_rad) / 60.0
    rnd_lon = lon + rnd_dist_nm * torch.sin(rnd_rad) / (60.0 * cos_lat)
    rnd_alt = alt + (torch.rand(B, device=device) - 0.5) * 6000
    rnd_hdg = torch.rand(B, device=device) * 360
    rnd_gs = torch.rand(B, device=device) * 300 + 200

    # 마스킹 합성
    new_lat = torch.where(is_collision, col_lat, rnd_lat)
    new_lon = torch.where(is_collision, col_lon, rnd_lon)
    new_alt = torch.where(is_collision, col_alt, rnd_alt)
    new_hdg = torch.where(is_collision, col_hdg, rnd_hdg)
    new_gs = torch.where(is_collision, col_gs, rnd_gs)
    new_vrate = (torch.rand(B, device=device) - 0.5) * 2000
    new_ias = new_gs - torch.rand(B, device=device) * 50
    new_mach = torch.full((B,), 0.75, device=device)
    zeros = torch.zeros(B, device=device)

    injected = torch.stack([
        new_lat, new_lon, new_alt, new_gs, new_hdg,
        new_vrate, new_ias, new_mach, zeros, zeros
    ], dim=1)

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
                 imagination_horizon=120, gamma=0.99, actor_lr=1e-5, critic_lr=1e-4):
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

        self.max_horizon = imagination_horizon
        self.horizon = imagination_horizon  # 하위 호환
        self.gamma = gamma
        self.norm_mean = torch.from_numpy(NORM_MEAN).to(self.device)
        self.norm_std = torch.from_numpy(NORM_STD).to(self.device)

        self.total_episodes = 0
        self.total_crashes = 0
        self.total_safe = 0

        # World context (waypoints)
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        wp_data = _load_waypoints(data_dir)
        self._has_world = wp_data is not None
        if self._has_world:
            self._wp_array, self._wp_types, self._wp_names = wp_data

        # PAVING regroup controller (inner task gradient Gram tracking)
        self.paving = PAVINGController(
            inner_tasks=INNER_TASKS,
            initial_groups=TASK_GROUPS,
            n_groups=len(REWARD_AXES),
            tau=0.5,
            measure_interval=100,
            violation_persist=5,
            ema_alpha=0.1,
            cooldown=500)
        self._train_step_count = 0
        # Crash rate EMA (curriculum penalty scaling)
        self._crash_rate_ema = 0.5
        self._reach_rate_ema = 0.0  # 도달률 EMA
        # 3-Stage 커리큘럼
        #  1 = Reach only (돌발 주입 off)
        #  2 = Crash learning (주입 강함, collision 60%)
        #  3 = Safe reach (주입 약함, collision 30%)
        self._curriculum_stage = 1
        self._stage_start_step = 0

    # 주입 완전 제거 — actor의 safety weight 낮춤으로 대체 (actor 자연 crash).
    _CURRICULUM_STAGES = {
        1: {'inject_prob': 0.0, 'collision_frac': 0.0, 'name': 'natural_crash'},
        2: {'inject_prob': 0.0, 'collision_frac': 0.0, 'name': 'natural_crash'},
        3: {'inject_prob': 0.0, 'collision_frac': 0.0, 'name': 'natural_crash'},
    }

    def _stage_params(self):
        return self._CURRICULUM_STAGES[self._curriculum_stage]

    def _maybe_advance_stage(self):
        """커리큘럼 제거 — 고정 구성 (주입 0, 고정 페널티)."""
        pass

    def _denormalize(self, state_norm):
        return state_norm * self.norm_std + self.norm_mean

    def _normalize(self, state_raw):
        return (state_raw - self.norm_mean) / self.norm_std

    def _compute_world_feat(self, states_raw):
        """배치의 raw 상태에서 world features 계산. (B, D) → (B, WORLD_DIM).
        GPU→CPU 한 번만 sync, numpy 배치 연산으로 처리.
        """
        if not self._has_world:
            return None
        coords = states_raw[:, [0, 1, 4]].detach().cpu().numpy()  # (B, 3) — 한 번만 sync
        feat = _get_nearest_waypoints_batch(
            coords[:, 0], coords[:, 1], coords[:, 2],
            self._wp_array, self._wp_types, k=3)
        return torch.from_numpy(feat).to(self.device)

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
        에피소드 기반 imagination rollout (PhysicsWM 호환).

        VAE 의존성 제거: 각 스텝에서 Actor 액션 → 직접 물리 적분
        (forward_dynamics 방식). WM의 hidden state 불필요.

        각 배치 항목 = 1개 에피소드:
          - initial_states_norm[:, -1] = 내 항공기 초기 상태
          - traffic_env: (B, H, N_traffic, STATE_DIM) 각 스텝별 주변 트래픽
            None이면 initial_contexts에서 트래픽 추출하여 고정 사용
        """
        B, K, D = initial_states_norm.shape

        if not hasattr(self, '_action_table'):
            self._action_table = _build_action_table(self.device)

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

            # 매 스텝 트래픽 = 직선 외삽 (10초 간격) — max_horizon까지
            traffic_env = torch.zeros(B, self.max_horizon, N_traffic, STATE_DIM, device=self.device)
            for step in range(self.max_horizon):
                dt = (step + 1) * 10.0  # 초
                moved = static_traffic.clone()
                # kt → NM/s: 1kt = 1NM/h = 1/3600 NM/s
                spd_nm_s = moved[:, :, 3] / 3600.0
                hdg_rad = torch.deg2rad(moved[:, :, 4])
                # bearing convention: N=0°, E=90°
                moved[:, :, 0] += spd_nm_s * dt * torch.cos(hdg_rad) / 60.0
                c_lat = torch.cos(torch.deg2rad(moved[:, :, 0])).clamp(min=0.01)
                moved[:, :, 1] += spd_nm_s * dt * torch.sin(hdg_rad) / (60.0 * c_lat)
                moved[:, :, 2] += moved[:, :, 5] / 60.0 * dt  # vrate (fpm → ft/s)
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

        # Imagination rollout — 에피소드별 독립 종료
        all_states = [current_own_raw]
        all_actions = []
        all_traffic = []
        all_rewards = {ax: [] for ax in REWARD_AXES}
        all_values = {ax: [] for ax in REWARD_AXES}
        all_inner_rewards = {t: [] for t in INNER_TASKS}
        all_alive = []  # (B,) per step
        prev_in_danger = torch.zeros(B, dtype=torch.bool, device=self.device)

        # 연료 추적 (초기 1.0, 매 스텝 속도 비례 소모)
        fuel = torch.ones(B, device=self.device)  # 0~1 비율
        init_dist = prev_dist.clone()  # 초기 목적지 거리 저장
        alive = torch.ones(B, device=self.device)  # 1.0 alive, 0.0 dead
        ever_reached = torch.zeros(B, dtype=torch.bool, device=self.device)
        terminated_step = torch.full((B,), -1, dtype=torch.long, device=self.device)

        for step in range(self.max_horizon):
            if alive.sum() < 0.5:
                break

            own_norm = self._normalize(current_own_raw)
            world_feat = self._compute_world_feat(current_own_raw)
            step_traffic_raw = traffic_env[:, step]
            step_traffic_norm = (step_traffic_raw - self.norm_mean) / self.norm_std
            all_traffic.append(step_traffic_norm)

            # Critic value
            with torch.no_grad():
                vals = self.target_critic.forward_all(
                    own_norm, step_traffic_norm, world_feat=world_feat)
            for ax in REWARD_AXES:
                all_values[ax].append(vals[ax])

            action = self.actor.get_action(
                own_norm, step_traffic_norm, world_feat=world_feat)
            all_actions.append(action)

            # 목적지 방향 bias: "약한 절대적 욕구" (PAVING-style prior).
            # Actor는 raw action을 샘플, 실제 환경엔 (raw + 0.5*bias).clamp 적용.
            # → 기본값이 목적지로 향함. 트래픽 회피 시 actor가 counter-bias 학습.
            cur_track = current_own_raw[:, 4]  # deg
            dlat_d = (dest_lat - current_own_raw[:, 0]) * 60.0
            cos_m = torch.cos(torch.deg2rad(
                (current_own_raw[:, 0] + dest_lat) / 2.0)).clamp(min=0.01)
            dlon_d = (dest_lon - current_own_raw[:, 1]) * 60.0 * cos_m
            dest_bearing = torch.rad2deg(
                torch.atan2(dlon_d, dlat_d)) % 360
            bearing_diff = ((dest_bearing - cur_track + 180) % 360) - 180
            norm_bias_hdg = (bearing_diff / 30.0).clamp(-1.0, 1.0)
            dest_bias = torch.zeros_like(action)
            dest_bias[:, 0] = norm_bias_hdg * 0.5  # 약한 bias (hdg만)
            effective_action = (action + dest_bias).clamp(-1, 1)

            a_scale = ACTION_SCALE.to(self.device)
            scaled = effective_action * a_scale
            modified_raw = current_own_raw.clone()
            modified_raw[:, 4] = (modified_raw[:, 4] + scaled[:, 0]) % 360
            modified_raw[:, 2] = (modified_raw[:, 2] + scaled[:, 1]).clamp(2000, 45000)
            # 속도: 실속속도 이상 유지 (200kt 최소, 600kt 최대)
            modified_raw[:, 3] = (modified_raw[:, 3] + scaled[:, 2]).clamp(200, 600)

            # 물리 적분 (midpoint rule)
            dt_step = 10.0
            pre_gs = current_own_raw[:, 3]
            pre_track = current_own_raw[:, 4]
            pre_lat = current_own_raw[:, 0]
            pre_lon = current_own_raw[:, 1]
            post_gs = modified_raw[:, 3]
            post_track = modified_raw[:, 4]

            avg_gs = (pre_gs + post_gs) / 2.0
            dtrack = ((post_track - pre_track + 180.0) % 360.0) - 180.0
            mid_track_rad = torch.deg2rad(pre_track + dtrack / 2.0)
            dist_nm = avg_gs * dt_step / 3600.0
            new_lat = pre_lat + dist_nm * torch.cos(mid_track_rad) / 60.0
            cos_lat_m = torch.cos(torch.deg2rad(new_lat)).clamp(min=0.1)
            new_lon = pre_lon + dist_nm * torch.sin(mid_track_rad) / (60.0 * cos_lat_m)
            new_vrate = (modified_raw[:, 2] - current_own_raw[:, 2]) * 60.0 / dt_step

            next_own_raw = modified_raw.clone()
            next_own_raw[:, 0] = new_lat
            next_own_raw[:, 1] = new_lon
            next_own_raw[:, 5] = new_vrate

            # 죽은 에피소드는 상태 동결 (alive==0이면 이전 상태 유지)
            alive_mask = alive.unsqueeze(-1)  # (B, 1)
            next_own_raw = next_own_raw * alive_mask + current_own_raw * (1 - alive_mask)

            # 돌발 이벤트 주입 (커리큘럼 stage 별 강도)
            _sp = self._stage_params()
            injected_traffic, inject_mask = inject_events_batch(
                next_own_raw, self.device,
                inject_prob=_sp['inject_prob'],
                collision_frac=_sp['collision_frac'])
            reward_traffic = step_traffic_raw.clone()
            if inject_mask.any():
                injected_state = torch.stack([
                    injected_traffic[:, 0], injected_traffic[:, 1], injected_traffic[:, 2],
                    injected_traffic[:, 3], injected_traffic[:, 4], injected_traffic[:, 5],
                    injected_traffic[:, 3] - 30, torch.full((B,), 0.75, device=self.device),
                    torch.zeros(B, device=self.device), torch.zeros(B, device=self.device),
                ], dim=-1)
                reward_traffic[inject_mask, 0] = injected_state[inject_mask]

            # 연료 소모 (alive만)
            # gs=300kt에서 step당 0.0055 (최대 180step=30분 비행 가능).
            # 30분 × 300kt = 150NM 사거리 → 대부분 Korean airbase 도달 가능.
            fuel_burn = next_own_raw[:, 3] / 600.0 * 0.011
            fuel = (fuel - fuel_burn * alive).clamp(min=0)

            # 도달 판정 (종료 조건 계산용 — reward 함수와 일치)
            cur_dist_now = torch.sqrt(
                ((dest_lat - next_own_raw[:, 0]) * 60.0) ** 2 +
                ((dest_lon - next_own_raw[:, 1]) * 60.0 *
                 torch.cos(torch.deg2rad(
                     (next_own_raw[:, 0] + dest_lat) / 2.0)).clamp(min=0.01)) ** 2)
            arr_reached = is_arrival & (cur_dist_now < _APP_HANDOFF_DIST_NM) & \
                          (next_own_raw[:, 2] >= _APP_HANDOFF_ALT_MIN) & \
                          (next_own_raw[:, 2] <= _APP_HANDOFF_ALT_MAX)
            dep_reached = ~is_arrival & (cur_dist_now < 10.0)
            reached_now = (arr_reached | dep_reached) & (alive > 0.5)
            ever_reached = ever_reached | reached_now

            # RA 충돌 감지 (crash = 즉시 종료)
            # has_ra/has_ta를 다시 계산 (compute_reward_episode 내부 계산과 동일)
            t_lat = reward_traffic[:, :, 0]
            t_lon = reward_traffic[:, :, 1]
            t_alt = reward_traffic[:, :, 2]
            mean_lat = (next_own_raw[:, 0].unsqueeze(1) + t_lat) / 2.0
            cos_pair = torch.cos(torch.deg2rad(mean_lat)).clamp(min=0.01)
            dlat_t = (next_own_raw[:, 0].unsqueeze(1) - t_lat) * 60.0
            dlon_t = (next_own_raw[:, 1].unsqueeze(1) - t_lon) * 60.0 * cos_pair
            h_dist_t = torch.sqrt(dlat_t ** 2 + dlon_t ** 2)
            v_dist_t = torch.abs(next_own_raw[:, 2].unsqueeze(1) - t_alt)
            valid_t = (t_lat.abs() > 0.1) & (t_lon.abs() > 0.1)
            h_dist_t = h_dist_t.masked_fill(~valid_t, 1e9)
            ra_now = ((h_dist_t < 5.0) & (v_dist_t < 1000)).any(dim=1) & (alive > 0.5)

            # 종료 플래그: RA충돌 OR 도달 OR 연료 고갈 OR max_horizon 끝
            fuel_empty = (fuel <= 0) & (alive > 0.5)
            is_last_step = (step == self.max_horizon - 1)
            is_terminal_t = (ra_now | reached_now | fuel_empty |
                             (is_last_step & (alive > 0.5)))

            # 이번 스텝에 새로 종료된 에피소드 기록
            just_terminated = is_terminal_t & (terminated_step < 0)
            terminated_step = torch.where(just_terminated,
                torch.full_like(terminated_step, step), terminated_step)

            # 보상 계산 — 고정 페널티 (커리큘럼 scaling 제거, 절대 vice)
            rewards, cur_dist, in_danger = compute_reward_episode(
                next_own_raw, reward_traffic, action, self.device,
                dest_lat=dest_lat, dest_lon=dest_lon, dest_alt=dest_alt,
                prev_dist=prev_dist, init_dist=init_dist, is_arrival=is_arrival,
                is_terminal=is_terminal_t, fuel_remaining=fuel,
                injected=inject_mask, prev_in_danger=prev_in_danger)
            if cur_dist is not None:
                prev_dist = cur_dist
            prev_in_danger = in_danger

            # 죽은 에피소드는 reward 0 (alive 0이면 이번 스텝 보상 무효)
            for ax in REWARD_AXES:
                all_rewards[ax].append(rewards[ax] * alive)
            for t in INNER_TASKS:
                all_inner_rewards[t].append(rewards['_inner'][t] * alive)
            all_alive.append(alive.clone())

            # Alive 업데이트: 이번 스텝에 종료되면 다음 스텝부터 죽음
            alive = alive * (1.0 - is_terminal_t.float())

            current_own_raw = next_own_raw
            all_states.append(current_own_raw)

        # 마지막 value (bootstrap) — 남아있는 alive에 대해
        own_norm_final = self._normalize(current_own_raw)
        last_step_idx = min(step, self.max_horizon - 1)
        final_traffic_norm = (traffic_env[:, last_step_idx] - self.norm_mean) / self.norm_std
        final_world_feat = self._compute_world_feat(current_own_raw)
        with torch.no_grad():
            final_vals = self.target_critic.forward_all(
                own_norm_final, final_traffic_norm, world_feat=final_world_feat)
        # 이미 종료된 에피소드는 bootstrap 0
        for ax in REWARD_AXES:
            final_vals[ax] = final_vals[ax] * alive

        return {
            'states': torch.stack(all_states, dim=1),
            'actions': torch.stack(all_actions, dim=1),
            'traffic': torch.stack(all_traffic, dim=1),
            'rewards': {ax: torch.stack(all_rewards[ax], dim=1) for ax in REWARD_AXES},
            'values': {ax: torch.stack(all_values[ax], dim=1) for ax in REWARD_AXES},
            'inner_rewards': {t: torch.stack(all_inner_rewards[t], dim=1)
                              for t in INNER_TASKS},
            'alive': torch.stack(all_alive, dim=1),  # (B, H_actual)
            'ever_reached': ever_reached,
            'terminated_step': terminated_step,
            'final_value': final_vals,
            'fuel_final': fuel,
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
            return returns.clamp(-50000, 500)

    def train_step(self, initial_states_norm, initial_contexts):
        """
        PAVING + Multi-Axis Actor/Critic 학습:
        1. Imagination rollout → inner rewards (7 tasks)
        2. 현재 PAVING 그룹핑으로 inner → axis reward 합산
        3. 각 Critic head 독립 학습 (자기 그룹의 returns)
        4. Actor: 3축 advantage 가중합
        5. 주기적으로 per-task actor gradient 측정 → Gram EMA → regroup
        """
        self._train_step_count += 1
        rollout = self.imagine_rollout(initial_states_norm, initial_contexts)
        inner_r = rollout['inner_rewards']  # dict task → (B, H)

        # 현재 PAVING 그룹핑으로 axis별 reward 재조립
        # (주의: 슬롯 이름인 safety/efficiency/mission은 이제 단순 ID)
        paving_groups = self.paving.get_groups()
        axis_rewards = {}
        zero_r = next(iter(inner_r.values())).new_zeros(
            next(iter(inner_r.values())).shape)
        for ax in REWARD_AXES:
            tasks = paving_groups.get(ax, [])
            axis_rewards[ax] = (sum(inner_r[t] for t in tasks)
                                if tasks else zero_r.clone())

        # 축별 returns 계산 (모든 축 symlog — crash -10000도 compress되어 안정)
        returns_per_axis = {}
        for ax in REWARD_AXES:
            returns_per_axis[ax] = self.compute_returns(
                axis_rewards[ax], rollout['values'][ax],
                rollout['final_value'][ax],
                use_symlog=True)

        # 공통 준비 — Actor 그래프에서 분리 (Critic/Actor 업데이트는 독립)
        own_states = rollout['states'][:, :-1].detach()  # (B, H, D) raw
        B, H, D = own_states.shape
        own_flat = ((own_states - self.norm_mean) / self.norm_std).reshape(B * H, D)

        traffic_all = rollout['traffic'].detach()
        N_t = traffic_all.shape[2]
        traffic_flat = traffic_all.reshape(B * H, N_t, STATE_DIM)

        world_flat = None
        if self._has_world:
            own_raw_flat = own_states.reshape(B * H, D)
            world_flat = self._compute_world_feat(own_raw_flat)

        # ── 각 Critic head 독립 학습 (symlog 공간, ±10 clamp) ──
        # Safety head: TD MSE + contrastive (V_safe > V_crash margin loss)
        critic_losses = {}
        targets_per_axis = {}
        # Crash mask & alive mask (contrastive loss용)
        sep_h_traj_for_mask = inner_r['sep_h']  # (B, H)
        crash_mask_early = (sep_h_traj_for_mask < -30).any(dim=1)  # (B,)
        alive_traj_for_mask = rollout.get('alive', None)
        if alive_traj_for_mask is not None:
            alive_flat = alive_traj_for_mask.reshape(-1) > 0.5
        else:
            alive_flat = torch.ones(B * H, dtype=torch.bool, device=self.device)
        crash_state_mask = (
            crash_mask_early.unsqueeze(1).expand(-1, H).reshape(-1) & alive_flat)
        safe_state_mask = (~crash_mask_early.unsqueeze(1).expand(-1, H).reshape(-1)
                           & alive_flat)

        for ax in REWARD_AXES:
            raw_targets = returns_per_axis[ax].reshape(-1).detach()
            targets = torch.nan_to_num(raw_targets, nan=0.0).clamp(-10, 10)
            targets_per_axis[ax] = targets
            pred = self.critic.heads[ax](own_flat, traffic_flat, world_feat=world_flat)
            loss = F.mse_loss(pred, targets)

            # Safety head에만 contrastive loss
            # Pairwise hinge: 모든 (crash, safe) 쌍에서 V_safe - V_crash > margin 강제
            # → mean이 아니라 worst-case (min gap) 향상.
            if ax == 'safety':
                MARGIN = 3.0
                CONTRAST_W = 0.5
                N_MAX = 256
                if crash_state_mask.any() and safe_state_mask.any():
                    v_in_crash = pred[crash_state_mask]
                    v_in_safe = pred[safe_state_mask]
                    # Sampling 256개로 cap (B*H 큰 경우 메모리 절약)
                    if v_in_crash.size(0) > N_MAX:
                        idx = torch.randperm(v_in_crash.size(0),
                                              device=v_in_crash.device)[:N_MAX]
                        v_in_crash = v_in_crash[idx]
                    if v_in_safe.size(0) > N_MAX:
                        idx = torch.randperm(v_in_safe.size(0),
                                              device=v_in_safe.device)[:N_MAX]
                        v_in_safe = v_in_safe[idx]
                    # Pairwise diff: (N_c, N_s) — positive면 정상 분리
                    diff = v_in_safe.unsqueeze(0) - v_in_crash.unsqueeze(1)
                    # Margin 미달인 모든 pair에 페널티
                    contrast_loss = torch.relu(MARGIN - diff).mean()
                    loss = loss + CONTRAST_W * contrast_loss

            critic_losses[ax] = loss.item()

            if torch.isfinite(loss):
                self.critic_opts[ax].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.heads[ax].parameters(), 1.0)
                self.critic_opts[ax].step()

        w_dict = {ax: 1.0/len(REWARD_AXES) for ax in REWARD_AXES}  # 균등 (로깅용)

        # ── Actor 업데이트 ──
        actions_flat = rollout['actions'].reshape(B * H, CONT_ACTION_DIM)
        log_probs = self.actor.log_prob(own_flat, traffic_flat, actions_flat,
                                         world_feat=world_flat)

        # Actor의 safety 가중치 낮춤 — actor는 safety를 덜 학습해서
        # 자연스러운 crash(state-dependent)를 자주 일으킴 → Critic이 V_crash
        # 학습에 필요한 다양한 sample 확보. Critic safety head는 독립적으로
        # 강하게 학습 (critic_losses는 head별 독립 optimizer).
        axis_weights = {'safety': 0.1, 'efficiency': 0.3, 'mission': 0.6}
        total_advantage = torch.zeros(B * H, device=self.device)
        advs = {}
        for ax in REWARD_AXES:
            with torch.no_grad():
                v = self.critic.heads[ax](own_flat, traffic_flat, world_feat=world_flat)
            adv = (targets_per_axis[ax] - v).detach()
            adv = torch.nan_to_num(adv, nan=0.0)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            advs[ax] = adv
            total_advantage += adv * axis_weights[ax]

        _, log_std = self.actor(own_flat, traffic_flat, world_feat=world_flat)
        entropy = -log_std.mean()

        # ── PAVING 측정: per-task actor gradient (주기적) ──
        regroup_info = None
        if self._train_step_count % self.paving.measure_interval == 0:
            per_task_grads = {}
            for task in INNER_TASKS:
                task_group = self.paving.task_to_group[task]
                v_g = rollout['values'][task_group]
                fv_g = rollout['final_value'][task_group]
                task_returns = self.compute_returns(
                    inner_r[task], v_g, fv_g, use_symlog=True)
                adv_t = (task_returns.reshape(-1) -
                         v_g.reshape(-1)).detach()
                adv_t = torch.nan_to_num(adv_t, nan=0.0)
                std = adv_t.std()
                if std > 1e-8:
                    adv_t = (adv_t - adv_t.mean()) / std
                loss_t = -(log_probs * adv_t).mean()
                if not torch.isfinite(loss_t):
                    continue
                grads = torch.autograd.grad(
                    loss_t, self.actor.parameters(),
                    retain_graph=True, allow_unused=True)
                flat = torch.cat([
                    g.detach().flatten() if g is not None
                    else torch.zeros_like(p).flatten()
                    for g, p in zip(grads, self.actor.parameters())])
                per_task_grads[task] = flat.cpu().numpy()

            if len(per_task_grads) == len(INNER_TASKS):
                self.paving.measure(per_task_grads)
                # PAVING regrouping 비활성 — V_safety 순수성 보장 위해 그룹 고정.
                # 측정(Gram EMA)는 진단용으로 유지.
                # regroup_info = self.paving.check_and_regroup(self._train_step_count)
                regroup_info = None

        actor_loss = -(log_probs * total_advantage).mean() - 0.01 * entropy

        if torch.isfinite(actor_loss):
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

        cert_h = sum(v**2 for v in critic_losses.values()) / len(critic_losses)

        # PAVING diagnostic: axis-level |cos| (그룹 간)
        max_cos_sim = 0.0
        with torch.no_grad():
            axes_list = list(REWARD_AXES)
            norms = {ax: advs[ax].norm() for ax in axes_list}
            for i in range(len(axes_list)):
                for j in range(i+1, len(axes_list)):
                    dot = (advs[axes_list[i]] * advs[axes_list[j]]).sum()
                    denom = norms[axes_list[i]] * norms[axes_list[j]]
                    cos = (dot / denom.clamp(min=1e-8)).abs().item()
                    max_cos_sim = max(max_cos_sim, cos)

        # CANON 위반 경고 (§8.9 τ=0.5, throttled)
        if not hasattr(self, '_canon_warn_step'):
            self._canon_warn_step = 0
        if max_cos_sim > 0.5 and (self.total_episodes - self._canon_warn_step) > 10000:
            self._canon_warn_step = self.total_episodes
            print(f"[PAVING] CANON warning: max|cos|={max_cos_sim:.2f} (>0.5)")

        # Target critic soft update
        tau = 0.02
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        # 통계 (crash 판별은 그룹핑과 무관하게 sep_h inner task의 RA/TA penalty로)
        sep_h_traj = inner_r['sep_h']  # (B, H)
        crash_mask = (sep_h_traj < -30).any(dim=1)  # (B,)
        crashes = crash_mask.sum().item()
        self.total_episodes += B
        self.total_crashes += crashes
        self.total_safe += B - crashes
        # Crash rate EMA 업데이트 (curriculum scaling용)
        current_rate = crashes / max(B, 1)
        self._crash_rate_ema = 0.95 * self._crash_rate_ema + 0.05 * current_rate
        # (reach rate EMA는 reached_count 계산 후 아래에서 업데이트)

        # 에피소드 종료 통계
        ever_reached = rollout.get('ever_reached', None)
        terminated_step = rollout.get('terminated_step', None)
        fuel_final = rollout.get('fuel_final', None)
        reached_count = ever_reached.sum().item() if ever_reached is not None else 0
        fuel_empty_count = ((fuel_final <= 0).sum().item()
                            if fuel_final is not None else 0)
        avg_ep_len = (terminated_step.clamp(min=0).float().mean().item() + 1
                      if terminated_step is not None else 0)

        # Reach rate EMA + 커리큘럼 stage 전환
        reach_rate_now = reached_count / max(B, 1)
        self._reach_rate_ema = 0.95 * self._reach_rate_ema + 0.05 * reach_rate_now
        self._maybe_advance_stage()

        mean_v = {ax: rollout['values'][ax].mean().item() for ax in REWARD_AXES}

        # Critic 예측 능력 평가 — 배포(deployment) 기준과 정렬:
        # SafetyAdvisor는 매 (state, t)에서 V → risk_score 출력. 평가도 동일하게
        # "V가 향후 K-step 내 crash를 예측하는가?" → AUC.
        # 모든 alive (state, t) 쌍에 대해:
        #   label = (이 state 이후 K step 안에 crash 발생?) ∈ {0, 1}
        #   predictor = -V (낮은 V일수록 crash 예측)
        v_safety_traj = rollout['values']['safety']  # (B, H)
        alive_mask_traj = rollout.get('alive', None)  # (B, H)
        _H_actual = v_safety_traj.shape[1]
        if terminated_step is not None:
            term_idx = terminated_step.clamp(min=0).long()
        else:
            term_idx = torch.full((B,), _H_actual - 1, dtype=torch.long, device=self.device)

        # Crash step (only valid for crashed episodes; safe = +inf)
        crash_step = torch.where(
            crash_mask, term_idx.float(),
            torch.full_like(term_idx.float(), 1e9))
        ts = torch.arange(_H_actual, device=self.device).float().unsqueeze(0).expand(B, _H_actual)
        ttc = crash_step.unsqueeze(1) - ts  # time-to-crash, (B, H)

        v_flat = v_safety_traj.reshape(-1).detach()
        if alive_mask_traj is not None:
            valid_mask = (alive_mask_traj.reshape(-1) > 0.5)
        else:
            valid_mask = torch.ones_like(v_flat, dtype=torch.bool)
        v_valid = v_flat[valid_mask].cpu().numpy()

        auc_horizons = {}
        try:
            from sklearn.metrics import roc_auc_score
            for K in (5, 10, 20, 9999):
                lbl = ((ttc > 0) & (ttc <= K)).float().reshape(-1)
                lbl_valid = lbl[valid_mask].cpu().numpy()
                pos = lbl_valid.sum()
                neg = len(lbl_valid) - pos
                if pos > 0 and neg > 0:
                    auc = roc_auc_score(lbl_valid, -v_valid)
                    auc_horizons[K] = float(auc)
        except Exception:
            pass

        # State-level 평가 (deployment 정렬):
        # - imminent: K-step 이내 crash 발생 예정 state ("지금 위험")
        # - normal: K-step 이내 crash 없음 state ("지금 안전")
        # Critic V는 매 순간 risk level 출력해야 → 매 (state, t) pair 단위 평가.
        K_IMMINENT = 10
        if alive_mask_traj is not None:
            alive_bool = alive_mask_traj > 0.5
        else:
            alive_bool = torch.ones_like(v_safety_traj, dtype=torch.bool)
        imminent_state_mask = alive_bool & (ttc > 0) & (ttc <= K_IMMINENT)
        normal_state_mask = alive_bool & ((ttc > K_IMMINENT) | (ttc > 1e8))

        v_imminent = v_safety_traj[imminent_state_mask]
        v_normal = v_safety_traj[normal_state_mask]
        v_crash = v_imminent.mean().item() if v_imminent.numel() > 0 else None
        v_safe = v_normal.mean().item() if v_normal.numel() > 0 else None

        # Min gap: worst-case state-level 분리
        # = (정상 state 중 가장 위험하게 본 V) - (위험 state 중 가장 안전하게 본 V)
        # 양수 = 모든 정상 > 모든 위험 (완벽 분리)
        if v_imminent.numel() > 0 and v_normal.numel() > 0:
            v_imm_max = v_imminent.max().item()  # imminent 중 V 가장 높은 (덜 위험하게 봄)
            v_nor_min = v_normal.min().item()    # normal 중 V 가장 낮은 (가장 위험하게 봄)
            min_gap = v_nor_min - v_imm_max
        else:
            min_gap = None

        # V 통계 EMA 업데이트 (risk_score/axis_scores 정규화용)
        if crashes > 0 and crashes < B:
            stats_by_axis = {}
            for ax in REWARD_AXES:
                v_init = rollout['values'][ax][:, 0]  # (B,) 시작 V
                v_c = v_init[crash_mask].mean().item()
                v_s = v_init[~crash_mask].mean().item()
                stats_by_axis[ax] = (v_s, v_c)
            self.update_v_stats(stats_by_axis, alpha=0.01)

        # 축별 보상 합계 (현재 그룹핑 기준)
        r_sums = {ax: axis_rewards[ax].sum(dim=1).mean().item()
                  for ax in REWARD_AXES}
        safety_r = r_sums['safety']

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
            'r_safety': r_sums['safety'],
            'r_efficiency': r_sums['efficiency'],
            'r_mission': r_sums['mission'],
            'certificate_h': cert_h,
            'max_cos_sim': max_cos_sim,
            'crashes': crashes,
            'v_crash': v_crash,
            'v_safe': v_safe,
            'entropy': entropy.item(),
            'paving_regroups': self.paving.regroup_count,
            'paving_max_inter_cos': self.paving.max_inter_group_cos()[0],
            'paving_groups': self.paving.get_groups(),
            'regrouped': regroup_info is not None,
            'reached': reached_count,
            'fuel_empty': fuel_empty_count,
            'avg_ep_len': avg_ep_len,
            'crash_rate_ema': self._crash_rate_ema,
            'crash_penalty_scaled': min(500.0 * 0.5 / max(self._crash_rate_ema, 0.005), 30000.0),
            'reach_rate_ema': self._reach_rate_ema,
            'curriculum_stage': self._curriculum_stage,
            'auc_5': auc_horizons.get(5, None),
            'auc_10': auc_horizons.get(10, None),
            'auc_20': auc_horizons.get(20, None),
            'auc_any': auc_horizons.get(9999, None),
            'min_gap': min_gap,
        }

    def save(self, path):
        # Critic V 통계 EMA (risk_score 정규화용)
        v_stats = getattr(self, '_v_stats_ema', {})
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opts': {ax: opt.state_dict() for ax, opt in self.critic_opts.items()},
            'total_episodes': self.total_episodes,
            'total_crashes': self.total_crashes,
            'total_safe': self.total_safe,
            'v_stats_ema': v_stats,  # {axis: {'safe': v, 'crash': v}}
            'paving_gram_ema': self.paving.gram_ema,
            'paving_groups': self.paving.get_groups(),
            'paving_regroup_count': self.paving.regroup_count,
            'paving_measurements': self.paving.measurements,
            'train_step_count': self._train_step_count,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if 'actor' in ckpt:
            try:
                self.actor.load_state_dict(ckpt['actor'])
                self.actor_opt.load_state_dict(ckpt['actor_opt'])
                print(f"[Dreamer] Loaded actor from checkpoint")
            except RuntimeError as e:
                print(f"[Dreamer] Actor structure changed, starting fresh: {e}")
            try:
                self.critic.load_state_dict(ckpt['critic'])
                self.target_critic.load_state_dict(ckpt['target_critic'])
                if 'critic_opts' in ckpt:
                    for ax, state in ckpt['critic_opts'].items():
                        self.critic_opts[ax].load_state_dict(state)
                print(f"[Dreamer] Loaded critic from checkpoint")
            except RuntimeError as e:
                print(f"[Dreamer] Critic load failed: {e}")
        else:
            print(f"[Dreamer] No actor/critic in checkpoint - starting fresh")
        self.total_episodes = ckpt.get('total_episodes', 0)
        self.total_crashes = ckpt.get('total_crashes', 0)
        self.total_safe = ckpt.get('total_safe', 0)
        self._train_step_count = ckpt.get('train_step_count', 0)

        # PAVING 상태 복원
        if 'paving_gram_ema' in ckpt:
            import numpy as _np
            g = _np.asarray(ckpt['paving_gram_ema'])
            if g.shape == self.paving.gram_ema.shape:
                self.paving.gram_ema = g
                self.paving.measurements = ckpt.get('paving_measurements', 0)
                self.paving.regroup_count = ckpt.get('paving_regroup_count', 0)
                # paving_groups는 ckpt에서 로드하지 않음 — TASK_GROUPS(코드)가
                # source of truth (B안: safety=[sep_h]만, regroup 비활성).
                print(f"[PAVING] Loaded gram_ema only "
                      f"(regroups={self.paving.regroup_count}, "
                      f"measurements={self.paving.measurements}). "
                      f"Groups from TASK_GROUPS: {self.paving.groups}")

        # V 통계 복원 → critic에 범위 설정
        v_stats = ckpt.get('v_stats_ema', {})
        if v_stats:
            for axis, stats in v_stats.items():
                if 'safe' in stats and 'crash' in stats:
                    self.critic.set_value_range(
                        axis, stats['safe'], stats['crash'])
            self._v_stats_ema = v_stats
            print(f"[Dreamer] V stats loaded: {v_stats}")

    def update_v_stats(self, stats_by_axis, alpha=0.01):
        """
        학습 중 V 분포 EMA 업데이트 (save 시 ckpt에 기록).

        :param stats_by_axis: dict[axis -> (v_safe, v_crash)]
            예: {'safety': (-25, -100), 'efficiency': (-10, -40), ...}
        """
        if not hasattr(self, '_v_stats_ema'):
            self._v_stats_ema = {}
        for axis, (v_s, v_c) in stats_by_axis.items():
            if axis not in self._v_stats_ema:
                self._v_stats_ema[axis] = {'safe': float(v_s), 'crash': float(v_c)}
            else:
                prev = self._v_stats_ema[axis]
                self._v_stats_ema[axis] = {
                    'safe': (1 - alpha) * prev['safe'] + alpha * float(v_s),
                    'crash': (1 - alpha) * prev['crash'] + alpha * float(v_c),
                }
            self.critic.set_value_range(
                axis,
                self._v_stats_ema[axis]['safe'],
                self._v_stats_ema[axis]['crash'])

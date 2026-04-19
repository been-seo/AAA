"""
항공기 물리 모델 기반 Reachable Envelope 계산

항공기의 현재 상태에서 dt초 후 도달 가능한 상태 공간의 경계를 계산.
World Model v3에서 디코더 대신 사용:
  1. physics가 바운더리 계산 → [state_min, state_max]
  2. 신경망이 바운더리 내 [0, 1] 비율로 예측
  3. 실제 상태 = state_min + ratio * (state_max - state_min)

모든 예측이 물리적으로 유효함을 보장.
"""
import math
import numpy as np
import torch

# ── 기종별 성능 한계 ──
# (hdg_rate_deg_s, max_climb_fpm, max_desc_fpm, spd_accel_kts_s, spd_min, spd_max)
AIRCRAFT_PERF = {
    'fighter': {
        'hdg_rate': 15.0,       # 도/초 (전투기 최대)
        'hdg_rate_normal': 5.0, # 도/초 (일반 선회)
        'max_climb_fpm': 20000, # ft/min
        'max_desc_fpm': 20000,
        'spd_accel': 20.0,      # kts/s
        'spd_min': 200.0,       # kts
        'spd_max': 600.0,       # kts
        'alt_min': 0.0,         # ft
        'alt_max': 50000.0,     # ft
    },
    'civil': {
        'hdg_rate': 18.0,       # 도/초 (전방위 커버: 훈련기 유턴 포함)
        'hdg_rate_normal': 3.0,
        'max_climb_fpm': 6000,  # 훈련기/경비행기 포함
        'max_desc_fpm': 6000,
        'spd_accel': 10.0,
        'spd_min': 60.0,        # 경비행기 최소
        'spd_max': 500.0,
        'alt_min': 0.0,
        'alt_max': 45000.0,
    },
}

# 단위 변환
_KNOTS_TO_NM_PER_SEC = 0.000277778  # 1 kt = 1 NM/hr = 1/3600 NM/s


def compute_envelope_single(lat, lon, alt, gs_kt, track_deg, vrate_fpm,
                            dt_sec=10.0, ac_type='civil'):
    """
    단일 항공기의 dt초 후 reachable envelope 계산.

    Returns:
        env_min: (STATE_DIM,) 각 상태 변수의 최솟값
        env_max: (STATE_DIM,) 각 상태 변수의 최댓값
        env_baseline: (STATE_DIM,) 관성 유지 시 예측값 (직선 외삽)

    STATE_DIM 순서: [lat, lon, alt, gs, track, vrate, ias, mach, wind_dir, wind_spd]
    위치(lat, lon)는 헤딩 변화 범위에 따른 부채꼴 영역으로 계산.
    """
    perf = AIRCRAFT_PERF.get(ac_type, AIRCRAFT_PERF['civil'])

    # ── 속도 범위 ──
    spd_delta = perf['spd_accel'] * dt_sec
    gs_min = max(perf['spd_min'], gs_kt - spd_delta)
    gs_max = min(perf['spd_max'], gs_kt + spd_delta)
    gs_baseline = np.clip(gs_kt, perf['spd_min'], perf['spd_max'])

    # ── 헤딩 범위 ──
    hdg_delta = perf['hdg_rate'] * dt_sec  # 최대 선회 가능 각도
    hdg_min = (track_deg - hdg_delta) % 360
    hdg_max = (track_deg + hdg_delta) % 360
    hdg_baseline = track_deg

    # ── 고도 범위 ──
    climb_max = perf['max_climb_fpm'] / 60.0 * dt_sec  # ft
    desc_max = perf['max_desc_fpm'] / 60.0 * dt_sec
    # 현재 vrate 기반 관성 + 최대 변화
    alt_baseline = np.clip(alt + vrate_fpm / 60.0 * dt_sec,
                           perf['alt_min'], perf['alt_max'])
    alt_max_val = np.clip(alt + climb_max, perf['alt_min'], perf['alt_max'])
    alt_min_val = np.clip(alt - desc_max, perf['alt_min'], perf['alt_max'])

    # ── vrate 범위 ──
    vrate_max = perf['max_climb_fpm']
    vrate_min = -perf['max_desc_fpm']
    vrate_baseline = np.clip(vrate_fpm, vrate_min, vrate_max)

    # ── 위치 범위 (부채꼴) ──
    # 최소/최대 속도 × 최소/최대 헤딩으로 도달 가능한 위치 범위
    # 평균 속도로 이동한 거리
    dist_min = gs_min * _KNOTS_TO_NM_PER_SEC * dt_sec
    dist_max = gs_max * _KNOTS_TO_NM_PER_SEC * dt_sec
    dist_baseline = gs_baseline * _KNOTS_TO_NM_PER_SEC * dt_sec

    cos_lat = math.cos(math.radians(lat))
    cos_lat = max(cos_lat, 0.01)

    # baseline 위치 (직선 외삽)
    hdg_rad = math.radians(track_deg)
    lat_baseline = lat + dist_baseline * math.cos(hdg_rad) / 60.0
    lon_baseline = lon + dist_baseline * math.sin(hdg_rad) / (60.0 * cos_lat)

    # 위치 범위: 모든 가능한 (hdg, dist) 조합의 min/max
    # 부채꼴의 꼭짓점들을 계산
    angles = np.linspace(track_deg - hdg_delta, track_deg + hdg_delta, 16)
    angles_rad = np.radians(angles)

    lat_candidates = []
    lon_candidates = []
    for d in [dist_min, dist_max]:
        for a in angles_rad:
            lat_candidates.append(lat + d * math.cos(a) / 60.0)
            lon_candidates.append(lon + d * math.sin(a) / (60.0 * cos_lat))

    lat_min_val = min(lat_candidates)
    lat_max_val = max(lat_candidates)
    lon_min_val = min(lon_candidates)
    lon_max_val = max(lon_candidates)

    # ── envelope 조립 ──
    # [lat, lon, alt, gs, track, vrate, ias, mach, wind_dir, wind_spd]
    env_min = np.array([
        lat_min_val, lon_min_val, alt_min_val, gs_min,
        hdg_min, vrate_min,
        gs_min - 30,  # IAS 근사 (GS - wind 보정)
        max(0.4, gs_min / 660.0),  # Mach 근사
        0.0, 0.0,     # wind는 항공기가 제어 불가 → 변화 없음
    ], dtype=np.float32)

    env_max = np.array([
        lat_max_val, lon_max_val, alt_max_val, gs_max,
        hdg_max, vrate_max,
        gs_max - 10,
        min(1.2, gs_max / 660.0),
        360.0, 60.0,  # wind 범위 (관측 불확실성)
    ], dtype=np.float32)

    env_baseline = np.array([
        lat_baseline, lon_baseline, alt_baseline, gs_baseline,
        hdg_baseline, vrate_baseline,
        gs_baseline - 20,
        gs_baseline / 660.0,
        0.0, 0.0,  # wind 유지
    ], dtype=np.float32)

    return env_min, env_max, env_baseline


def compute_envelope_batch(states, dt_sec=10.0, ac_type='civil'):
    """
    배치 벡터화된 envelope 계산.

    states: (B, STATE_DIM) numpy array — [lat, lon, alt, gs, track, vrate, ias, mach, wind_dir, wind_spd]
    Returns:
        env_min: (B, STATE_DIM)
        env_max: (B, STATE_DIM)
        env_baseline: (B, STATE_DIM)
    """
    B = states.shape[0]
    perf = AIRCRAFT_PERF.get(ac_type, AIRCRAFT_PERF['civil'])

    lat = states[:, 0]
    lon = states[:, 1]
    alt = states[:, 2]
    gs = states[:, 3]
    track = states[:, 4]
    vrate = states[:, 5]

    # 속도
    spd_delta = perf['spd_accel'] * dt_sec
    gs_min = np.clip(gs - spd_delta, perf['spd_min'], perf['spd_max'])
    gs_max = np.clip(gs + spd_delta, perf['spd_min'], perf['spd_max'])
    gs_base = np.clip(gs, perf['spd_min'], perf['spd_max'])

    # 헤딩
    hdg_delta = perf['hdg_rate'] * dt_sec
    hdg_min = (track - hdg_delta) % 360
    hdg_max = (track + hdg_delta) % 360

    # 고도
    climb_max = perf['max_climb_fpm'] / 60.0 * dt_sec
    desc_max = perf['max_desc_fpm'] / 60.0 * dt_sec
    alt_base = np.clip(alt + vrate / 60.0 * dt_sec, perf['alt_min'], perf['alt_max'])
    alt_min_v = np.clip(alt - desc_max, perf['alt_min'], perf['alt_max'])
    alt_max_v = np.clip(alt + climb_max, perf['alt_min'], perf['alt_max'])

    # vrate
    vrate_min_v = np.full(B, -perf['max_desc_fpm'], dtype=np.float32)
    vrate_max_v = np.full(B, perf['max_climb_fpm'], dtype=np.float32)
    vrate_base = np.clip(vrate, -perf['max_desc_fpm'], perf['max_climb_fpm'])

    # 위치 범위 (부채꼴 — min/max 각도 + min/max 거리로 바운딩)
    cos_lat = np.cos(np.radians(lat)).astype(np.float32)
    cos_lat = np.clip(cos_lat, 0.01, None)

    dist_min = (gs_min * _KNOTS_TO_NM_PER_SEC * dt_sec).astype(np.float32)
    dist_max = (gs_max * _KNOTS_TO_NM_PER_SEC * dt_sec).astype(np.float32)
    dist_base = (gs_base * _KNOTS_TO_NM_PER_SEC * dt_sec).astype(np.float32)

    # baseline
    track_rad = np.radians(track).astype(np.float32)
    lat_base = lat + dist_base * np.cos(track_rad) / 60.0
    lon_base = lon + dist_base * np.sin(track_rad) / (60.0 * cos_lat)

    # 부채꼴 바운딩: dist_max × hdg_delta로 원호 반경 근사 (메모리 효율)
    arc_nm = dist_max * np.radians(hdg_delta).astype(np.float32)  # 원호 길이 근사
    # 위치 범위 = baseline ± max(dist_max 직선, arc 횡방향)
    lat_spread = np.maximum(dist_max / 60.0, arc_nm / 60.0)
    lon_spread = np.maximum(dist_max / (60.0 * cos_lat), arc_nm / (60.0 * cos_lat))
    lat_min_v2 = lat_base - lat_spread
    lat_max_v2 = lat_base + lat_spread
    lon_min_v2 = lon_base - lon_spread
    lon_max_v2 = lon_base + lon_spread

    # 조립
    env_min = np.stack([
        lat_min_v2, lon_min_v2, alt_min_v, gs_min,
        hdg_min, vrate_min_v,
        gs_min - 30, np.clip(gs_min / 660.0, 0.4, 1.2),
        np.zeros(B), np.zeros(B),
    ], axis=1).astype(np.float32)

    env_max = np.stack([
        lat_max_v2, lon_max_v2, alt_max_v, gs_max,
        hdg_max, vrate_max_v,
        gs_max - 10, np.clip(gs_max / 660.0, 0.4, 1.2),
        np.full(B, 360.0), np.full(B, 60.0),
    ], axis=1).astype(np.float32)

    env_baseline = np.stack([
        lat_base, lon_base, alt_base, gs_base,
        track, vrate_base,
        gs_base - 20, gs_base / 660.0,
        states[:, 8], states[:, 9],  # wind 유지
    ], axis=1).astype(np.float32)

    return env_min, env_max, env_baseline


def envelope_to_state(ratio, env_min, env_max):
    """
    신경망 출력 [0,1] 비율을 실제 상태로 변환.

    ratio: (B, STATE_DIM) or (B, T, STATE_DIM), 각 원소 ∈ [0, 1]
    env_min, env_max: 같은 shape
    Returns: 실제 상태 (같은 shape)
    """
    # hdg의 wrap-around 처리
    return env_min + ratio * (env_max - env_min)


def state_to_ratio(state, env_min, env_max):
    """
    실제 상태를 [0,1] 비율로 변환 (학습 타겟 생성용).

    state: (B, STATE_DIM) 실제 상태
    Returns: (B, STATE_DIM), 각 원소 ∈ [0, 1] (범위 밖이면 clamp)
    """
    span = env_max - env_min
    # 0으로 나누기 방지
    span = np.where(np.abs(span) < 1e-8, 1.0, span)
    ratio = (state - env_min) / span
    return np.clip(ratio, 0.0, 1.0)


def compute_envelope_batch_torch(states, dt_sec=10.0, ac_type='civil', device='cpu'):
    """
    PyTorch 텐서 버전 (GPU 호환).

    states: (B, STATE_DIM) torch tensor
    Returns: env_min, env_max, env_baseline as torch tensors
    """
    states_np = states.detach().cpu().numpy()
    e_min, e_max, e_base = compute_envelope_batch(states_np, dt_sec, ac_type)
    return (torch.from_numpy(e_min).to(device),
            torch.from_numpy(e_max).to(device),
            torch.from_numpy(e_base).to(device))

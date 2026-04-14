"""
Monte Carlo 기반 Conflict 확률 예측기

World Model (TrajectoryPredictor)을 사용하여:
1. 각 항공기의 미래 궤적을 확률적으로 N회 샘플링
2. 모든 항공기 쌍의 미래 분리 거리 계산
3. 분리 위반 확률 = conflict 확률 출력

SafetyAdvisor에 통합하여 기존 선형 외삽 대비:
- 기동 패턴 학습 기반의 현실적 예측
- 불확실성 정량화 (확률적 conflict 감지)
- 다체 interaction 반영
"""
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .dataset import STATE_DIM, CONTEXT_DIM, MAX_NEIGHBORS, NORM_MEAN, NORM_STD


@dataclass
class ConflictPrediction:
    """conflict 예측 결과"""
    icao_a: str
    icao_b: str
    probability: float          # P(conflict) 0~1
    expected_time_sec: float    # 예상 conflict 발생 시간 (초)
    min_h_dist_nm: float        # 예상 최소 수평 거리 (NM, 중앙값)
    min_v_dist_ft: float        # 예상 최소 수직 거리 (ft, 중앙값)
    worst_h_dist_nm: float      # 5th percentile 수평 거리
    worst_v_dist_ft: float      # 5th percentile 수직 거리
    confidence: float           # 예측 신뢰도 (모델 불확실성 기반)
    position: tuple = (0, 0)    # 예상 conflict 위치 (lat, lon)


def _haversine_nm_batch(lat1, lon1, lat2, lon2):
    """Batch haversine (numpy, NM)"""
    R = 3440.065
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    a = np.clip(a, 0, 1)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


class ConflictDetector:
    """
    World Model 기반 conflict 확률 예측기

    사용법:
        detector = ConflictDetector(model, device)
        predictions = detector.detect(aircraft_states, dt_sec=10)
    """

    def __init__(self, model, device='cuda',
                 num_mc_samples=50,
                 future_steps=12,
                 horiz_sep_nm=5.0,
                 vert_sep_ft=1000.0,
                 scan_radius_nm=80.0):
        """
        :param model: TrajectoryPredictor 인스턴스
        :param num_mc_samples: Monte Carlo 샘플 수
        :param future_steps: 예측 스텝 수
        :param horiz_sep_nm: 수평 분리 기준 (NM)
        :param vert_sep_ft: 수직 분리 기준 (ft)
        :param scan_radius_nm: 분석 대상 반경 (NM)
        """
        self.model = model
        self.device = device
        self.num_mc = num_mc_samples
        self.future_steps = future_steps
        self.horiz_sep = horiz_sep_nm
        self.vert_sep = vert_sep_ft
        self.scan_radius = scan_radius_nm

        self.norm_mean = NORM_MEAN
        self.norm_std = NORM_STD

        self._last_scan = 0
        self._cache = {}  # icao → recent states buffer
        self._callsign_map = {}  # icao → {callsign, moa}

    def update_state(self, icao, state_dict, callsign=None, moa=None):
        """
        항공기 상태 업데이트 (실시간 ADS-B 피드에서 호출)

        :param icao: 항공기 ICAO 코드
        :param state_dict: ADS-B 딕셔너리 (lat, lon, alt, gs, track, vrate, ...)
        """
        state = np.array([
            state_dict.get('lat', 0),
            state_dict.get('lon', 0),
            state_dict.get('baro_altitude_ft', 0) or state_dict.get('alt_current', 0),
            state_dict.get('ground_speed_kt', 0) or state_dict.get('spd_current', 0),
            state_dict.get('true_track_deg', 0) or state_dict.get('track_true_deg', 0),
            state_dict.get('vertical_rate_ft_min', 0),
            state_dict.get('ias_kt', 0),
            state_dict.get('mach', 0),
            state_dict.get('wind_direction_deg', 0),
            state_dict.get('wind_speed_kt', 0),
        ], dtype=np.float32)

        if icao not in self._cache:
            self._cache[icao] = []

        self._cache[icao].append(state)

        # callsign/moa 매핑 업데이트
        if callsign or moa:
            if icao not in self._callsign_map:
                self._callsign_map[icao] = {}
            if callsign:
                self._callsign_map[icao]['callsign'] = callsign
            if moa:
                self._callsign_map[icao]['moa'] = moa

        # 최근 20개만 유지 (약 200초)
        if len(self._cache[icao]) > 20:
            self._cache[icao] = self._cache[icao][-20:]

    def remove_stale(self, max_age_steps=12):
        """오래된 항공기 제거"""
        to_remove = [icao for icao, states in self._cache.items()
                     if len(states) < 2]
        for icao in to_remove:
            del self._cache[icao]

    def _build_context(self, target_icao, target_state, all_states):
        """target 항공기 주변의 neighbor context 생성"""
        ctx = np.zeros((MAX_NEIGHBORS, CONTEXT_DIM), dtype=np.float32)
        t_lat, t_lon, t_alt = target_state[0], target_state[1], target_state[2]
        t_gs, t_track, t_vr = target_state[3], target_state[4], target_state[5]

        neighbors = []
        for icao, states in all_states.items():
            if icao == target_icao or len(states) == 0:
                continue
            s = states[-1]
            lat, lon, alt = s[0], s[1], s[2]

            # 빠른 거리 필터 (degree 기반 근사)
            if abs(lat - t_lat) > 1.0 or abs(lon - t_lon) > 1.5:
                continue

            dx = (lon - t_lon) * 60.0 * math.cos(math.radians(t_lat))
            dy = (lat - t_lat) * 60.0
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 50:
                continue

            dalt = (alt - t_alt) / 1000.0
            dgs = (s[3] - t_gs) / 100.0
            dtrack = ((s[4] - t_track + 180) % 360 - 180) / 180.0
            dvrate = (s[5] - t_vr) / 1000.0
            neighbors.append((dist, [dx, dy, dalt, dgs, dtrack, dvrate]))

        neighbors.sort(key=lambda x: x[0])
        for i, (_, vec) in enumerate(neighbors[:MAX_NEIGHBORS]):
            ctx[i] = vec

        return ctx

    @torch.no_grad()
    def detect(self, dt_sec=10.0, past_steps=6, min_prob=0.01):
        """
        현재 캐시된 항공기 상태로 conflict 예측 실행

        :param dt_sec: 데이터 수집 간격 (초)
        :param past_steps: 사용할 과거 스텝 수
        :param min_prob: 반환할 최소 conflict 확률
        :return: list of ConflictPrediction
        """
        self.model.eval()

        # 충분한 히스토리가 있는 항공기만
        valid_icaos = [icao for icao, states in self._cache.items()
                       if len(states) >= past_steps]

        if len(valid_icaos) < 2:
            return []

        # 각 항공기의 과거 시퀀스 + context 준비
        past_seqs = {}
        ctx_seqs = {}

        for icao in valid_icaos:
            states = self._cache[icao][-past_steps:]
            past_arr = np.stack(states)  # (K, STATE_DIM)

            # 정규화
            past_norm = (past_arr - self.norm_mean) / self.norm_std

            # Context (각 타임스텝별)
            ctxs = []
            for i, s in enumerate(states):
                ctx = self._build_context(icao, s, self._cache)
                ctxs.append(ctx)
            ctx_arr = np.stack(ctxs)  # (K, MAX_N, CTX_DIM)

            past_seqs[icao] = torch.from_numpy(past_norm).to(self.device)
            ctx_seqs[icao] = torch.from_numpy(ctx_arr).to(self.device)

        # 배치로 예측
        icao_list = list(past_seqs.keys())
        B = len(icao_list)
        batch_past = torch.stack([past_seqs[ic] for ic in icao_list])  # (B, K, D)
        batch_ctx = torch.stack([ctx_seqs[ic] for ic in icao_list])  # (B, K, MAX_N, CTX)

        # Monte Carlo 궤적 예측: (B, MC, T, STATE_DIM) — 비정규화된 상태
        pred_trajs = self.model.predict(
            batch_past, batch_ctx,
            num_samples=self.num_mc,
            future_steps=self.future_steps)

        # numpy로 변환
        trajs_np = pred_trajs.cpu().numpy()  # (B, MC, T, D)

        # 군 관련 쌍만 conflict 계산 (민-민은 우리 관할 아님)
        predictions = []

        for i in range(B):
            for j in range(i + 1, B):
                icao_a = icao_list[i]
                icao_b = icao_list[j]

                # 민-민 스킵: 둘 다 PF_가 아니면 민항기끼리
                a_is_mil = icao_a.startswith('PF_')
                b_is_mil = icao_b.startswith('PF_')
                if not a_is_mil and not b_is_mil:
                    continue

                # 같은 공역 전투기끼리 스킵 (훈련 중)
                if a_is_mil and b_is_mil:
                    moa_a = self._callsign_map.get(icao_a, {}).get('moa', '')
                    moa_b = self._callsign_map.get(icao_b, {}).get('moa', '')
                    if moa_a and moa_b and moa_a == moa_b:
                        continue

                # 현재 거리 체크 (scan radius 밖이면 스킵)
                cur_a = self._cache[icao_a][-1]
                cur_b = self._cache[icao_b][-1]
                cur_dist = _haversine_nm_batch(
                    np.array([cur_a[0]]), np.array([cur_a[1]]),
                    np.array([cur_b[0]]), np.array([cur_b[1]]))[0]
                if cur_dist > self.scan_radius:
                    continue

                pred = self._analyze_pair(
                    icao_a, icao_b,
                    trajs_np[i], trajs_np[j],
                    dt_sec)

                if pred and pred.probability >= min_prob:
                    # icao → callsign 변환
                    pred.icao_a = self._callsign_map.get(icao_a, {}).get(
                        'callsign', icao_a)
                    pred.icao_b = self._callsign_map.get(icao_b, {}).get(
                        'callsign', icao_b)
                    predictions.append(pred)

        # 확률 높은 순 정렬
        predictions.sort(key=lambda p: -p.probability)
        return predictions

    def _analyze_pair(self, icao_a, icao_b, traj_a, traj_b, dt_sec):
        """
        두 항공기 쌍의 Monte Carlo 궤적에서 conflict 분석

        :param traj_a: (MC, T, STATE_DIM) 비정규화된 예측 궤적
        :param traj_b: (MC, T, STATE_DIM) 비정규화된 예측 궤적
        :return: ConflictPrediction or None
        """
        MC, T, _ = traj_a.shape

        # 각 Monte Carlo 샘플, 각 타임스텝에서의 분리 거리
        lat_a, lon_a, alt_a = traj_a[:, :, 0], traj_a[:, :, 1], traj_a[:, :, 2]
        lat_b, lon_b, alt_b = traj_b[:, :, 0], traj_b[:, :, 1], traj_b[:, :, 2]

        # 수평 거리 (MC, T)
        h_dist = _haversine_nm_batch(lat_a, lon_a, lat_b, lon_b)
        # 수직 거리 (MC, T)
        v_dist = np.abs(alt_a - alt_b)

        # Conflict 판정: 수평 < 5NM AND 수직 < 1000ft
        conflict_mask = (h_dist < self.horiz_sep) & (v_dist < self.vert_sep)

        # 각 MC 샘플에서 conflict 발생 여부
        conflict_any = conflict_mask.any(axis=1)  # (MC,)
        probability = conflict_any.mean()

        if probability < 0.001:
            return None

        # 통계 계산
        min_h_per_sample = h_dist.min(axis=1)  # (MC,)
        min_v_per_sample = v_dist.min(axis=1)  # (MC,)

        # conflict 발생 시간 (첫 위반 타임스텝)
        conflict_times = []
        for mc in range(MC):
            steps = np.where(conflict_mask[mc])[0]
            if len(steps) > 0:
                conflict_times.append(steps[0] * dt_sec)

        expected_time = np.median(conflict_times) if conflict_times else T * dt_sec

        # 예상 conflict 위치 (conflict 발생 시점의 중간점)
        if conflict_times:
            first_step = int(np.median([np.where(conflict_mask[mc])[0][0]
                                        for mc in range(MC) if conflict_mask[mc].any()]))
            first_step = min(first_step, T - 1)
            pos_lat = (lat_a[:, first_step].mean() + lat_b[:, first_step].mean()) / 2
            pos_lon = (lon_a[:, first_step].mean() + lon_b[:, first_step].mean()) / 2
        else:
            pos_lat, pos_lon = 0, 0

        # 신뢰도: 궤적 분산이 작을수록 높음
        spread_a = np.std(lat_a[:, -1]) * 60  # NM 단위 위치 분산
        spread_b = np.std(lat_b[:, -1]) * 60
        avg_spread = (spread_a + spread_b) / 2
        confidence = max(0, min(1, 1.0 - avg_spread / 20.0))

        return ConflictPrediction(
            icao_a=icao_a,
            icao_b=icao_b,
            probability=float(probability),
            expected_time_sec=float(expected_time),
            min_h_dist_nm=float(np.median(min_h_per_sample)),
            min_v_dist_ft=float(np.median(min_v_per_sample)),
            worst_h_dist_nm=float(np.percentile(min_h_per_sample, 5)),
            worst_v_dist_ft=float(np.percentile(min_v_per_sample, 5)),
            confidence=float(confidence),
            position=(float(pos_lat), float(pos_lon)),
        )

    def get_risk_matrix(self):
        """
        모든 추적 항공기 쌍의 conflict 위험도 매트릭스

        Returns:
            dict[(icao_a, icao_b)] → probability
        """
        predictions = self.detect()
        matrix = {}
        for p in predictions:
            matrix[(p.icao_a, p.icao_b)] = p.probability
            matrix[(p.icao_b, p.icao_a)] = p.probability
        return matrix

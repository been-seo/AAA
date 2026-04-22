"""
Physics-based 궤적 예측기: 항공기 카테고리별 예측 전략

- RouteFollowingPredictor: IFR 항로 추종 (거의 결정론적)
- TransitPredictor: 직선 이동 (기지/공항 향)
- MOABoundedPredictor: 공역 내 확률적 기동
"""
import math
import random

import numpy as np

from utils.geo import calculate_distance, calculate_bearing


# 물리 제약
CIVIL_HDG_RATE = 3.0       # °/sec (표준율 선회)
MILITARY_HDG_RATE = 6.0    # °/sec (전투기)
CIVIL_CLIMB_RATE = 2000    # ft/min
CIVIL_DESC_RATE = 1500     # ft/min
MIL_CLIMB_RATE = 6000      # ft/min

KNOTS_TO_NM_PER_SEC = 1.0 / 3600.0

# STATE_DIM 매핑 (비정규화 좌표)
# [lat, lon, alt, gs, track, vrate, ias, mach, wind_dir, wind_spd]
STATE_DIM = 10


def _advance_position(lat, lon, track_deg, gs_kt, dt_sec):
    """현재 위치에서 dt초 후 위치 계산"""
    dist_nm = gs_kt * KNOTS_TO_NM_PER_SEC * dt_sec
    hdg_rad = math.radians(track_deg)
    new_lat = lat + dist_nm * math.cos(hdg_rad) / 60.0
    cos_lat = math.cos(math.radians(lat))
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6
    new_lon = lon + dist_nm * math.sin(hdg_rad) / (60.0 * cos_lat)
    return new_lat, new_lon


def _turn_toward(current_hdg, target_hdg, max_rate, dt_sec):
    """목표 헤딩으로 제한된 선회"""
    diff = (target_hdg - current_hdg + 180) % 360 - 180
    max_turn = max_rate * dt_sec
    if abs(diff) <= max_turn:
        return target_hdg % 360
    return (current_hdg + math.copysign(max_turn, diff)) % 360


def _state_array(lat, lon, alt, gs, track, vrate, ias=0, mach=0,
                 wind_dir=0, wind_spd=0):
    """STATE_DIM 배열 생성"""
    return np.array([lat, lon, alt, gs, track, vrate,
                     ias or gs * 0.93, mach or gs / 660.0,
                     wind_dir, wind_spd], dtype=np.float32)


class RouteFollowingPredictor:
    """
    IFR 항로 추종 예측: WP 시퀀스를 따라 물리 기반 직진/선회

    항로 위 B737이 BOPTA→KARMA로 가면, WP까지 직진 후 다음 WP로 선회.
    거의 결정론적 — MC 샘플은 미세 노이즈만 추가.
    """

    def __init__(self, dt_sec=10.0, hdg_rate=CIVIL_HDG_RATE,
                 wp_arrival_nm=3.0):
        self.dt = dt_sec
        self.hdg_rate = hdg_rate
        self.wp_arrival_nm = wp_arrival_nm

    def predict(self, state, route_wps, future_steps=12,
                num_samples=50, confidence=0.9):
        """
        :param state: dict {lat, lon, alt, gs, track, vrate, ...}
        :param route_wps: list of {name, lat, lon} — 남은 웨이포인트
        :param confidence: 높을수록 분산 작음
        :return: (num_samples, future_steps, STATE_DIM) numpy array
        """
        lat = state.get("lat", 0)
        lon = state.get("lon", 0)
        alt = state.get("alt", 0) or state.get("baro_altitude_ft", 0)
        gs = state.get("gs", 0) or state.get("ground_speed_kt", 0)
        track = state.get("track", 0) or state.get("true_track_deg", 0)
        vrate = state.get("vrate", 0) or state.get("vertical_rate_ft_min", 0)

        if gs < 50:
            gs = 250  # 정지 시 기본값

        # 결정론적 궤적 생성
        det_traj = self._deterministic_trajectory(
            lat, lon, alt, gs, track, vrate, route_wps, future_steps)

        # MC 샘플: 상황 의존적 노이즈
        # 시간/위치/상태에 따라 spread가 달라짐
        samples = np.zeros((num_samples, future_steps, STATE_DIM),
                           dtype=np.float32)

        # 각 타임스텝의 context-dependent spread 계산
        step_spreads = self._compute_step_spreads(
            det_traj, route_wps, gs, vrate, confidence, future_steps)

        for s in range(num_samples):
            for t in range(future_steps):
                base = det_traj[t].copy()
                sp = step_spreads[t]

                # 위치 노이즈 (NM → degree)
                base[0] += np.random.normal(0, sp['lateral'] / 60.0)
                cos_l = max(math.cos(math.radians(base[0])), 0.5)
                base[1] += np.random.normal(0, sp['lateral'] / (60.0 * cos_l))
                # 고도 노이즈
                base[2] += np.random.normal(0, sp['vertical'])
                # 속도 노이즈
                base[3] += np.random.normal(0, sp['speed'])
                base[3] = max(100, base[3])
                # 헤딩 노이즈
                base[4] = (base[4] + np.random.normal(0, sp['heading'])) % 360
                samples[s, t] = base

        return samples

    def _compute_step_spreads(self, det_traj, route_wps, gs, vrate,
                               confidence, future_steps):
        """
        시간축별 context-dependent spread 계산

        고려 요소:
        - 시간: 멀수록 불확실성 증가 (sqrt 비례)
        - WP 근접: 선회 가능성 → lateral 증가
        - 고도 변경: vrate 있으면 vertical 증가
        - 속도: 빠를수록 위치 불확실성 빠르게 성장
        - confidence: 항로 매칭 확신도
        """
        spreads = []
        base_lateral = 0.1 * (1.0 - confidence * 0.7)  # NM, 기본값
        speed_factor = gs / 450.0  # 450kt 기준 정규화

        for t in range(future_steps):
            # 시간 성장 (sqrt: 초기 급증 후 완화)
            time_factor = math.sqrt((t + 1) / future_steps)

            # WP 근접도: det_traj에서 다음 WP까지 거리
            wp_factor = 1.0
            if route_wps:
                traj_lat, traj_lon = det_traj[t][0], det_traj[t][1]
                for wp in route_wps[:2]:  # 가까운 2개만
                    d = calculate_distance(traj_lat, traj_lon,
                                          wp["lat"], wp["lon"])
                    if d < 5.0:  # 5NM 이내 = 선회 구간
                        wp_factor = 1.0 + (5.0 - d) / 5.0 * 2.0  # 최대 3배
                        break

            # 고도 변경 불확실성
            vert_base = 50.0  # ft
            if abs(vrate) > 300:
                vert_base = 200.0  # 상승/하강 중이면 4배

            lateral = base_lateral * time_factor * wp_factor * speed_factor
            vertical = vert_base * time_factor
            speed_noise = 5.0 * time_factor
            heading_noise = 0.5 * time_factor * wp_factor

            spreads.append({
                'lateral': lateral,
                'vertical': vertical,
                'speed': speed_noise,
                'heading': heading_noise,
            })

        return spreads

    def _deterministic_trajectory(self, lat, lon, alt, gs, track, vrate,
                                   route_wps, future_steps):
        """물리 기반 결정론적 궤적"""
        traj = np.zeros((future_steps, STATE_DIM), dtype=np.float32)

        wp_idx = 0
        cur_lat, cur_lon = lat, lon
        cur_alt, cur_gs = alt, gs
        cur_track = track
        cur_vrate = vrate

        for t in range(future_steps):
            # 다음 WP 결정
            if wp_idx < len(route_wps):
                wp = route_wps[wp_idx]
                wp_lat, wp_lon = wp["lat"], wp["lon"]
                dist_to_wp = calculate_distance(
                    cur_lat, cur_lon, wp_lat, wp_lon)

                if dist_to_wp < self.wp_arrival_nm:
                    wp_idx += 1
                    if wp_idx < len(route_wps):
                        wp = route_wps[wp_idx]
                        wp_lat, wp_lon = wp["lat"], wp["lon"]

                # WP를 향해 선회
                target_hdg = calculate_bearing(
                    cur_lat, cur_lon, wp_lat, wp_lon)
                cur_track = _turn_toward(
                    cur_track, target_hdg, self.hdg_rate, self.dt)
            # WP 다 소진 → 현재 헤딩 유지 (직진)

            # 위치 이동
            cur_lat, cur_lon = _advance_position(
                cur_lat, cur_lon, cur_track, cur_gs, self.dt)

            # 고도: vrate 유지 (강하/상승은 지속됨)
            if abs(cur_vrate) > 100:
                cur_alt += cur_vrate * self.dt / 60.0
                # 강하: 지면/최저고도에서 멈춤
                if cur_vrate < 0:
                    cur_alt = max(0, cur_alt)
                    # 속도도 강하 시 점진 감소 (접근 감속)
                    if cur_alt < 10000:
                        cur_gs = max(180, cur_gs - 2)
                # 상승: 순항고도 도달 시 수평비행 전환
                if cur_vrate > 0 and cur_alt >= 35000:
                    cur_vrate = 0
                    # vrate는 거의 유지 (0.98 = 매우 느린 감쇠)
                else:
                    cur_vrate *= 0.98

            traj[t] = _state_array(
                cur_lat, cur_lon, cur_alt, cur_gs, cur_track, cur_vrate)

        return traj


class TransitPredictor:
    """
    직선 이동 예측: 기지/공항 방향으로 직선, 물리 제약 적용

    군용기의 기지↔공역 이동, 또는 목적지로 향하는 항공기.
    """

    def __init__(self, dt_sec=10.0, hdg_rate=CIVIL_HDG_RATE):
        self.dt = dt_sec
        self.hdg_rate = hdg_rate

    def predict(self, state, dest, future_steps=12, num_samples=50,
                is_military=False):
        """
        :param state: dict {lat, lon, alt, gs, track, ...}
        :param dest: dict {lat, lon, ...}
        :return: (num_samples, future_steps, STATE_DIM)
        """
        lat = state.get("lat", 0)
        lon = state.get("lon", 0)
        alt = state.get("alt", 0) or state.get("baro_altitude_ft", 0)
        gs = state.get("gs", 0) or state.get("ground_speed_kt", 0)
        track = state.get("track", 0) or state.get("true_track_deg", 0)
        vrate = state.get("vrate", 0) or state.get("vertical_rate_ft_min", 0)

        if gs < 50:
            gs = 350 if is_military else 250

        hdg_rate = MILITARY_HDG_RATE if is_military else self.hdg_rate
        noise_lat = 0.5 if is_military else 0.3  # NM

        dest_lat, dest_lon = dest["lat"], dest["lon"]
        total_dist = calculate_distance(lat, lon, dest_lat, dest_lon)

        samples = np.zeros((num_samples, future_steps, STATE_DIM),
                           dtype=np.float32)

        for s in range(num_samples):
            cur_lat, cur_lon = lat, lon
            cur_alt, cur_gs = alt, gs
            cur_track = track
            cur_vrate = vrate

            for t in range(future_steps):
                # 목적지 방향
                target_hdg = calculate_bearing(
                    cur_lat, cur_lon, dest_lat, dest_lon)
                cur_track = _turn_toward(
                    cur_track, target_hdg, hdg_rate, self.dt)

                # 이동
                cur_lat, cur_lon = _advance_position(
                    cur_lat, cur_lon, cur_track, cur_gs, self.dt)

                # 고도: vrate 유지 (강하/상승은 지속)
                if abs(cur_vrate) > 100:
                    cur_alt += cur_vrate * self.dt / 60.0
                    cur_alt = max(0, cur_alt)
                    cur_vrate *= 0.98  # 매우 느린 감쇠
                    # 강하 중 속도 감소
                    if cur_vrate < -500 and cur_alt < 10000:
                        cur_gs = max(180, cur_gs - 2)
                else:
                    # vrate 없으면 거리 기반 강하/감속
                    remain = calculate_distance(
                        cur_lat, cur_lon, dest_lat, dest_lon)
                    if remain < 30:
                        cur_alt = max(1000, cur_alt - 500 * self.dt / 60.0)
                        cur_gs = max(180, cur_gs - 5)

                # 노이즈
                n_lat = cur_lat + np.random.normal(0, noise_lat / 60.0)
                cos_l = max(math.cos(math.radians(cur_lat)), 0.5)
                n_lon = cur_lon + np.random.normal(0, noise_lat / (60.0 * cos_l))
                n_alt = cur_alt + np.random.normal(0, 100)
                n_gs = cur_gs + np.random.normal(0, 10)
                n_trk = (cur_track + np.random.normal(0, 1)) % 360

                samples[s, t] = _state_array(
                    n_lat, n_lon, n_alt, max(100, n_gs), n_trk, cur_vrate)

        return samples


class ApproachPredictor:
    """
    공항 접근/장주 영역 기반 예측 (Reachability region)

    공항 근처 항공기는 특정 궤적이 아니라 '공항 주변 어딘가'로 예측.
    거리가 가까울수록 공항 중심으로 분포 수렴.

    Mixture components:
      C1 (40%): 현재 궤적 직진 유지
      C2 (30%): 공항 중심으로 수렴 (direct approach)
      C3 (15%): 장주 패턴 진입 (활주로 방향 기준 측풍)
      C4 (10%): Go-around 준비 (상승)
      C5 (5%): Holding 패턴 (원형 대기)
    """

    def __init__(self, dt_sec=10.0, hdg_rate=CIVIL_HDG_RATE):
        self.dt = dt_sec
        self.hdg_rate = hdg_rate
        self.mixture_weights = {
            'direct': 0.40,
            'airport_center': 0.30,
            'pattern': 0.15,
            'go_around': 0.10,
            'holding': 0.05,
        }

    def _airport_sigma(self, dist_to_apt_nm, t_step):
        """공항 거리 기반 불확실성 반경 (NM)"""
        # 시간 성장
        time_sigma = 0.3 * math.sqrt(t_step + 1)
        # 공항 인력: 가까울수록 넓어짐
        if dist_to_apt_nm > 50:
            apt_sigma = 0
        elif dist_to_apt_nm > 20:
            # 50NM → 0, 20NM → 5NM 선형
            apt_sigma = 5.0 * (50 - dist_to_apt_nm) / 30
        elif dist_to_apt_nm > 10:
            # 20NM → 5NM, 10NM → 8NM
            apt_sigma = 5 + 3 * (20 - dist_to_apt_nm) / 10
        else:
            # <10NM: 공항 반경 자체
            apt_sigma = 8.0 + 2.0 * (10 - dist_to_apt_nm) / 10
        return time_sigma + apt_sigma

    def _sample_mode(self):
        """Mixture component 샘플링"""
        r = random.random()
        cum = 0
        for mode, w in self.mixture_weights.items():
            cum += w
            if r < cum:
                return mode
        return 'direct'

    def predict(self, state, airport, future_steps=12, num_samples=50,
                 runway_track=None):
        """
        :param state: dict {lat, lon, alt, gs, track, vrate, ...}
        :param airport: dict {lat, lon, icao, ...}
        :param runway_track: 활주로 방향 (deg), 모르면 None
        :return: (num_samples, future_steps, STATE_DIM)
        """
        lat = state.get("lat", 0)
        lon = state.get("lon", 0)
        alt = state.get("alt", 0) or state.get("baro_altitude_ft", 0)
        gs = state.get("gs", 0) or state.get("ground_speed_kt", 0)
        track = state.get("track", 0) or state.get("true_track_deg", 0)
        vrate = state.get("vrate", 0) or state.get("vertical_rate_ft_min", 0)

        if gs < 50:
            gs = 200

        apt_lat, apt_lon = airport["lat"], airport["lon"]
        apt_alt = airport.get("alt", 0)  # airport elevation
        init_dist = calculate_distance(lat, lon, apt_lat, apt_lon)

        samples = np.zeros((num_samples, future_steps, STATE_DIM),
                           dtype=np.float32)

        for s in range(num_samples):
            mode = self._sample_mode()

            cur_lat, cur_lon = lat, lon
            cur_alt, cur_gs = alt, gs
            cur_track = track
            cur_vrate = vrate

            # Mode별 초기 파라미터
            if mode == 'go_around':
                cur_vrate = max(cur_vrate, 1500)  # 급상승 시작
            elif mode == 'pattern' and runway_track is not None:
                # 장주 downwind = 활주로 반대 방향
                cur_track = (runway_track + 180) % 360
            elif mode == 'holding':
                # 홀딩 시작 시 속도/고도 유지
                cur_vrate = 0

            for t in range(future_steps):
                cur_dist = calculate_distance(
                    cur_lat, cur_lon, apt_lat, apt_lon)

                # Mode별 target 결정
                if mode == 'direct':
                    # 공항 직진
                    target_hdg = calculate_bearing(
                        cur_lat, cur_lon, apt_lat, apt_lon)
                elif mode == 'airport_center':
                    # 공항 방향 기본 + 끌림 강함
                    target_hdg = calculate_bearing(
                        cur_lat, cur_lon, apt_lat, apt_lon)
                elif mode == 'pattern':
                    # 장주 패턴: 선회 주기
                    target_hdg = cur_track + 90 * math.sin(t * 0.3)
                elif mode == 'go_around':
                    # 복행: 초기 활주로 방향, 이후 선회
                    if t < 3:
                        target_hdg = runway_track or cur_track
                    else:
                        target_hdg = cur_track + 30  # 선회
                elif mode == 'holding':
                    # 홀딩: 공항 주위 원형
                    target_hdg = (cur_track + 15) % 360
                else:
                    target_hdg = cur_track

                # 선회
                cur_track = _turn_toward(
                    cur_track, target_hdg, self.hdg_rate, self.dt)

                # 이동
                cur_lat, cur_lon = _advance_position(
                    cur_lat, cur_lon, cur_track, cur_gs, self.dt)

                # 고도: mode별
                if mode == 'go_around':
                    cur_alt = min(cur_alt + cur_vrate * self.dt / 60.0, 5000)
                    cur_vrate *= 0.95
                elif mode in ('direct', 'airport_center') and cur_dist < 30:
                    # 강하
                    target_desc = 500 + (30 - cur_dist) * 50
                    cur_alt = max(apt_alt, cur_alt - target_desc * self.dt / 60.0)
                    cur_gs = max(140, cur_gs - 3)
                elif mode == 'holding':
                    pass  # 고도 유지
                elif abs(cur_vrate) > 100:
                    cur_alt += cur_vrate * self.dt / 60.0
                    cur_alt = max(apt_alt, cur_alt)
                    cur_vrate *= 0.98

                # 공항 중심 인력 (airport_center 모드 강화)
                if mode == 'airport_center' and cur_dist > 2:
                    pull = 0.3  # 매 스텝 공항 쪽으로 30% 이동
                    cur_lat = cur_lat * (1 - pull * 0.1) + apt_lat * (pull * 0.1)
                    cur_lon = cur_lon * (1 - pull * 0.1) + apt_lon * (pull * 0.1)

                # 불확실성 노이즈: 공항 거리 기반
                sigma_nm = self._airport_sigma(cur_dist, t)
                n_lat = cur_lat + np.random.normal(0, sigma_nm / 60.0)
                cos_l = max(math.cos(math.radians(cur_lat)), 0.5)
                n_lon = cur_lon + np.random.normal(
                    0, sigma_nm / (60.0 * cos_l))
                n_alt = cur_alt + np.random.normal(0, 200)
                n_gs = cur_gs + np.random.normal(0, 10)
                n_trk = (cur_track + np.random.normal(0, 3)) % 360

                samples[s, t] = _state_array(
                    n_lat, n_lon, n_alt, max(100, n_gs), n_trk, cur_vrate)

        return samples


class MOABoundedPredictor:
    """
    공역 내 확률적 기동 예측: MOA 경계 내에서 랜덤 WP 순회

    전투기의 MOA 내 훈련 기동을 시뮬레이션.
    높은 분산, MOA 경계에서 clamp.
    """

    def __init__(self, dt_sec=10.0, hdg_rate=MILITARY_HDG_RATE):
        self.dt = dt_sec
        self.hdg_rate = hdg_rate

    def predict(self, state, moa_vertices, future_steps=12,
                num_samples=50, min_alt=5000, max_alt=45000):
        """
        :param state: dict {lat, lon, alt, gs, track, ...}
        :param moa_vertices: list of (lat, lon) — MOA 경계 다각형
        :return: (num_samples, future_steps, STATE_DIM)
        """
        lat = state.get("lat", 0)
        lon = state.get("lon", 0)
        alt = state.get("alt", 0) or state.get("baro_altitude_ft", 0)
        gs = state.get("gs", 0) or state.get("ground_speed_kt", 0)
        track = state.get("track", 0) or state.get("true_track_deg", 0)

        if gs < 100:
            gs = 400

        # MOA 중심점
        lats = [v[0] for v in moa_vertices]
        lons = [v[1] for v in moa_vertices]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        samples = np.zeros((num_samples, future_steps, STATE_DIM),
                           dtype=np.float32)

        for s in range(num_samples):
            # 각 샘플마다 랜덤 WP 2~3개 생성
            wps = []
            for _ in range(random.randint(2, 3)):
                wp = self._random_point_in_polygon(
                    moa_vertices, min_alt, max_alt)
                if wp:
                    wps.append(wp)

            cur_lat, cur_lon = lat, lon
            cur_alt, cur_gs = alt, gs
            cur_track = track
            wp_idx = 0

            for t in range(future_steps):
                # 현재 WP로 이동
                if wp_idx < len(wps):
                    wp = wps[wp_idx]
                    dist = calculate_distance(
                        cur_lat, cur_lon, wp[0], wp[1])
                    if dist < 3.0:
                        wp_idx += 1
                        if wp_idx < len(wps):
                            wp = wps[wp_idx]

                    target_hdg = calculate_bearing(
                        cur_lat, cur_lon, wp[0], wp[1])
                else:
                    # WP 소진 → 랜덤 선회
                    target_hdg = (cur_track + random.uniform(-30, 30)) % 360

                cur_track = _turn_toward(
                    cur_track, target_hdg, self.hdg_rate, self.dt)

                # 속도 변동
                cur_gs += random.uniform(-10, 10)
                cur_gs = max(250, min(600, cur_gs))

                # 고도 변동
                cur_alt += random.uniform(-500, 500) * self.dt / 60.0
                cur_alt = max(min_alt, min(max_alt, cur_alt))

                # 이동
                cur_lat, cur_lon = _advance_position(
                    cur_lat, cur_lon, cur_track, cur_gs, self.dt)

                # MOA 경계 체크: 밖이면 중심으로 복귀
                from .world_context import _point_in_polygon
                if not _point_in_polygon(cur_lat, cur_lon, moa_vertices):
                    cur_track = calculate_bearing(
                        cur_lat, cur_lon, center_lat, center_lon)
                    cur_lat, cur_lon = _advance_position(
                        cur_lat, cur_lon, cur_track, cur_gs, self.dt)

                samples[s, t] = _state_array(
                    cur_lat, cur_lon, cur_alt, cur_gs, cur_track, 0)

        return samples

    def _random_point_in_polygon(self, vertices, min_alt, max_alt):
        """다각형 내 랜덤 위치"""
        from .world_context import _point_in_polygon
        lats = [v[0] for v in vertices]
        lons = [v[1] for v in vertices]
        for _ in range(50):
            lat = random.uniform(min(lats), max(lats))
            lon = random.uniform(min(lons), max(lons))
            if _point_in_polygon(lat, lon, vertices):
                return (lat, lon, random.uniform(
                    max(min_alt, 5000), min(max_alt, 45000)))
        return None

"""
공역 관리 (MOA Hot/Cold, 공항 접근 금지, Hot MOA 임무 전투기)
다각형/원형 공역 모두 지원. planner용으로는 원형 근사를 반환.
"""
import math
import time
import random
import string
import threading

from config import (
    MOA_LIST, MOA_TOGGLE_INTERVAL_SEC,
    R_ZONE_LIST, R_ZONE_TOGGLE_INTERVAL_SEC,
    AIRPORTS, AIRPORT_EXCLUSION_RADIUS_NM, AIRPORT_EXCLUSION_ALT_FT,
    STATIC_OBSTACLES, KNOTS_TO_MPS, M_TO_NM,
)
from utils.geo import calculate_distance, calculate_bearing


# ── 임무 전투기 설정 ──
PATROL_MIN_FIGHTERS = 2
PATROL_MAX_FIGHTERS = 4
PATROL_WP_COUNT = 4          # 웨이포인트 개수
PATROL_WP_ARRIVAL_NM = 3.0   # WP 도달 판정 거리
PATROL_SPD_MIN = 300          # kts
PATROL_SPD_MAX = 550
PATROL_HDG_RATE = 3.0         # °/sec (표준율 선회)
PATROL_SEP_HORIZ_NM = 5.0    # 임무기 간 수평 분리 기준
PATROL_SEP_VERT_FT = 1000.0  # 임무기 간 수직 분리 기준
PATROL_AVOID_HDG_DEG = 30.0   # 회피 선회각
PATROL_AVOID_ALT_FT = 2000.0  # 회피 고도 변경
PATROL_BOUNDARY_BUFFER_NM = 5.0  # MOA 경계 버퍼 (이 이내면 중심 복귀)


def _polygon_to_circle(airspace):
    """다각형 공역 → 원형 근사 (planner 호환). vertices가 없으면 그대로 반환."""
    verts = airspace.get("vertices")
    if not verts:
        # 이미 원형 (DMZ 등)
        return airspace

    lats = [v[0] for v in verts]
    lons = [v[1] for v in verts]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    lat_span = (max(lats) - min(lats)) * 60 / 2
    lon_span = (max(lons) - min(lons)) * 60 * 0.8 / 2  # cos(36) 보정
    radius_nm = max(lat_span, lon_span, 2.0)

    return {
        "name": airspace["name"],
        "lat": center_lat,
        "lon": center_lon,
        "radius_nm": radius_nm,
        "min_alt": airspace["min_alt"],
        "max_alt": airspace["max_alt"],
    }


def _random_callsign_4():
    """랜덤 4글자 대문자 콜사인 생성"""
    return ''.join(random.choices(string.ascii_uppercase, k=4))


def _point_in_polygon(lat, lon, vertices):
    """Ray casting point-in-polygon"""
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = vertices[i]
        yj, xj = vertices[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _dist_to_polygon_border(lat, lon, vertices):
    """점에서 다각형 경계까지 최소 거리 (NM). 내부면 양수, 외부면 음수."""
    inside = _point_in_polygon(lat, lon, vertices)
    min_dist = float('inf')
    n = len(vertices)
    for i in range(n):
        y1, x1 = vertices[i]
        y2, x2 = vertices[(i + 1) % n]
        # 점→선분 최근접점까지 거리 (위경도 기반 근사)
        dx, dy = x2 - x1, y2 - y1
        denom = dx * dx + dy * dy
        if denom < 1e-12:
            d = calculate_distance(lat, lon, y1, x1)
        else:
            t = max(0, min(1, ((lon - x1) * dx + (lat - y1) * dy) / denom))
            d = calculate_distance(lat, lon, y1 + t * dy, x1 + t * dx)
        if d < min_dist:
            min_dist = d
    return min_dist if inside else -min_dist


def _random_point_in_polygon(vertices, min_alt, max_alt):
    """다각형 내 랜덤 위치 생성 (rejection sampling)"""
    lats = [v[0] for v in vertices]
    lons = [v[1] for v in vertices]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    for _ in range(200):
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        if _point_in_polygon(lat, lon, vertices):
            alt = random.uniform(max(min_alt, 5000), min(max_alt, 45000))
            return lat, lon, alt
    # fallback: 중심점
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    alt = random.uniform(max(min_alt, 5000), min(max_alt, 45000))
    return center_lat, center_lon, alt


class PatrolFighter:
    """
    Hot MOA 내부에서 임무하는 전투기.
    웨이포인트를 순회하며, 다른 임무기와의 분리를 유지한다.
    """

    def __init__(self, moa_name, vertices, min_alt, max_alt):
        self.moa_name = moa_name
        self.vertices = vertices
        self.min_alt = min_alt
        self.max_alt = max_alt
        self.callsign = _random_callsign_4()

        # 중심점 캐시
        lats = [v[0] for v in vertices]
        lons = [v[1] for v in vertices]
        self._center_lat = sum(lats) / len(lats)
        self._center_lon = sum(lons) / len(lons)

        # 초기 위치
        lat, lon, alt = _random_point_in_polygon(vertices, min_alt, max_alt)
        self.lat = lat
        self.lon = lon
        self.alt = float(alt)
        self.spd = random.uniform(PATROL_SPD_MIN, PATROL_SPD_MAX)
        self.hdg = random.uniform(0, 360)

        # 웨이포인트
        self.waypoints = []
        self.wp_idx = 0
        self._generate_waypoints()

        # 회피 상태
        self._avoiding = False
        self._avoid_timer = 0.0
        self._border_check_timer = 0.0  # 경계 체크 주기 제한

    def _generate_waypoints(self):
        """MOA 내부에 새 웨이포인트 세트 생성 (경계 버퍼 내 제외)"""
        self.waypoints = []
        for _ in range(PATROL_WP_COUNT):
            # 경계에서 최소 버퍼 거리 이상 안쪽에 생성
            for _attempt in range(50):
                lat, lon, alt = _random_point_in_polygon(
                    self.vertices, self.min_alt, self.max_alt)
                if _dist_to_polygon_border(lat, lon, self.vertices) >= PATROL_BOUNDARY_BUFFER_NM:
                    break
            self.waypoints.append((lat, lon, alt))
        self.wp_idx = 0

    def update(self, dt, other_fighters):
        """1프레임 업데이트: 웨이포인트 추종 + 분리 유지"""
        if dt <= 0:
            return

        # 회피 타이머
        if self._avoiding:
            self._avoid_timer -= dt
            if self._avoid_timer <= 0:
                self._avoiding = False

        # 분리 위반 체크 → 회피 (경계 체크 타이밍에 함께 수행)
        if not self._avoiding and self._border_check_timer >= 1.9:
            for other in other_fighters:
                if other is self:
                    continue
                h_dist = calculate_distance(self.lat, self.lon, other.lat, other.lon)
                v_dist = abs(self.alt - other.alt)
                if h_dist < PATROL_SEP_HORIZ_NM and v_dist < PATROL_SEP_VERT_FT:
                    self._start_avoidance(other)
                    break

        # 목표 결정
        if self._avoiding:
            # 회피 중에는 현재 hdg 유지 (이미 _start_avoidance에서 설정)
            target_hdg = self.hdg
        else:
            # 웨이포인트 추종
            if not self.waypoints:
                self._generate_waypoints()
            wp = self.waypoints[self.wp_idx]
            dist_to_wp = calculate_distance(self.lat, self.lon, wp[0], wp[1])

            if dist_to_wp < PATROL_WP_ARRIVAL_NM:
                self.wp_idx += 1
                if self.wp_idx >= len(self.waypoints):
                    self._generate_waypoints()
                wp = self.waypoints[self.wp_idx]

            target_hdg = calculate_bearing(self.lat, self.lon, wp[0], wp[1])

            # 목표 고도 서서히 추종
            alt_diff = wp[2] - self.alt
            if abs(alt_diff) > 100:
                rate = 2000.0 / 60.0  # 2000 ft/min
                self.alt += math.copysign(min(rate * dt, abs(alt_diff)), alt_diff)
            # 고도 범위 클램프
            self.alt = max(self.min_alt, min(self.max_alt, self.alt))

        # 헤딩 선회 (표준율)
        hdg_diff = (target_hdg - self.hdg + 180) % 360 - 180
        max_turn = PATROL_HDG_RATE * dt
        if abs(hdg_diff) > 0.5:
            self.hdg = (self.hdg + math.copysign(min(max_turn, abs(hdg_diff)), hdg_diff)) % 360
        else:
            self.hdg = target_hdg % 360

        # 속도 변동 (랜덤 drift)
        self.spd += random.uniform(-2, 2) * dt
        self.spd = max(PATROL_SPD_MIN, min(PATROL_SPD_MAX, self.spd))

        # 이동
        speed_nm_s = self.spd * KNOTS_TO_MPS * M_TO_NM
        dist = speed_nm_s * dt
        hdg_rad = math.radians(self.hdg)
        self.lat += dist * math.cos(hdg_rad) / 60.0
        cos_lat = math.cos(math.radians(self.lat))
        if abs(cos_lat) >= 1e-6:
            self.lon += dist * math.sin(hdg_rad) / (60.0 * cos_lat)

        # MOA 경계 버퍼 체크 (2초마다, 매 프레임은 비용이 큼)
        self._border_check_timer -= dt
        if self._border_check_timer <= 0:
            self._border_check_timer = 2.0
            border_dist = _dist_to_polygon_border(self.lat, self.lon, self.vertices)
            if border_dist < PATROL_BOUNDARY_BUFFER_NM:
                self.hdg = calculate_bearing(
                    self.lat, self.lon, self._center_lat, self._center_lon)
                self._avoiding = False
                if border_dist < 0:
                    self._generate_waypoints()

    def _start_avoidance(self, other):
        """다른 임무기와의 분리 유지를 위한 회피 기동"""
        self._avoiding = True
        self._avoid_timer = 15.0  # 15초간 회피 유지

        brg_to_other = calculate_bearing(self.lat, self.lon, other.lat, other.lon)
        # 반대 방향으로 선회
        hdg_diff = (brg_to_other - self.hdg + 180) % 360 - 180
        if hdg_diff > 0:
            self.hdg = (self.hdg - PATROL_AVOID_HDG_DEG) % 360
        else:
            self.hdg = (self.hdg + PATROL_AVOID_HDG_DEG) % 360

        # 고도도 분리
        if self.alt >= other.alt:
            self.alt = min(self.max_alt, self.alt + PATROL_AVOID_ALT_FT)
        else:
            self.alt = max(self.min_alt, self.alt - PATROL_AVOID_ALT_FT)

    def to_aircraft_dict(self):
        """SafetyAdvisor 호환 딕셔너리"""
        return {
            "callsign": self.callsign,
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "spd": self.spd,
            "hdg": self.hdg,
            "moa": self.moa_name,
        }


class AirspaceManager:
    """
    MOA/R구역 Hot/Cold 상태 관리 + 공항 접근 금지 구역 + Hot 공역 임무 전투기

    사용법:
        mgr = AirspaceManager(destination_icao="RKPK")
        obstacles = mgr.get_obstacles()  # planner용 (원형 근사)
        mgr.update(dt)                   # 매 프레임 호출 (토글 체크 + 임무기 업데이트)
        fighters = mgr.get_patrol_fighters()  # 모든 임무 전투기 리스트
    """

    def __init__(self, destination_icao=None):
        self.destination_icao = destination_icao

        # MOA 상태
        self.moa_states = {}
        for moa in MOA_LIST:
            self.moa_states[moa["name"]] = random.choice([True, False])
        self._last_moa_toggle = time.time()

        # R구역 상태 (제한공역 Hot/Cold)
        self.rzone_states = {}
        for rz in R_ZONE_LIST:
            self.rzone_states[rz["name"]] = random.choice([True, False])
        self._last_rzone_toggle = time.time()

        self._lock = threading.Lock()

        # 임무 전투기: {airspace_name: [PatrolFighter, ...]}
        self.patrol_fighters = {}
        self._init_patrol_fighters()

    def _init_patrol_fighters(self):
        """초기 Hot 공역(MOA+R)에 임무 전투기 배치"""
        with self._lock:
            for moa in MOA_LIST:
                name = moa["name"]
                if self.moa_states.get(name, False):
                    self._spawn_fighters_for_moa(moa)
            for rz in R_ZONE_LIST:
                name = rz["name"]
                if self.rzone_states.get(name, False):
                    self._spawn_fighters_for_moa(rz)

    def _spawn_fighters_for_moa(self, moa):
        """MOA 내부에 임무 전투기 2~4대 생성"""
        verts = moa.get("vertices")
        if not verts or len(verts) < 3:
            return
        name = moa["name"]
        count = random.randint(PATROL_MIN_FIGHTERS, PATROL_MAX_FIGHTERS)
        fighters = []
        for _ in range(count):
            f = PatrolFighter(name, verts, moa["min_alt"], moa["max_alt"])
            fighters.append(f)
        self.patrol_fighters[name] = fighters

    def _remove_fighters_for_moa(self, moa_name):
        """MOA의 임무 전투기 제거"""
        self.patrol_fighters.pop(moa_name, None)

    def update(self, dt=0):
        """매 프레임 호출: MOA/R구역 토글 + 임무 전투기 업데이트"""
        now = time.time()

        # MOA 토글
        if now - self._last_moa_toggle >= MOA_TOGGLE_INTERVAL_SEC:
            with self._lock:
                toggle_count = random.randint(1, min(3, len(MOA_LIST)))
                toggled = random.sample(list(self.moa_states.keys()), toggle_count)
                for name in toggled:
                    old = self.moa_states[name]
                    self.moa_states[name] = not old
                    print(f"[Airspace] MOA {name} -> {'HOT' if not old else 'COLD'}")
                    if old and not self.moa_states[name]:
                        self._remove_fighters_for_moa(name)
                    elif not old and self.moa_states[name]:
                        moa = next((m for m in MOA_LIST if m["name"] == name), None)
                        if moa:
                            self._spawn_fighters_for_moa(moa)
                self._last_moa_toggle = now

        # R구역 토글
        # R97/R108 (서해 사격장)은 매우 드물게 활성화 (실사격 빈도 반영)
        _RARE_RZONE_PREFIXES = ("R97", "R108")
        if R_ZONE_LIST and now - self._last_rzone_toggle >= R_ZONE_TOGGLE_INTERVAL_SEC:
            with self._lock:
                toggle_count = random.randint(1, min(3, len(R_ZONE_LIST)))
                toggled = random.sample(list(self.rzone_states.keys()), toggle_count)
                for name in toggled:
                    old = self.rzone_states[name]
                    # 사격장은 HOT 전환 확률 5% (COLD 전환은 항상 허용)
                    if not old and name.startswith(_RARE_RZONE_PREFIXES):
                        if random.random() > 0.005:
                            continue
                    self.rzone_states[name] = not old
                    print(f"[Airspace] R-zone {name} -> {'HOT' if not old else 'COLD'}")
                    if old and not self.rzone_states[name]:
                        self._remove_fighters_for_moa(name)
                    elif not old and self.rzone_states[name]:
                        rz = next((r for r in R_ZONE_LIST if r["name"] == name), None)
                        if rz:
                            self._spawn_fighters_for_moa(rz)
                self._last_rzone_toggle = now

        # 임무 전투기 업데이트 (lock 최소화: 리스트 복사 후 외부에서 업데이트)
        if dt > 0:
            with self._lock:
                snapshot = [(name, list(fighters)) for name, fighters in self.patrol_fighters.items()]
            for moa_name, fighters in snapshot:
                for f in fighters:
                    f.update(dt, fighters)

    def is_moa_hot(self, name):
        with self._lock:
            return self.moa_states.get(name, False)

    def is_rzone_hot(self, name):
        with self._lock:
            return self.rzone_states.get(name, False)

    def get_moa_status(self):
        with self._lock:
            return dict(self.moa_states)

    def get_rzone_status(self):
        with self._lock:
            return dict(self.rzone_states)

    def get_patrol_fighters(self):
        """모든 임무 전투기 리스트 반환"""
        with self._lock:
            result = []
            for fighters in self.patrol_fighters.values():
                result.extend(fighters)
            return list(result)

    def get_patrol_fighters_by_moa(self, moa_name):
        """특정 MOA의 임무 전투기만 반환"""
        with self._lock:
            return list(self.patrol_fighters.get(moa_name, []))

    def get_obstacles(self):
        """
        planner용 장애물 리스트 (원형 근사, lat/lon/radius_nm 형식)
        """
        obstacles = [_polygon_to_circle(o) for o in STATIC_OBSTACLES]

        with self._lock:
            # Hot MOA
            for moa in MOA_LIST:
                if self.moa_states.get(moa["name"], False):
                    obstacles.append(_polygon_to_circle(moa))
            # Hot R구역
            for rz in R_ZONE_LIST:
                if self.rzone_states.get(rz["name"], False):
                    obstacles.append(_polygon_to_circle(rz))

        for apt in AIRPORTS:
            if apt["icao"] == self.destination_icao:
                continue  # 목적지만 접근 가능, 나머지 전부 접근 금지
            obstacles.append({
                "name": f"APT_{apt['icao']}",
                "lat": apt["lat"],
                "lon": apt["lon"],
                "radius_nm": AIRPORT_EXCLUSION_RADIUS_NM,
                "min_alt": 0,
                "max_alt": AIRPORT_EXCLUSION_ALT_FT,
            })

        return obstacles

    def get_patrol_traffic_vectors(self):
        """임무 전투기를 planner용 트래픽 벡터 형식으로 반환"""
        with self._lock:
            vectors = []
            for fighters in self.patrol_fighters.values():
                for pf in fighters:
                    vectors.append({
                        'lat': pf.lat, 'lon': pf.lon, 'alt': pf.alt,
                        'spd_kts': pf.spd, 'track_deg': pf.hdg,
                        'vrate_fpm': 0,
                    })
            return vectors

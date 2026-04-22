"""
World Context Builder: 항공기의 "세계"를 파악하여 예측 전략 결정

항공기가 어디에 있고, 뭘 하고 있고, 어디로 가는지를 판단:
- IFR 민항기: 항로 매칭 → Route-following 예측
- VFR 경항공기: 일반 방향 + 넓은 분산
- 군용기 MOA 내: 공역 경계 내 확률적 예측
- 군용기 transit: 기지↔공역 직선 예측
- 야간/주말 MOA 관통: Cold MOA를 직항로로 자유 통과
"""
import math
import json
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from utils.geo import calculate_distance, calculate_bearing


class FlightMode(Enum):
    ROUTE_FOLLOWING = "route_following"   # IFR 항로 추종
    APPROACH = "approach"                 # 공항 접근 (강하 중)
    DEPARTURE = "departure"              # 공항 출발 (상승 중)
    TRANSIT = "transit"                   # 목적지로 직선 이동
    MOA_PATROL = "moa_patrol"             # 공역 내 기동
    MOA_TRANSIT = "moa_transit"           # Cold MOA 관통 직항 (야간/주말)
    VFR_FREE = "vfr_free"                # VFR 자유비행
    UNKNOWN = "unknown"


class AircraftCategory(Enum):
    IFR_CIVIL = "ifr_civil"       # 대형/중형 민항기 (B737, A320 등)
    VFR_LIGHT = "vfr_light"       # 경항공기 (C172, PA28 등)
    MILITARY = "military"          # 군용기
    UNKNOWN = "unknown"


@dataclass
class AircraftContext:
    """항공기의 world context — 예측 전략 결정에 사용"""
    flight_mode: FlightMode = FlightMode.UNKNOWN
    category: AircraftCategory = AircraftCategory.UNKNOWN

    # 항로 매칭 (ROUTE_FOLLOWING)
    matched_route: Optional[str] = None
    route_wps: List[dict] = field(default_factory=list)  # 남은 WP 리스트
    route_confidence: float = 0.0  # 0~1

    # MOA (MOA_PATROL / MOA_TRANSIT)
    moa_name: Optional[str] = None
    moa_vertices: Optional[list] = None
    moa_is_hot: bool = True  # True=활성(군 사용중), False=Cold(자유통과)

    # Transit / Approach / Departure 목적지
    transit_dest: Optional[dict] = None  # {icao, lat, lon, dist_nm}
    approach_airport: Optional[dict] = None  # 접근 중인 공항

    # 기본 정보
    heading_stability: float = 0.0   # 0~1, 헤딩 안정도
    altitude_band: str = "cruising"  # climbing/cruising/descending


def _point_in_polygon(lat, lon, vertices):
    """Ray casting point-in-polygon"""
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = vertices[i]
        yj, xj = vertices[j]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _dist_to_segment_nm(lat, lon, lat1, lon1, lat2, lon2):
    """점에서 선분까지 최소 거리 (NM, 위경도 근사)"""
    dx, dy = lon2 - lon1, lat2 - lat1
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return calculate_distance(lat, lon, lat1, lon1)
    t = max(0, min(1, ((lon - lon1) * dx + (lat - lat1) * dy) / denom))
    closest_lat = lat1 + t * dy
    closest_lon = lon1 + t * dx
    return calculate_distance(lat, lon, closest_lat, closest_lon)


def _bearing_between(lat1, lon1, lat2, lon2):
    """두 점 사이 bearing (degrees)"""
    return calculate_bearing(lat1, lon1, lat2, lon2)


class WorldContextBuilder:
    """항공기 상태 + 공역 데이터로 world context 생성"""

    def __init__(self, routes_path=None, moa_list=None, rzone_list=None,
                 airports=None, airspace_manager=None):
        # 항로 데이터
        self.routes = {}
        self.all_wps = []  # (name, lat, lon, route_names)
        if routes_path:
            self._load_routes(routes_path)
        elif routes_path is None:
            # 기본 경로
            default = Path(__file__).parent.parent.parent / "ats_routes_named.json"
            if default.exists():
                self._load_routes(str(default))

        # 공역 데이터
        self.moa_list = moa_list or []
        self.rzone_list = rzone_list or []
        self.airports = airports or []

        # AirspaceManager 참조 (MOA Hot/Cold 실시간 상태)
        self.airspace_manager = airspace_manager

        # 공역이 없으면 config에서 로드
        if not self.moa_list:
            self._load_airspace_from_config()

        # WP 좌표 numpy 배열 (빠른 검색용)
        if self.all_wps:
            self._wp_lats = np.array([w[1] for w in self.all_wps])
            self._wp_lons = np.array([w[2] for w in self.all_wps])
        else:
            self._wp_lats = np.array([])
            self._wp_lons = np.array([])

    def _load_routes(self, path):
        """ats_routes_named.json 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.routes = data.get("routes", {})

        # 모든 WP를 중복 제거하여 수집 (어떤 항로에 속하는지 기록)
        wp_map = {}  # name → {lat, lon, routes}
        for route_name, wps in self.routes.items():
            for wp in wps:
                name = wp["name"]
                if name not in wp_map:
                    wp_map[name] = {
                        "lat": wp["lat"], "lon": wp["lon"],
                        "routes": set()
                    }
                wp_map[name]["routes"].add(route_name)

        self.all_wps = [
            (name, info["lat"], info["lon"], list(info["routes"]))
            for name, info in wp_map.items()
        ]

    def _load_airspace_from_config(self):
        """config.py에서 공역/공항 데이터 로드"""
        try:
            from config import MOA_LIST, R_ZONE_LIST, AIRPORTS
            self.moa_list = MOA_LIST
            self.rzone_list = R_ZONE_LIST
            self.airports = AIRPORTS
        except ImportError:
            pass

    def classify(self, state, history=None, adsb_info=None, data_time=None):
        """
        항공기 상태로 world context 판단

        :param state: dict {lat, lon, alt, gs, track, vrate, ...}
        :param history: list of recent states (oldest first), optional
        :param adsb_info: dict {category, aircraft_model, icao}, optional
        :param data_time: 데이터 타임스탬프 (datetime 또는 epoch float).
                          MOA 야간/주말 판단에 사용. None이면 AirspaceManager 상태만 참조.
        :return: AircraftContext
        """
        ctx = AircraftContext()
        lat = state.get("lat", 0)
        lon = state.get("lon", 0)
        alt = state.get("alt", 0) or state.get("baro_altitude_ft", 0)
        gs = state.get("gs", 0) or state.get("ground_speed_kt", 0)
        track = state.get("track", 0) or state.get("true_track_deg", 0)
        vrate = state.get("vrate", 0) or state.get("vertical_rate_ft_min", 0)

        # 1. 항공기 카테고리 추정
        ctx.category = self._classify_category(gs, alt, adsb_info)

        # 2. 고도 밴드
        if vrate > 300:
            ctx.altitude_band = "climbing"
        elif vrate < -300:
            ctx.altitude_band = "descending"
        else:
            ctx.altitude_band = "cruising"

        # 3. 헤딩 안정도 (히스토리 있으면)
        ctx.heading_stability = self._compute_heading_stability(history)

        # 4. MOA/R-zone 안에 있는지 체크
        moa = self._check_moa_containment(lat, lon, alt)
        if moa:
            ctx.moa_name = moa["name"]
            ctx.moa_vertices = moa.get("vertices")
            ctx.moa_is_hot = self._is_moa_hot(moa["name"], data_time)

        # 5. 카테고리별 flight mode 결정
        if ctx.category == AircraftCategory.MILITARY:
            if moa and ctx.moa_is_hot:
                ctx.flight_mode = FlightMode.MOA_PATROL
            elif moa and not ctx.moa_is_hot:
                # 군용기인데 Cold MOA 안 → transit 중일 가능성
                dest = self._find_nearest_military_base(lat, lon, track)
                if dest:
                    ctx.flight_mode = FlightMode.TRANSIT
                    ctx.transit_dest = dest
                else:
                    ctx.flight_mode = FlightMode.MOA_PATROL  # 잔류 가능
            else:
                # 군용기인데 MOA 밖 → 기지로 이동 중
                dest = self._find_nearest_military_base(lat, lon, track)
                if dest:
                    ctx.flight_mode = FlightMode.TRANSIT
                    ctx.transit_dest = dest
                else:
                    ctx.flight_mode = FlightMode.UNKNOWN

        elif ctx.category == AircraftCategory.IFR_CIVIL:
            # 우선 체크: 공항 근처 + 수직 상태로 approach/departure 판단
            apt_phase = self._check_airport_phase(lat, lon, alt, track, vrate)
            if apt_phase:
                ctx.flight_mode = apt_phase["mode"]
                ctx.approach_airport = apt_phase["airport"]
                ctx.transit_dest = apt_phase["airport"]

            # Cold MOA 안에 있으면 → 야간/주말 직항 관통
            elif moa and not ctx.moa_is_hot:
                dest = self._find_nearest_airport_ahead(
                    lat, lon, track, civil_only=False)
                if dest:
                    ctx.flight_mode = FlightMode.MOA_TRANSIT
                    ctx.transit_dest = dest
                else:
                    ctx.flight_mode = FlightMode.MOA_TRANSIT

            else:
                # 항로 매칭 시도
                route_match = self._match_route(lat, lon, track, alt)
                if route_match:
                    ctx.flight_mode = FlightMode.ROUTE_FOLLOWING
                    ctx.matched_route = route_match["route"]
                    ctx.route_wps = route_match["remaining_wps"]
                    ctx.route_confidence = route_match["confidence"]
                else:
                    # 항로 못 찾았지만 안정적 직진 → transit
                    if ctx.heading_stability > 0.8:
                        dest = self._find_nearest_airport_ahead(
                            lat, lon, track, civil_only=True)
                        if dest:
                            ctx.flight_mode = FlightMode.TRANSIT
                            ctx.transit_dest = dest
                        else:
                            ctx.flight_mode = FlightMode.UNKNOWN
                    else:
                        ctx.flight_mode = FlightMode.UNKNOWN

        elif ctx.category == AircraftCategory.VFR_LIGHT:
            ctx.flight_mode = FlightMode.VFR_FREE

        return ctx

    def _classify_category(self, gs, alt, adsb_info=None):
        """ADS-B + 속도/고도 기반 카테고리 추정"""
        if adsb_info:
            icao = adsb_info.get("icao", "")
            # 시뮬레이션 군용기
            if icao.startswith("PF_"):
                return AircraftCategory.MILITARY

            model = adsb_info.get("aircraft_model", "").upper()
            # 군용기 타입 (실제 ADS-B에는 거의 없지만)
            mil_types = {"F16", "F15", "F35", "KF21", "FA50", "T50",
                         "C130", "KC135", "E737"}
            if any(t in model for t in mil_types):
                return AircraftCategory.MILITARY

            # 경항공기
            light_types = {"C172", "C152", "PA28", "PA32", "DA40", "DA42",
                           "SR22", "P28A", "C182", "BE36"}
            if any(t in model for t in light_types):
                return AircraftCategory.VFR_LIGHT

            # ADS-B category 필드
            cat = adsb_info.get("category", 0)
            if isinstance(cat, int):
                if cat >= 0xA3:  # A3=Large, A4=B757, A5=Heavy
                    return AircraftCategory.IFR_CIVIL
                elif cat == 0xA1:  # Light
                    return AircraftCategory.VFR_LIGHT

        # 속도/고도 휴리스틱 (완화: approach/descent 고려)
        # 대형기는 approach 중 속도 낮고 고도 낮음 (gs 180+, alt 1k+)
        if gs >= 180 and alt >= 1000:
            return AircraftCategory.IFR_CIVIL
        elif gs < 130 and alt < 8000:
            return AircraftCategory.VFR_LIGHT

        return AircraftCategory.UNKNOWN

    def _compute_heading_stability(self, history):
        """과거 히스토리에서 헤딩 안정도 계산 (0=불안정, 1=완전 직진)"""
        if not history or len(history) < 3:
            return 0.5  # 정보 부족 → 중간값

        tracks = []
        for s in history[-6:]:
            t = s.get("track", 0) or s.get("true_track_deg", 0)
            tracks.append(t)

        if len(tracks) < 2:
            return 0.5

        # 연속 헤딩 변화의 절대값 평균
        deltas = []
        for i in range(1, len(tracks)):
            d = (tracks[i] - tracks[i-1] + 180) % 360 - 180
            deltas.append(abs(d))

        avg_delta = sum(deltas) / len(deltas)
        # 0° 변화 → 1.0, 10° 변화 → 0.0
        stability = max(0, 1.0 - avg_delta / 10.0)
        return stability

    def _check_moa_containment(self, lat, lon, alt):
        """MOA/R-zone 안에 있는지 체크"""
        for moa in self.moa_list + self.rzone_list:
            verts = moa.get("vertices")
            if not verts:
                continue
            if alt < moa.get("min_alt", 0) or alt > moa.get("max_alt", 60000):
                continue
            if _point_in_polygon(lat, lon, verts):
                return moa
        return None

    def _check_airport_phase(self, lat, lon, alt, track, vrate):
        """
        공항 근처에서 강하율/상승률로 approach/departure 판단

        - 50NM 이내 + 강하 500fpm 이상 + 헤딩이 공항 향함 → APPROACH
        - 30NM 이내 + 상승 500fpm 이상 → DEPARTURE
        """
        if abs(vrate) < 500:
            return None  # 수평비행이면 해당 없음

        for apt in self.airports:
            dist = calculate_distance(lat, lon, apt["lat"], apt["lon"])
            bearing_to_apt = calculate_bearing(lat, lon, apt["lat"], apt["lon"])
            hdg_offset = abs((track - bearing_to_apt + 180) % 360 - 180)

            if vrate < -500 and dist < 50 and hdg_offset < 45:
                # 강하 중 + 공항 50NM 이내 + 공항 방향 → APPROACH
                return {
                    "mode": FlightMode.APPROACH,
                    "airport": {
                        "icao": apt["icao"], "name": apt["name"],
                        "lat": apt["lat"], "lon": apt["lon"],
                        "dist_nm": dist,
                    }
                }
            elif vrate > 500 and dist < 30:
                # 상승 중 + 공항 30NM 이내 → DEPARTURE
                return {
                    "mode": FlightMode.DEPARTURE,
                    "airport": {
                        "icao": apt["icao"], "name": apt["name"],
                        "lat": apt["lat"], "lon": apt["lon"],
                        "dist_nm": dist,
                    }
                }

        return None

    def _is_moa_hot(self, moa_name, data_time=None):
        """
        MOA가 활성(Hot)인지 판단.

        판단 순서:
        1. AirspaceManager가 있으면 실시간 상태 사용 (가장 정확)
        2. data_time이 있으면 시간대로 추정 (야간/주말 = Cold)
        3. 둘 다 없으면 보수적으로 Hot 가정
        """
        # 1. AirspaceManager 실시간 상태
        if self.airspace_manager:
            if hasattr(self.airspace_manager, 'is_moa_hot'):
                return self.airspace_manager.is_moa_hot(moa_name)
            if hasattr(self.airspace_manager, 'is_rzone_hot'):
                return self.airspace_manager.is_rzone_hot(moa_name)

        # 2. 데이터 타임스탬프 기반 추정
        if data_time is not None:
            if isinstance(data_time, (int, float)):
                data_time = datetime.fromtimestamp(data_time)
            if isinstance(data_time, datetime):
                hour = data_time.hour  # KST (한국 로컬 데이터)
                weekday = data_time.weekday()  # 0=월 ~ 6=일
                # 야간 (21:00 ~ 07:00) 또는 주말 → Cold 가능성 높음
                is_night = hour >= 21 or hour < 7
                is_weekend = weekday >= 5  # 토, 일
                if is_night or is_weekend:
                    return False  # Cold

        # 3. 기본: 보수적으로 Hot
        return True

    def _match_route(self, lat, lon, track, alt, max_dist_nm=5.0,
                     max_hdg_offset=20.0):
        """
        현재 위치/헤딩으로 가장 적합한 항로 매칭

        Returns: {route, remaining_wps, confidence} or None
        """
        if not self.routes:
            return None

        best = None
        best_score = float('inf')

        for route_name, wps in self.routes.items():
            for i in range(len(wps) - 1):
                wp_a = wps[i]
                wp_b = wps[i + 1]

                # 선분까지 거리
                dist = _dist_to_segment_nm(
                    lat, lon,
                    wp_a["lat"], wp_a["lon"],
                    wp_b["lat"], wp_b["lon"])

                if dist > max_dist_nm:
                    continue

                # 선분 방향과 현재 헤딩 비교
                seg_bearing = _bearing_between(
                    wp_a["lat"], wp_a["lon"],
                    wp_b["lat"], wp_b["lon"])

                hdg_offset = abs((track - seg_bearing + 180) % 360 - 180)

                # 역방향 허용 (항로 양방향)
                if hdg_offset > 90:
                    hdg_offset = 180 - hdg_offset
                    # 역방향이면 WP 순서 뒤집기
                    direction = -1
                else:
                    direction = 1

                if hdg_offset > max_hdg_offset:
                    continue

                # 점수: 거리 + 헤딩 오프셋 (낮을수록 좋음)
                score = dist + hdg_offset * 0.1

                if score < best_score:
                    best_score = score

                    # 남은 WP 리스트 구성
                    if direction == 1:
                        remaining = wps[i + 1:]
                    else:
                        remaining = list(reversed(wps[:i + 1]))

                    # 신뢰도: 거리 < 1NM & 헤딩 < 5° → 높음
                    if dist < 1.0 and hdg_offset < 5.0:
                        confidence = 0.95
                    elif dist < 3.0 and hdg_offset < 10.0:
                        confidence = 0.7
                    else:
                        confidence = 0.4

                    best = {
                        "route": route_name,
                        "remaining_wps": remaining,
                        "confidence": confidence,
                        "dist_nm": dist,
                        "hdg_offset": hdg_offset,
                    }

        return best

    def _find_nearest_military_base(self, lat, lon, track,
                                     max_dist_nm=100):
        """헤딩 방향에 있는 가장 가까운 군 기지"""
        best = None
        best_dist = max_dist_nm

        for apt in self.airports:
            if not apt.get("mil", False):
                continue
            dist = calculate_distance(lat, lon, apt["lat"], apt["lon"])
            if dist > max_dist_nm:
                continue
            bearing = calculate_bearing(lat, lon, apt["lat"], apt["lon"])
            hdg_offset = abs((track - bearing + 180) % 360 - 180)
            if hdg_offset > 45:
                continue
            if dist < best_dist:
                best_dist = dist
                best = {
                    "icao": apt["icao"],
                    "name": apt["name"],
                    "lat": apt["lat"],
                    "lon": apt["lon"],
                    "dist_nm": dist,
                }

        return best

    def _find_nearest_airport_ahead(self, lat, lon, track,
                                     max_dist_nm=150, civil_only=False):
        """헤딩 방향 앞에 있는 가장 가까운 공항"""
        best = None
        best_dist = max_dist_nm

        for apt in self.airports:
            if civil_only and apt.get("mil", False):
                continue
            dist = calculate_distance(lat, lon, apt["lat"], apt["lon"])
            if dist > max_dist_nm:
                continue
            bearing = calculate_bearing(lat, lon, apt["lat"], apt["lon"])
            hdg_offset = abs((track - bearing + 180) % 360 - 180)
            if hdg_offset > 30:
                continue
            if dist < best_dist:
                best_dist = dist
                best = {
                    "icao": apt["icao"],
                    "name": apt["name"],
                    "lat": apt["lat"],
                    "lon": apt["lon"],
                    "dist_nm": dist,
                }

        return best

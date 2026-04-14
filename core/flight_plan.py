"""
Flight Plan 추출기

레코딩 데이터를 사전 스캔하여 각 항공기의 이륙 이벤트를 추출.
실제 운용에서는 국토교통부가 Flight Plan을 뿌리므로,
레코딩에서 추출한 이륙 이벤트가 그 역할을 대신함.

추출 항목:
  - 이륙 시각 (timestamp)
  - 출발 공항 (가장 가까운 공항)
  - 목적지 방향 (궤적 분석, 가장 가까운 도착 공항)
  - callsign, icao24
  - 순항 고도/속도 (궤적 최대값)

사용:
  planner = FlightPlanExtractor(airports)
  planner.scan_jsonl("data/recordings/xxx.jsonl")
  upcoming = planner.get_upcoming_departures(current_ts, lookahead_sec=600)
"""
import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class FlightPlan:
    """추출된 비행 계획"""
    icao24: str
    callsign: str
    dep_time: float           # 이륙 시각 (epoch)
    dep_airport: str          # 출발 공항 ICAO (e.g. "RKSI")
    dep_airport_name: str     # 출발 공항 이름
    dest_airport: Optional[str] = None       # 도착 공항 ICAO (추론)
    dest_airport_name: Optional[str] = None
    heading_deg: float = 0.0  # 이륙 후 초기 방향
    cruise_alt_ft: float = 0.0  # 순항 고도 (궤적 최대)
    cruise_spd_kt: float = 0.0  # 순항 속도
    aircraft_model: str = ""
    registration: str = ""


def _haversine_nm(lat1, lon1, lat2, lon2):
    """두 좌표 간 거리 (NM)"""
    R = 3440.065  # 지구 반경 NM
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class FlightPlanExtractor:
    """레코딩에서 Flight Plan 추출"""

    def __init__(self, airports, dep_radius_nm=8.0, arr_radius_nm=15.0,
                 min_cruise_alt_ft=5000):
        """
        :param airports: config.AIRPORTS 리스트
        :param dep_radius_nm: 이 반경 내에서 이륙 감지
        :param arr_radius_nm: 이 반경 내에서 도착 추론
        :param min_cruise_alt_ft: 이 고도 이상이면 airborne 확정
        """
        self.airports = airports
        self.dep_radius = dep_radius_nm
        self.arr_radius = arr_radius_nm
        self.min_cruise_alt = min_cruise_alt_ft
        self.plans: List[FlightPlan] = []
        self._plans_by_time: List[FlightPlan] = []  # dep_time 정렬

    def _nearest_airport(self, lat, lon, radius_nm=None):
        """가장 가까운 공항 반환. 반경 밖이면 None."""
        best = None
        best_dist = float('inf')
        for ap in self.airports:
            d = _haversine_nm(lat, lon, ap['lat'], ap['lon'])
            if d < best_dist:
                best_dist = d
                best = ap
        if radius_nm and best_dist > radius_nm:
            return None, best_dist
        return best, best_dist

    def scan_jsonl(self, jsonl_path, progress_interval=500):
        """
        JSONL 레코딩을 스캔하여 Flight Plan 추출.

        알고리즘:
        1. 각 항공기의 전체 궤적을 수집
        2. 이륙 감지: on_ground→airborne 전환 또는 공항 근처 저고도 첫 출현
        3. 목적지 추론: 궤적 마지막 위치에서 가장 가까운 공항
        """
        print(f"[FlightPlan] Scanning: {jsonl_path}")

        # 1단계: 항공기별 궤적 수집
        trajectories: Dict[str, list] = {}  # icao24 → [(ts, ac_dict), ...]
        snap_count = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts = obj.get('timestamp', 0)
                aircraft = obj.get('aircraft', [])
                snap_count += 1

                for ac in aircraft:
                    lat = ac.get('lat')
                    lon = ac.get('lon')
                    if lat is None or lon is None:
                        continue
                    icao = ac.get('icao24', '')
                    if not icao:
                        continue
                    if icao not in trajectories:
                        trajectories[icao] = []
                    trajectories[icao].append((ts, ac))

                if snap_count % progress_interval == 0:
                    print(f"  [FlightPlan] {snap_count} snapshots, "
                          f"{len(trajectories)} aircraft...")

        print(f"  [FlightPlan] Loaded {snap_count} snapshots, "
              f"{len(trajectories)} unique aircraft")

        # 2단계: 각 항공기에서 이륙 이벤트 추출
        self.plans = []
        for icao, traj in trajectories.items():
            if len(traj) < 3:
                continue
            traj.sort(key=lambda x: x[0])
            plans = self._extract_departures(icao, traj)
            self.plans.extend(plans)

        # 시간순 정렬
        self.plans.sort(key=lambda p: p.dep_time)
        self._plans_by_time = self.plans

        print(f"[FlightPlan] Extracted {len(self.plans)} departure plans")
        return self.plans

    def scan_db(self, db_path, progress_interval=500):
        """SQLite DB 레코딩에서 Flight Plan 추출."""
        from core.adsb_db import ADSBDatabase

        print(f"[FlightPlan] Scanning DB: {db_path}")
        db = ADSBDatabase(db_path, readonly=True)

        # 모든 ICAO 가져오기
        icaos = [row['icao24'] for row in db._conn.execute(
            "SELECT DISTINCT icao24 FROM aircraft WHERE lat IS NOT NULL"
        ).fetchall()]
        print(f"  [FlightPlan] {len(icaos)} unique aircraft in DB")

        self.plans = []
        for i, icao in enumerate(icaos):
            rows = db._conn.execute(
                "SELECT s.timestamp, a.* FROM aircraft a "
                "JOIN snapshots s ON s.id = a.snapshot_id "
                "WHERE a.icao24 = ? AND a.lat IS NOT NULL "
                "ORDER BY s.timestamp",
                (icao,)
            ).fetchall()

            if len(rows) < 3:
                continue

            traj = [(row['timestamp'], dict(row)) for row in rows]
            plans = self._extract_departures(icao, traj)
            self.plans.extend(plans)

            if (i + 1) % progress_interval == 0:
                print(f"  [FlightPlan] {i+1}/{len(icaos)} aircraft processed...")

        db.close()

        self.plans.sort(key=lambda p: p.dep_time)
        self._plans_by_time = self.plans
        print(f"[FlightPlan] Extracted {len(self.plans)} departure plans")
        return self.plans

    def _extract_departures(self, icao, traj):
        """
        한 항공기의 궤적에서 이륙 이벤트를 추출.
        같은 항공기가 여러 번 이착륙할 수 있으므로 leg 단위로 분리.

        Leg 분리 기준:
        - on_ground 0→1 전환 (착륙) → 1→0 전환 (이륙)
        - 또는 타임스탬프 갭 > 30분 (레이더 범위 밖 갔다 돌아옴)
        """
        plans = []

        # 궤적을 leg로 분리
        legs = self._split_into_legs(traj)

        for leg in legs:
            plan = self._extract_single_departure(icao, leg)
            if plan:
                plans.append(plan)

        return plans

    def _split_into_legs(self, traj, gap_threshold_sec=1800):
        """궤적을 leg 단위로 분리. on_ground 전환 또는 시간 갭 기준."""
        legs = []
        current_leg = []

        for i, (ts, ac) in enumerate(traj):
            og = ac.get('on_ground', 0)

            # 시간 갭 체크
            if current_leg:
                prev_ts = current_leg[-1][0]
                if ts - prev_ts > gap_threshold_sec:
                    if current_leg:
                        legs.append(current_leg)
                    current_leg = []

            # on_ground → airborne 전환 = 새 leg 시작
            if current_leg and og != 1:
                prev_og = current_leg[-1][1].get('on_ground', 0)
                if prev_og == 1:
                    # 착륙 상태 → 이륙: 이전 leg 저장, 새 leg 시작
                    legs.append(current_leg)
                    current_leg = []

            current_leg.append((ts, ac))

        if current_leg:
            legs.append(current_leg)

        return legs

    def _extract_single_departure(self, icao, leg):
        """단일 leg에서 이륙 이벤트 추출. 이륙이 없으면 None."""
        if len(leg) < 3:
            return None

        # 이륙 지점 찾기
        dep_idx = None
        prev_on_ground = None

        for i, (ts, ac) in enumerate(leg):
            og = ac.get('on_ground', 0)
            alt = ac.get('baro_altitude_ft') or ac.get('alt_baro') or 0

            if prev_on_ground == 1 and og != 1:
                dep_idx = i
                break
            elif i == 0 and og != 1:
                # 첫 포인트가 이미 airborne
                # 고도 1500ft 이상이면 상공 통과일 가능성 높음 — 이륙 아님
                if alt > 1500:
                    return None
                lat = ac.get('lat', 0)
                lon = ac.get('lon', 0)
                ap, dist = self._nearest_airport(lat, lon, self.dep_radius)
                if ap and dist < 5.0:  # 5NM 이내만 이륙으로 인정 (8NM은 너무 넓음)
                    dep_idx = i
                    break
            prev_on_ground = og

        if dep_idx is None:
            return None

        dep_ts, dep_ac = leg[dep_idx]
        dep_lat = dep_ac.get('lat', 0)
        dep_lon = dep_ac.get('lon', 0)
        dep_ap, dep_dist = self._nearest_airport(dep_lat, dep_lon)

        if dep_ap is None:
            return None

        # 공항에서 너무 멀면 이륙이 아님
        if dep_dist > self.dep_radius:
            return None

        # callsign (leg 내에서)
        callsign = ''
        model = ''
        reg = ''
        category = ''
        for _, ac in leg:
            if not callsign:
                cs = ac.get('callsign', '').strip()
                if cs:
                    callsign = cs
            if not model:
                model = ac.get('aircraft_model') or ac.get('model') or ''
            if not reg:
                reg = ac.get('registration') or ''
            if not category:
                category = ac.get('category') or ''

        # 순항 정보 (이륙 이후 구간만)
        max_alt = 0
        max_spd = 0
        for i in range(dep_idx, len(leg)):
            ts, ac = leg[i]
            alt = ac.get('baro_altitude_ft') or ac.get('alt_baro') or 0
            spd = ac.get('ground_speed_kt') or ac.get('gs_kt') or 0
            if alt > max_alt:
                max_alt = alt
            if spd > max_spd:
                max_spd = spd

        # 관제에 무의미한 저고도/저속 비행 필터링
        # (경량기, 헬기 등 — category A1=Light, B=balloon/glider 등)
        if max_alt < self.min_cruise_alt and max_spd < 150:
            return None

        # 궤적이 너무 짧으면 (데이터 누락) 무시
        leg_duration = leg[-1][0] - leg[dep_idx][0]
        if leg_duration < 60:  # 1분 미만
            return None

        # 이륙 후 방향
        hdg_idx = min(dep_idx + 5, len(leg) - 1)
        hdg_ac = leg[hdg_idx][1]
        heading_after_dep = hdg_ac.get('true_track_deg') or hdg_ac.get('track_deg') or 0

        # 목적지 추론: leg 마지막 위치에서 가장 가까운 공항
        last_ts, last_ac = leg[-1]
        last_lat = last_ac.get('lat', 0)
        last_lon = last_ac.get('lon', 0)
        last_alt = last_ac.get('baro_altitude_ft') or last_ac.get('alt_baro') or 0
        last_og = last_ac.get('on_ground', 0)
        dest_ap = None

        # 착륙 확인된 경우만 목적지 표시 (on_ground=1 + 공항 근처)
        # ADS-B 커버리지 한계로 마지막 위치 기반 추론은 오탐이 많음
        if last_og == 1:
            arr_ap, arr_dist = self._nearest_airport(last_lat, last_lon, self.arr_radius)
            if arr_ap and arr_ap['icao'] != dep_ap['icao']:
                dest_ap = arr_ap

        return FlightPlan(
            icao24=icao,
            callsign=callsign,
            dep_time=dep_ts,
            dep_airport=dep_ap['icao'],
            dep_airport_name=dep_ap['name'],
            dest_airport=dest_ap['icao'] if dest_ap else None,
            dest_airport_name=dest_ap['name'] if dest_ap else None,
            heading_deg=heading_after_dep,
            cruise_alt_ft=max_alt,
            cruise_spd_kt=max_spd,
            aircraft_model=model,
            registration=reg,
        )

    def get_upcoming_departures(self, current_ts, lookahead_sec=600):
        """
        현재 시각 기준 앞으로 lookahead_sec 이내 이륙 예정 항공기 반환.
        (Flight Plan이 사전에 주어진 것처럼 동작)

        Returns: List[FlightPlan] — dep_time 순 정렬
        """
        upcoming = []
        grace = 30  # 방금 뜬 것도 잠시 표시 (배속 리플레이 대응)
        for plan in self._plans_by_time:
            if plan.dep_time < current_ts - grace:
                continue
            if plan.dep_time > current_ts + lookahead_sec:
                break
            upcoming.append(plan)
        return upcoming

    def get_active_flights(self, current_ts):
        """
        현재 시각 기준으로 이미 이륙한 (아직 비행 중일 수 있는) 항공기.
        이륙 후 4시간 이내만.
        """
        active = []
        for plan in self._plans_by_time:
            if plan.dep_time > current_ts:
                break
            if current_ts - plan.dep_time < 4 * 3600:
                active.append(plan)
        return active

    def get_departures_from_airport(self, airport_icao, current_ts, lookahead_sec=600):
        """특정 공항에서 곧 이륙할 항공기"""
        return [p for p in self.get_upcoming_departures(current_ts, lookahead_sec)
                if p.dep_airport == airport_icao]

    def get_airport_activity(self, current_ts, lookahead_sec=600):
        """
        공항별 이륙 예정 수 요약.
        Returns: dict[airport_icao] → (airport_name, count, [FlightPlan...])
        """
        upcoming = self.get_upcoming_departures(current_ts, lookahead_sec)
        activity = {}
        for plan in upcoming:
            key = plan.dep_airport
            if key not in activity:
                activity[key] = (plan.dep_airport_name, 0, [])
            name, cnt, plans = activity[key]
            activity[key] = (name, cnt + 1, plans + [plan])
        return activity

    def summary(self):
        """추출 결과 요약 출력"""
        if not self.plans:
            print("[FlightPlan] No plans extracted")
            return

        airports_dep = {}
        airports_arr = {}
        for p in self.plans:
            airports_dep[p.dep_airport] = airports_dep.get(p.dep_airport, 0) + 1
            if p.dest_airport:
                airports_arr[p.dest_airport] = airports_arr.get(p.dest_airport, 0) + 1

        print(f"\n[FlightPlan] === Summary ===")
        print(f"  Total departures: {len(self.plans)}")
        print(f"  Departure airports:")
        for ap, cnt in sorted(airports_dep.items(), key=lambda x: -x[1])[:10]:
            print(f"    {ap}: {cnt} departures")
        print(f"  Destination airports (inferred):")
        for ap, cnt in sorted(airports_arr.items(), key=lambda x: -x[1])[:10]:
            print(f"    {ap}: {cnt} arrivals")
        no_dest = sum(1 for p in self.plans if not p.dest_airport)
        print(f"  Unknown destination: {no_dest}")

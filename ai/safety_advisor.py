"""
Safety Advisor AI - 관제사 안전 조언 시스템

관제사에게 사전에 위험을 경고하고 안전 조언을 제공한다.
실시간으로 모든 항공기의 궤적을 분석하여:

1. 충돌 예측 (Conflict Detection & Resolution)
   - 수평/수직 분리 위반 예측
   - TCAS RA/TA 발생 전 사전 경고

2. 궤적 위험 분석
   - 제한구역 침범 예측
   - 위험한 선회/강하율 감지
   - 후류난기류 위험 감지

3. 트래픽 밀도 경고
   - 과밀 구역 사전 경고
   - 병목 구간 예측

4. 기동 권고
   - 안전한 고도/방위 변경 제안
   - 최적 분리 벡터 계산
"""
import math
import time
import threading
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
from pathlib import Path

import torch

from utils.geo import calculate_distance, calculate_bearing
from config import (
    STATIC_OBSTACLES, KNOTS_TO_MPS, M_TO_NM,
    SAFETY_LOOKAHEAD_SEC, SAFETY_LOOKAHEAD_STEPS, SAFETY_SCAN_INTERVAL_SEC,
    CONFLICT_HORIZ_NM, CONFLICT_VERT_FT, WAKE_TURBULENCE_NM,
    KADIZ_POLYGON, ATS_ROUTES, ATS_ROUTE_BUFFER_NM,
)

log = logging.getLogger(__name__)


class Severity(IntEnum):
    """경고 심각도"""
    INFO = 0       # 참고 정보
    CAUTION = 1    # 주의 (노란색)
    WARNING = 2    # 경고 (주황색)
    ALERT = 3      # 긴급 (빨간색)


@dataclass
class SafetyAlert:
    """안전 경고 데이터"""
    severity: Severity
    category: str           # "CONFLICT" | "AIRSPACE" | "MANEUVER" | "TRAFFIC" | "WAKE" | "SQUAWK" | "ATS_ROUTE"
    title: str              # 짧은 제목
    message: str            # 상세 설명
    aircraft_involved: list # 관련 항공기 콜사인 리스트
    time_to_event_sec: float = 0    # 이벤트까지 남은 시간 (0=즉시)
    recommendation: str = ""        # 관제사에게 권고 사항
    position: tuple = (0, 0)        # (lat, lon) 이벤트 예상 위치
    timestamp: float = field(default_factory=time.time)
    # ACK 시스템
    event_key: str = ""             # 이벤트 고유 키 (같은 이벤트 식별용)
    ackable: bool = True            # ACK 가능 여부 (False면 행동으로만 해소)

    @property
    def urgency_score(self):
        """정렬용 긴급도 점수 (높을수록 긴급)"""
        time_factor = max(0, 300 - self.time_to_event_sec) / 300 if self.time_to_event_sec > 0 else 1.0
        return self.severity * 100 + time_factor * 50


@dataclass
class AckRecord:
    """ACK 기록"""
    event_key: str
    acked_at: float          # ACK 시각 (epoch)
    severity_at_ack: Severity  # ACK 당시 심각도
    metric_at_ack: float = 0.0   # ACK 당시 핵심 수치 (거리 등)
    expires_at: float = 0.0  # 자동 만료 시각 (0=상태 기반만)


class SafetyAdvisor:
    """
    실시간 안전 조언 AI

    사용법:
        advisor = SafetyAdvisor()
        advisor.update(user_aircraft_list, other_aircraft_dict)
        alerts = advisor.get_alerts()  # 현재 활성 경고 목록
    """

    # ACK 불가능 카테고리 — 행동으로만 해소
    NON_ACKABLE = {"AIRSPACE_EXIT", "KADIZ_EXIT", "SQUAWK_EMRG", "MSA", "ATS_ALONG"}

    def __init__(self, world_model_path=None):
        self.alerts: list[SafetyAlert] = []
        self._lock = threading.Lock()
        self._last_scan = 0.0

        # ACK 상태머신
        self._ack_records: dict[str, AckRecord] = {}  # event_key → AckRecord
        self._ack_log: list[AckRecord] = []  # 사후 감사용 전체 로그
        self._ack_timeout_sec = 300.0  # ACK 후 5분 경과하면 자동 만료

        # World Model 기반 conflict detector (선택적)
        self._conflict_detector = None
        self._wm_last_scan = 0.0
        self._wm_scan_interval = 5.0  # World Model은 5초 간격
        if world_model_path:
            self._init_world_model(world_model_path)

    def _init_world_model(self, model_path):
        """학습된 World Model 로드"""
        try:
            from ai.world_model.trajectory_predictor import TrajectoryPredictor
            from ai.world_model.conflict_detector import ConflictDetector

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            args = checkpoint.get('args', {})
            model = TrajectoryPredictor(
                hidden_dim=args.get('hidden_dim', 256),
                latent_dim=args.get('latent_dim', 64),
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            self._conflict_detector = ConflictDetector(model, device=str(device))
            log.info(f"[SafetyAdvisor] World Model loaded from {model_path} "
                     f"(epoch {checkpoint.get('epoch', '?')})")
        except Exception as e:
            log.warning(f"[SafetyAdvisor] World Model load failed: {e}")
            self._conflict_detector = None

    def update(self, user_aircraft, other_aircraft_dict):
        """
        모든 분석을 실행하고 경고 목록을 갱신한다.
        매 시뮬레이션 프레임마다 호출 가능하나, 내부적으로 주기 제한.

        :param user_aircraft: 사용자 제어 항공기 리스트
        :param other_aircraft_dict: {icao24: Aircraft} 외부 항공기
        """
        now = time.time()
        if now - self._last_scan < SAFETY_SCAN_INTERVAL_SEC:
            return
        self._last_scan = now

        new_alerts = []
        all_aircraft = list(user_aircraft) + list(other_aircraft_dict.values())

        # World Model conflict detection (별도 주기)
        if self._conflict_detector and now - self._wm_last_scan >= self._wm_scan_interval:
            self._wm_last_scan = now
            wm_alerts = self._run_world_model_detection(
                user_aircraft, other_aircraft_dict)
            new_alerts.extend(wm_alerts)

        if user_aircraft:
            # 관제 모드: user_aircraft 중심으로 분석
            for uac in user_aircraft:
                others = [ac for ac in all_aircraft if ac is not uac]
                new_alerts.extend(self._check_conflicts(uac, others))
                new_alerts.extend(self._check_airspace_violations(uac))
                new_alerts.extend(self._check_dangerous_maneuvers(uac))
                new_alerts.extend(self._check_wake_turbulence(uac, others))
                new_alerts.extend(self._check_fuel(uac))
                new_alerts.extend(self._check_airspace_exit(uac))
                new_alerts.extend(self._check_kadiz_exit(uac))
                new_alerts.extend(self._check_ats_route(uac))
        else:
            # 감시 모드: 모든 항공기 쌍에 대해 conflict 분석
            ac_list = list(other_aircraft_dict.values())
            new_alerts.extend(self._check_all_pairs(ac_list))

        # Squawk 비상 항공기 감지 (모든 모드)
        new_alerts.extend(self._check_squawk_emergency(all_aircraft, user_aircraft))

        # 트래픽 밀도 경고
        new_alerts.extend(self._check_traffic_density(user_aircraft, all_aircraft))

        # ACK 필터링 적용 후 정렬
        self._expire_acks(now)
        with self._lock:
            filtered = []
            for alert in new_alerts:
                if not alert.event_key:
                    alert.event_key = self._make_event_key(alert)
                alert.ackable = alert.category not in self.NON_ACKABLE
                ack = self._ack_records.get(alert.event_key)
                if ack:
                    # ACK 무효화 조건: 상황 악화 (심각도 상승)
                    if alert.severity > ack.severity_at_ack:
                        del self._ack_records[alert.event_key]
                        alert._ack_state = 'escalated'
                    else:
                        alert._ack_state = 'acked'
                else:
                    alert._ack_state = 'new'
                filtered.append(alert)
            self.alerts = sorted(filtered, key=lambda a: -a.urgency_score)

    def get_alerts(self, min_severity=Severity.INFO, include_acked=False) -> list[SafetyAlert]:
        """
        현재 활성 경고를 심각도 순으로 반환.
        include_acked=False면 ACK된 경고는 제외 (UI에서 조용히 표시할 때는 True)
        """
        with self._lock:
            result = []
            for a in self.alerts:
                if a.severity < min_severity:
                    continue
                state = getattr(a, '_ack_state', 'new')
                if not include_acked and state == 'acked':
                    continue
                result.append(a)
            return result

    def get_acked_alerts(self) -> list[SafetyAlert]:
        """ACK된 진행 중 경고 (UI 배경 영역에 조용히 표시)"""
        with self._lock:
            return [a for a in self.alerts if getattr(a, '_ack_state', 'new') == 'acked']

    def get_alerts_for_aircraft(self, callsign) -> list[SafetyAlert]:
        """특정 항공기에 관련된 경고만 반환"""
        with self._lock:
            return [a for a in self.alerts if callsign in a.aircraft_involved]

    def acknowledge(self, event_key, metric=0.0):
        """
        관제사가 경고를 ACK. 동일 이벤트 반복 경고를 억제.
        Non-ackable 경고는 무시됨.

        :param event_key: 경고의 event_key
        :param metric: ACK 시점의 핵심 수치 (거리 등, 악화 감지용)
        """
        with self._lock:
            # non-ackable 확인
            for a in self.alerts:
                if a.event_key == event_key and not a.ackable:
                    return False

            now = time.time()
            rec = AckRecord(
                event_key=event_key,
                acked_at=now,
                severity_at_ack=Severity.INFO,
                metric_at_ack=metric,
                expires_at=now + self._ack_timeout_sec,
            )
            # ACK 시점 심각도 기록
            for a in self.alerts:
                if a.event_key == event_key:
                    rec.severity_at_ack = a.severity
                    break
            self._ack_records[event_key] = rec
            self._ack_log.append(rec)
            log.info(f"[ACK] {event_key} acknowledged at {now:.0f}")
            return True

    def _expire_acks(self, now):
        """만료된 ACK 제거 (시간 기반)"""
        expired = [k for k, v in self._ack_records.items()
                   if v.expires_at > 0 and now > v.expires_at]
        for k in expired:
            del self._ack_records[k]

    @staticmethod
    def _make_event_key(alert):
        """경고에서 고유 이벤트 키 생성. 같은 페어/같은 항공기의 같은 카테고리 = 같은 이벤트."""
        involved = sorted(alert.aircraft_involved)
        return f"{alert.category}:{'|'.join(involved)}"

    # ── 1. 충돌 예측 ──

    def _check_conflicts(self, uac, others):
        """관제 모드: 우리 전투기(uac) vs 다른 항공기 conflict 체크.
        민항기끼리는 체크 안 함 — uac는 항상 우리 전투기."""
        alerts = []
        dt = SAFETY_LOOKAHEAD_SEC / SAFETY_LOOKAHEAD_STEPS

        for oac in others:
            # 민항기 접근/이륙 패턴 필터 (저고도+저속 = 공항 근처 예측 가능)
            o_alt = getattr(oac, 'alt_current', 0) or 0
            o_spd = getattr(oac, 'ground_speed_kt', 0) or 0
            o_vrate = getattr(oac, 'vertical_rate_ft_min', 0) or 0
            if o_alt < 3000 and o_spd < 200:
                continue
            if o_alt < 5000 and o_vrate < -300:
                continue

            # 현재 거리가 너무 멀면 스킵 (50NM 이상)
            cur_dist = calculate_distance(uac.lat, uac.lon, oac.lat, oac.lon)
            if cur_dist > 50:
                continue

            min_h_dist = cur_dist
            min_v_dist = abs(uac.alt_current - oac.alt_current)
            conflict_time = 0
            conflict_pos = (0, 0)

            # 미래 궤적 외삽
            for step in range(1, SAFETY_LOOKAHEAD_STEPS + 1):
                t = dt * step
                u_pos = self._extrapolate(uac, t)
                o_pos = self._extrapolate_external(oac, t)

                h_dist = calculate_distance(u_pos[0], u_pos[1], o_pos[0], o_pos[1])
                v_dist = abs(u_pos[2] - o_pos[2])

                if h_dist < min_h_dist:
                    min_h_dist = h_dist
                    min_v_dist = v_dist
                    conflict_time = t
                    conflict_pos = ((u_pos[0] + o_pos[0]) / 2, (u_pos[1] + o_pos[1]) / 2)

            other_cs = getattr(oac, 'callsign', '') or getattr(oac, 'icao24', '?')

            # RA 수준 충돌 예측 (5NM/1000ft 이내)
            if min_h_dist < CONFLICT_HORIZ_NM and min_v_dist < CONFLICT_VERT_FT:
                sev = Severity.ALERT if conflict_time < 120 else Severity.WARNING
                brg = calculate_bearing(uac.lat, uac.lon, oac.lat, oac.lon)

                # 회피 방향 계산
                avoid_hdg, avoid_alt = self._suggest_avoidance(uac, oac, brg)

                alerts.append(SafetyAlert(
                    severity=sev,
                    category="CONFLICT",
                    title=f"충돌 예측: {other_cs}",
                    message=(f"{other_cs}과 {conflict_time:.0f}초 후 "
                             f"수평 {min_h_dist:.1f}NM / 수직 {min_v_dist:.0f}ft 접근 예상. "
                             f"현재 방위 {brg:.0f}\u00b0, 거리 {cur_dist:.1f}NM"),
                    aircraft_involved=[uac.callsign, other_cs],
                    time_to_event_sec=conflict_time,
                    recommendation=f"권고: HDG {avoid_hdg:.0f}\u00b0 또는 FL{avoid_alt / 100:.0f}으로 변경",
                    position=conflict_pos,
                ))

            # TA 수준 접근 경고 (10NM/2000ft 이내)
            elif min_h_dist < CONFLICT_HORIZ_NM * 2 and min_v_dist < CONFLICT_VERT_FT * 2:
                if conflict_time < 180:
                    alerts.append(SafetyAlert(
                        severity=Severity.CAUTION,
                        category="CONFLICT",
                        title=f"접근 주의: {other_cs}",
                        message=(f"{other_cs}과 {conflict_time:.0f}초 후 "
                                 f"{min_h_dist:.1f}NM/{min_v_dist:.0f}ft 접근 예상"),
                        aircraft_involved=[uac.callsign, other_cs],
                        time_to_event_sec=conflict_time,
                        recommendation="트래픽을 모니터링하세요",
                        position=conflict_pos,
                    ))

        return alerts

    def _suggest_avoidance(self, uac, oac, bearing_to_other):
        """안전한 회피 방향/고도를 계산"""
        # 상대 항공기의 반대쪽으로 30도 회피
        if ((bearing_to_other - uac.hdg_current + 180) % 360 - 180) > 0:
            avoid_hdg = (uac.hdg_current - 30) % 360  # 왼쪽 회피
        else:
            avoid_hdg = (uac.hdg_current + 30) % 360  # 오른쪽 회피

        # 고도: 상대보다 높으면 올라가고, 낮으면 내려감
        if uac.alt_current >= oac.alt_current:
            avoid_alt = uac.alt_current + 2000
        else:
            avoid_alt = uac.alt_current - 2000
        avoid_alt = max(5000, min(45000, avoid_alt))

        return avoid_hdg, avoid_alt

    # ── 2. 공역 제한구역 침범 예측 ──

    def _check_airspace_violations(self, uac):
        alerts = []
        dt = SAFETY_LOOKAHEAD_SEC / SAFETY_LOOKAHEAD_STEPS

        for zone in STATIC_OBSTACLES:
            # 다각형 공역: 중심점과 반경을 계산
            verts = zone.get("vertices")
            if verts and len(verts) >= 3:
                lats = [v[0] for v in verts]
                lons = [v[1] for v in verts]
                z_lat = sum(lats) / len(lats)
                z_lon = sum(lons) / len(lons)
                z_radius = max((max(lats) - min(lats)) * 60 / 2,
                               (max(lons) - min(lons)) * 60 * 0.8 / 2, 2.0)
            elif "lat" in zone:
                z_lat = zone["lat"]
                z_lon = zone["lon"]
                z_radius = zone.get("radius_nm", 20)
            else:
                continue

            # 현재 거리 확인
            cur_dist = calculate_distance(uac.lat, uac.lon, z_lat, z_lon)
            if cur_dist > z_radius + 30:
                continue

            # ── 현재 위치가 이미 공역 내부인지 체크 ──
            if verts and len(verts) >= 3:
                currently_inside = self._point_in_polygon(uac.lat, uac.lon, verts)
            else:
                currently_inside = cur_dist < z_radius
            currently_in_alt = zone["min_alt"] <= uac.alt_current <= zone["max_alt"]

            if currently_inside and currently_in_alt:
                away_hdg = (calculate_bearing(z_lat, z_lon, uac.lat, uac.lon)) % 360
                if uac.alt_current < (zone["min_alt"] + zone["max_alt"]) / 2:
                    safe_alt = zone["min_alt"] - 500
                else:
                    safe_alt = zone["max_alt"] + 500
                safe_alt = max(1000, safe_alt)

                alerts.append(SafetyAlert(
                    severity=Severity.ALERT,
                    category="AIRSPACE",
                    title=f"{zone['name']} 침범 중",
                    message=(f"현재 {zone['name']} 공역 내부 비행 중! "
                             f"(FL{zone['min_alt'] / 100:.0f}-{zone['max_alt'] / 100:.0f})"),
                    aircraft_involved=[uac.callsign],
                    time_to_event_sec=0,
                    recommendation=(f"즉시 이탈: HDG {away_hdg:.0f}\u00b0 선회 "
                                    f"또는 FL{safe_alt / 100:.0f}으로 고도 변경"),
                    position=(uac.lat, uac.lon),
                ))
                continue  # 이미 침범 중이면 예측 불필요

            # ── 미래 침범 예측 ──
            for step in range(1, SAFETY_LOOKAHEAD_STEPS + 1):
                t = dt * step
                pos = self._extrapolate(uac, t)

                if verts and len(verts) >= 3:
                    inside = self._point_in_polygon(pos[0], pos[1], verts)
                else:
                    dist = calculate_distance(pos[0], pos[1], z_lat, z_lon)
                    inside = dist < z_radius

                in_alt = zone["min_alt"] <= pos[2] <= zone["max_alt"]

                if inside and in_alt:
                    sev = Severity.ALERT if t < 60 else (Severity.WARNING if t < 180 else Severity.CAUTION)
                    away_hdg = (calculate_bearing(z_lat, z_lon, uac.lat, uac.lon)) % 360
                    if pos[2] < (zone["min_alt"] + zone["max_alt"]) / 2:
                        safe_alt = zone["min_alt"] - 500
                    else:
                        safe_alt = zone["max_alt"] + 500
                    safe_alt = max(1000, safe_alt)

                    alerts.append(SafetyAlert(
                        severity=sev,
                        category="AIRSPACE",
                        title=f"{zone['name']} 침범 예측",
                        message=(f"{t:.0f}초 후 {zone['name']} 진입 예상 "
                                 f"(FL{zone['min_alt'] / 100:.0f}-{zone['max_alt'] / 100:.0f})"),
                        aircraft_involved=[uac.callsign],
                        time_to_event_sec=t,
                        recommendation=(f"권고: HDG {away_hdg:.0f}\u00b0 선회 "
                                        f"또는 FL{safe_alt / 100:.0f}으로 고도 변경"),
                        position=(z_lat, z_lon),
                    ))
                    break

        return alerts

    # ── 3. 위험 기동 감지 ──

    def _check_dangerous_maneuvers(self, uac):
        alerts = []

        # 3a. 과도한 강하율
        if uac.is_user_controlled and uac.instruction_active:
            if uac.alt_current > uac.alt_target:
                alt_diff = uac.alt_current - uac.alt_target
                # 현재 고도가 낮은데 급강하하는 경우
                if uac.alt_current < 10000 and alt_diff > 3000:
                    alerts.append(SafetyAlert(
                        severity=Severity.WARNING,
                        category="MANEUVER",
                        title="저고도 급강하 경고",
                        message=(f"현재 고도 {uac.alt_current:.0f}ft에서 "
                                 f"{uac.alt_target:.0f}ft로 {alt_diff:.0f}ft 강하 중. "
                                 f"저고도에서의 급강하는 위험합니다."),
                        aircraft_involved=[uac.callsign],
                        recommendation="강하율을 줄이거나 목표 고도를 재검토하세요",
                    ))

            # 3b. 최저 안전 고도 위반
            if uac.alt_target < 2000 and uac.alt_current > 2000:
                alerts.append(SafetyAlert(
                    severity=Severity.ALERT,
                    category="MANEUVER",
                    title="최저안전고도(MSA) 위반",
                    message=f"목표 고도 {uac.alt_target:.0f}ft가 MSA(2000ft) 미만입니다",
                    aircraft_involved=[uac.callsign],
                    recommendation="목표 고도를 2000ft 이상으로 수정하세요",
                ))

        # 3c. 과속 (Mach 0.9 이상 ≈ 585kts)
        if uac.spd_current > 585:
            alerts.append(SafetyAlert(
                severity=Severity.CAUTION,
                category="MANEUVER",
                title="과속 경고",
                message=f"현재 속도 {uac.spd_current:.0f}kts (Mach 0.9 초과)",
                aircraft_involved=[uac.callsign],
                recommendation="속도를 Mach 0.85 (약 520kts) 이하로 감속하세요",
            ))

        # 3d. 반대 방향 비행 (목적지에서 멀어지는 경우)
        # 이건 RL 환경에서만 의미가 있으므로 여기선 건너뜀

        return alerts

    # ── 4. 후류난기류 ──

    def _check_wake_turbulence(self, uac, others):
        alerts = []
        for oac in others:
            if oac.is_user_controlled:
                continue
            dist = calculate_distance(uac.lat, uac.lon, oac.lat, oac.lon)
            if dist > WAKE_TURBULENCE_NM:
                continue
            alt_diff = abs(uac.alt_current - oac.alt_current)
            if alt_diff > 1000:
                continue

            # 같은 방향으로 비행하면서 뒤에 있는 경우
            brg_to_other = calculate_bearing(uac.lat, uac.lon, oac.lat, oac.lon)
            uac_hdg = uac.hdg_current
            hdg_diff = abs(brg_to_other - (uac_hdg + 180) % 360)
            hdg_diff = min(hdg_diff, 360 - hdg_diff)

            if hdg_diff < 30:  # 선행 항공기 뒤를 따라가는 경우
                other_cs = getattr(oac, 'callsign', '') or getattr(oac, 'icao24', '?')
                alerts.append(SafetyAlert(
                    severity=Severity.CAUTION,
                    category="WAKE",
                    title=f"후류난기류 주의: {other_cs}",
                    message=(f"{other_cs} 후방 {dist:.1f}NM, 고도차 {alt_diff:.0f}ft. "
                             f"후류난기류 영향권 내 비행 중"),
                    aircraft_involved=[uac.callsign, other_cs],
                    recommendation=("수직 분리 1000ft 이상 확보하거나 "
                                    "수평 거리 6NM 이상 유지하세요"),
                ))

        return alerts

    # ── 5. 연료 상태 ──

    def _check_fuel(self, uac):
        alerts = []
        if not hasattr(uac, 'fuel_lbs') or not uac.is_user_controlled:
            return alerts

        endurance = uac.fuel_endurance_min
        pct = uac.fuel_pct
        flow = getattr(uac, 'fuel_flow_lbh', 0)

        if getattr(uac, 'fuel_bingo', False):
            alerts.append(SafetyAlert(
                severity=Severity.ALERT,
                category="FUEL",
                title="BINGO FUEL",
                message=(f"잔여 연료 {uac.fuel_lbs:.0f}lbs ({pct:.0f}%), "
                         f"체공 가능 {endurance:.0f}분. 즉시 귀환 필요"),
                aircraft_involved=[uac.callsign],
                recommendation="최단 거리 기지로 즉시 귀환하세요",
            ))
        elif endurance < 40:
            alerts.append(SafetyAlert(
                severity=Severity.WARNING,
                category="FUEL",
                title="연료 부족 경고",
                message=(f"잔여 연료 {uac.fuel_lbs:.0f}lbs ({pct:.0f}%), "
                         f"유량 {flow:.0f}lbs/hr, 체공 {endurance:.0f}분"),
                aircraft_involved=[uac.callsign],
                recommendation="귀환 경로를 확인하고 불필요한 기동을 줄이세요",
            ))
        elif pct < 50:
            alerts.append(SafetyAlert(
                severity=Severity.CAUTION,
                category="FUEL",
                title="연료 50% 미만",
                message=(f"잔여 연료 {uac.fuel_lbs:.0f}lbs ({pct:.0f}%), "
                         f"유량 {flow:.0f}lbs/hr, 체공 {endurance:.0f}분"),
                aircraft_involved=[uac.callsign],
                recommendation="연료 소모 추이를 모니터링하세요",
            ))

        return alerts

    # ── 6. 트래픽 밀도 ──

    def _check_traffic_density(self, user_aircraft, all_aircraft):
        alerts = []
        for uac in user_aircraft:
            count = 0
            close_count = 0
            for ac in all_aircraft:
                if ac is uac:
                    continue
                d = calculate_distance(uac.lat, uac.lon, ac.lat, ac.lon)
                if d < 25:
                    count += 1
                if d < 10:
                    close_count += 1

            if close_count >= 5:
                alerts.append(SafetyAlert(
                    severity=Severity.WARNING,
                    category="TRAFFIC",
                    title="고밀도 트래픽 경고",
                    message=f"반경 10NM 내 {close_count}대 항공기 밀집",
                    aircraft_involved=[uac.callsign],
                    recommendation="트래픽 분산을 위해 고도 또는 경로 변경을 고려하세요",
                    position=(uac.lat, uac.lon),
                ))
            elif count >= 8:
                alerts.append(SafetyAlert(
                    severity=Severity.CAUTION,
                    category="TRAFFIC",
                    title="트래픽 증가 주의",
                    message=f"반경 25NM 내 {count}대 항공기 확인",
                    aircraft_involved=[uac.callsign],
                    recommendation="주변 트래픽 상황을 주시하세요",
                    position=(uac.lat, uac.lon),
                ))

        return alerts

    # ── 감시 모드: 모든 항공기 쌍 conflict 분석 ──

    def _check_all_pairs(self, ac_list):
        """user_aircraft 없을 때 — 군 관련 쌍만 conflict 예측

        규칙:
        - 민-민: 스킵 (민간 관제 소관)
        - 군-민: 항상 경고
        - 군-군, 같은 공역 훈련 중: 스킵 (정상 교전)
        - 군-군, 한쪽이 비할당 공역 통과: 경고
        - 군-군, 둘 다 통과 중 (공역 미할당): 경고
        - 군-군, 인접 공역 경계 접근: 경고
        """
        alerts = []
        dt = SAFETY_LOOKAHEAD_SEC / SAFETY_LOOKAHEAD_STEPS

        # 군/민 분리, 군은 소속 공역 태깅
        mil = []
        for ac in ac_list:
            icao = getattr(ac, 'icao24', '')
            if icao.startswith('PF_'):
                moa_name = getattr(ac, 'moa', None) or \
                           getattr(ac, '_moa_name', None) or ''
                mil.append((ac, moa_name))

        # 민항기 (non-PF)
        civ = [ac for ac in ac_list if not getattr(ac, 'icao24', '').startswith('PF_')]

        pairs = []

        # 군-민: 체크 (단, 공항 접근 중인 저고도 민항기는 제외)
        for m, m_moa in mil:
            for c in civ:
                # 민항기가 저고도 + 저속 = 접근/이륙 패턴 → 예측 가능, 경고 불필요
                c_alt = getattr(c, 'alt_current', 0) or 0
                c_spd = getattr(c, 'ground_speed_kt', 0) or 0
                c_vrate = getattr(c, 'vertical_rate_ft_min', 0) or 0
                if c_alt < 3000 and c_spd < 200:
                    continue
                # 민항기가 강하 중 + 공항 5NM 이내 = final approach
                if c_alt < 5000 and c_vrate < -300:
                    continue
                d = calculate_distance(m.lat, m.lon, c.lat, c.lon)
                if d < 25:
                    pairs.append((m, c, True))   # is_mil_civ=True

        # 군-군: 같은 공역이면 스킵
        for i in range(len(mil)):
            for j in range(i + 1, len(mil)):
                m_a, moa_a = mil[i]
                m_b, moa_b = mil[j]
                # 둘 다 같은 공역에 할당 = 같은 훈련 → 스킵
                if moa_a and moa_b and moa_a == moa_b:
                    continue
                d = calculate_distance(m_a.lat, m_a.lon, m_b.lat, m_b.lon)
                if d < 15:
                    pairs.append((m_a, m_b, False))

        for a, b, is_mil_civ in pairs:
            cur_dist = calculate_distance(a.lat, a.lon, b.lat, b.lon)
            min_h_dist = cur_dist
            min_v_dist = abs(a.alt_current - b.alt_current)
            conflict_time = 0
            conflict_pos = (0, 0)

            for step in range(1, SAFETY_LOOKAHEAD_STEPS + 1):
                t = dt * step
                a_pos = self._extrapolate_external(a, t)
                b_pos = self._extrapolate_external(b, t)
                h_dist = calculate_distance(a_pos[0], a_pos[1], b_pos[0], b_pos[1])
                v_dist = abs(a_pos[2] - b_pos[2])
                if h_dist < min_h_dist:
                    min_h_dist = h_dist
                    min_v_dist = v_dist
                    conflict_time = t
                    conflict_pos = ((a_pos[0]+b_pos[0])/2, (a_pos[1]+b_pos[1])/2)

            cs_a = getattr(a, 'callsign', '') or getattr(a, 'icao24', '?')
            cs_b = getattr(b, 'callsign', '') or getattr(b, 'icao24', '?')
            h_threshold = CONFLICT_HORIZ_NM * (1.5 if is_mil_civ else 1.0)
            v_threshold = CONFLICT_VERT_FT * (1.5 if is_mil_civ else 1.0)
            tag = " [MIL-CIV]" if is_mil_civ else ""

            if min_h_dist < h_threshold and min_v_dist < v_threshold:
                sev = Severity.ALERT if conflict_time < 120 or is_mil_civ else Severity.WARNING
                alerts.append(SafetyAlert(
                    severity=sev,
                    category="CONFLICT",
                    title=f"충돌 예측{tag}: {cs_a}↔{cs_b}",
                    message=(f"{conflict_time:.0f}초 후 "
                             f"{min_h_dist:.1f}NM/{min_v_dist:.0f}ft 접근"),
                    aircraft_involved=[cs_a, cs_b],
                    time_to_event_sec=conflict_time,
                    position=conflict_pos,
                ))
            elif min_h_dist < h_threshold * 2 and min_v_dist < v_threshold * 2:
                if conflict_time < 180:
                    alerts.append(SafetyAlert(
                        severity=Severity.WARNING if is_mil_civ else Severity.CAUTION,
                        category="CONFLICT",
                        title=f"접근 주의{tag}: {cs_a}↔{cs_b}",
                        message=(f"{conflict_time:.0f}초 후 "
                                 f"{min_h_dist:.1f}NM/{min_v_dist:.0f}ft"),
                        aircraft_involved=[cs_a, cs_b],
                        time_to_event_sec=conflict_time,
                        position=conflict_pos,
                    ))

            if len(alerts) > 20:
                break

        return alerts

    # ── World Model 기반 conflict detection ──

    def _run_world_model_detection(self, user_aircraft, other_aircraft_dict):
        """World Model로 확률적 conflict 예측"""
        alerts = []
        if not self._conflict_detector:
            return alerts

        try:
            # 모든 항공기 상태를 detector에 업데이트
            for uac in user_aircraft:
                self._conflict_detector.update_state(uac.callsign, {
                    'lat': uac.lat, 'lon': uac.lon,
                    'baro_altitude_ft': uac.alt_current,
                    'ground_speed_kt': uac.spd_current,
                    'true_track_deg': uac.hdg_current,
                    'vertical_rate_ft_min': getattr(uac, 'vertical_rate_ft_min', 0),
                    'ias_kt': uac.spd_current,
                    'mach': uac.spd_current / 660.0,
                })

            for icao, ac in other_aircraft_dict.items():
                self._conflict_detector.update_state(icao, {
                    'lat': ac.lat, 'lon': ac.lon,
                    'baro_altitude_ft': ac.alt_current,
                    'ground_speed_kt': ac.ground_speed_kt,
                    'true_track_deg': ac.track_true_deg,
                    'vertical_rate_ft_min': ac.vertical_rate_ft_min,
                    'ias_kt': getattr(ac, 'ias_kt', 0),
                    'mach': getattr(ac, 'mach', 0),
                })

            # Conflict 예측 실행
            predictions = self._conflict_detector.detect(
                dt_sec=10.0, past_steps=6, min_prob=0.1)

            for pred in predictions:
                # 관제사가 관리하는 항공기가 관련된 경우만
                user_callsigns = {uac.callsign for uac in user_aircraft}
                if pred.icao_a not in user_callsigns and pred.icao_b not in user_callsigns:
                    continue

                if pred.probability >= 0.7:
                    severity = Severity.ALERT
                elif pred.probability >= 0.4:
                    severity = Severity.WARNING
                else:
                    severity = Severity.CAUTION

                alerts.append(SafetyAlert(
                    severity=severity,
                    category="CONFLICT",
                    title=f"[WM] 충돌예측: {pred.icao_a}↔{pred.icao_b}",
                    message=(
                        f"AI 예측: {pred.expected_time_sec:.0f}초 후 conflict 확률 "
                        f"{pred.probability*100:.0f}%\n"
                        f"예상 최소분리: {pred.min_h_dist_nm:.1f}NM / "
                        f"{pred.min_v_dist_ft:.0f}ft\n"
                        f"최악(5%ile): {pred.worst_h_dist_nm:.1f}NM / "
                        f"{pred.worst_v_dist_ft:.0f}ft\n"
                        f"예측 신뢰도: {pred.confidence*100:.0f}%"
                    ),
                    aircraft_involved=[pred.icao_a, pred.icao_b],
                    time_to_event_sec=pred.expected_time_sec,
                    recommendation=(
                        f"World Model 기반 예측. "
                        f"{'즉각 분리 조치 필요' if pred.probability > 0.6 else '상황 모니터링 권장'}"
                    ),
                    position=pred.position,
                ))

        except Exception as e:
            log.warning(f"[SafetyAdvisor] World Model error: {e}")

        return alerts

    # ── 궤적 외삽 헬퍼 ──

    def _extrapolate(self, ac, t_sec):
        """사용자 항공기의 t초 후 예상 위치 (hdg/spd/alt 목표 반영)"""
        # 목표로 수렴하면서 이동
        hdg = ac.hdg_current
        spd = ac.spd_current
        alt = ac.alt_current

        if ac.instruction_active or ac.instruction_pending:
            # 간단히: t초 후에는 목표값에 가까워진다고 가정
            blend = min(1.0, t_sec / 60.0)  # 60초면 완전 수렴
            hdg_diff = (ac.hdg_target - ac.hdg_current + 180) % 360 - 180
            hdg = (ac.hdg_current + hdg_diff * blend) % 360
            spd = ac.spd_current + (ac.spd_target - ac.spd_current) * blend
            alt = ac.alt_current + (ac.alt_target - ac.alt_current) * blend

        spd_nm_s = spd * KNOTS_TO_MPS * M_TO_NM
        dist = spd_nm_s * t_sec
        rad = math.radians(hdg)
        lat = ac.lat + dist * math.cos(rad) / 60.0
        cos_lat = math.cos(math.radians(ac.lat))
        lon = ac.lon
        if abs(cos_lat) >= 1e-6:
            lon += dist * math.sin(rad) / (60.0 * cos_lat)
        return (lat, lon, alt)

    def _extrapolate_external(self, ac, t_sec):
        """외부 항공기의 t초 후 예상 위치 (선형 외삽)"""
        spd_nm_s = ac.ground_speed_kt * KNOTS_TO_MPS * M_TO_NM
        dist = spd_nm_s * t_sec
        rad = math.radians(ac.track_true_deg)
        lat = ac.lat + dist * math.cos(rad) / 60.0
        cos_lat = math.cos(math.radians(ac.lat))
        lon = ac.lon
        if abs(cos_lat) >= 1e-6:
            lon += dist * math.sin(rad) / (60.0 * cos_lat)
        alt = ac.alt_current + (ac.vertical_rate_ft_min / 60.0) * t_sec
        return (lat, lon, alt)

    # ── 공역 이탈 경고 (지정 공역 벗어남) ──

    def _check_airspace_exit(self, uac):
        """전투기가 지정 공역을 벗어나려 할 때 경고 (ACK 불가)"""
        alerts = []
        assigned_moa = getattr(uac, 'moa', None) or getattr(uac, 'assigned_airspace', None)
        if not assigned_moa:
            return alerts

        # 지정 공역의 다각형 찾기
        from config import MOA_LIST, R_ZONE_LIST
        zone = None
        for z in MOA_LIST + R_ZONE_LIST:
            if z['name'] == assigned_moa:
                zone = z
                break
        if not zone or not zone.get('vertices'):
            return alerts

        verts = zone['vertices']
        inside = self._point_in_polygon(uac.lat, uac.lon, verts)

        if not inside:
            alerts.append(SafetyAlert(
                severity=Severity.ALERT,
                category="AIRSPACE_EXIT",
                title=f"공역 이탈: {assigned_moa}",
                message=f"지정 공역 {assigned_moa} 외부 비행 중!",
                aircraft_involved=[uac.callsign],
                time_to_event_sec=0,
                recommendation=f"즉시 {assigned_moa} 공역으로 복귀하세요",
                position=(uac.lat, uac.lon),
                event_key=f"AIRSPACE_EXIT:{uac.callsign}:{assigned_moa}",
                ackable=False,
            ))
            return alerts

        # 미래 예측: 곧 나갈 것인가?
        dt = SAFETY_LOOKAHEAD_SEC / SAFETY_LOOKAHEAD_STEPS
        for step in range(1, min(SAFETY_LOOKAHEAD_STEPS + 1, 7)):  # 최대 ~100초
            t = dt * step
            pos = self._extrapolate(uac, t)
            if not self._point_in_polygon(pos[0], pos[1], verts):
                sev = Severity.ALERT if t < 60 else Severity.WARNING
                alerts.append(SafetyAlert(
                    severity=sev,
                    category="AIRSPACE_EXIT",
                    title=f"공역 이탈 예측: {assigned_moa}",
                    message=f"{t:.0f}초 후 {assigned_moa} 공역 경계 이탈 예상",
                    aircraft_involved=[uac.callsign],
                    time_to_event_sec=t,
                    recommendation="선회하여 공역 내 체류하세요",
                    position=(pos[0], pos[1]),
                    event_key=f"AIRSPACE_EXIT:{uac.callsign}:{assigned_moa}",
                    ackable=False,
                ))
                break

        return alerts

    # ── KADIZ 이탈 경고 ──

    def _check_kadiz_exit(self, uac):
        """전투기가 KADIZ를 벗어나면 절차적 + 외교적 문제 (ACK 불가)"""
        alerts = []
        if not KADIZ_POLYGON:
            return alerts

        inside = self._point_in_polygon(uac.lat, uac.lon, KADIZ_POLYGON)

        if not inside:
            alerts.append(SafetyAlert(
                severity=Severity.ALERT,
                category="KADIZ_EXIT",
                title="KADIZ 이탈!",
                message="한국방공식별구역(KADIZ) 외부 비행 중! 즉시 복귀 필요",
                aircraft_involved=[uac.callsign],
                time_to_event_sec=0,
                recommendation="즉시 KADIZ 내부로 복귀하세요",
                position=(uac.lat, uac.lon),
                event_key=f"KADIZ_EXIT:{uac.callsign}",
                ackable=False,
            ))
            return alerts

        # 미래 예측
        dt = SAFETY_LOOKAHEAD_SEC / SAFETY_LOOKAHEAD_STEPS
        for step in range(1, SAFETY_LOOKAHEAD_STEPS + 1):
            t = dt * step
            pos = self._extrapolate(uac, t)
            if not self._point_in_polygon(pos[0], pos[1], KADIZ_POLYGON):
                sev = Severity.ALERT if t < 60 else Severity.WARNING
                alerts.append(SafetyAlert(
                    severity=sev,
                    category="KADIZ_EXIT",
                    title="KADIZ 이탈 예측",
                    message=f"{t:.0f}초 후 KADIZ 경계 이탈 예상",
                    aircraft_involved=[uac.callsign],
                    time_to_event_sec=t,
                    recommendation="방향을 수정하여 KADIZ 내 체류하세요",
                    position=(pos[0], pos[1]),
                    event_key=f"KADIZ_EXIT:{uac.callsign}",
                    ackable=False,
                ))
                break

        return alerts

    # ── Squawk 비상 (7500/7600/7700) ──

    def _check_squawk_emergency(self, all_aircraft, user_aircraft):
        """
        비상 Squawk 항공기 감지. 우선권 부여.
        7700: 비상 (MAYDAY)
        7600: 통신두절 (NORDO)
        7500: 하이잭 (불법 간섭)

        이들 항공기 근처의 아군 전투기에게 경고. ACK 불가.
        """
        alerts = []
        emergency_squawks = {
            '7700': ('MAYDAY', '비상 선언 항공기'),
            '7600': ('NORDO', '통신두절 항공기'),
            '7500': ('HIJACK', '불법간섭 항공기'),
        }

        emrg_aircraft = []
        for ac in all_aircraft:
            sq = getattr(ac, 'squawk', None)
            if sq and sq in emergency_squawks:
                emrg_aircraft.append((ac, sq))

        for emrg_ac, squawk in emrg_aircraft:
            code_name, desc = emergency_squawks[squawk]
            cs = getattr(emrg_ac, 'callsign', getattr(emrg_ac, 'icao24', '?'))

            # 비상 항공기 자체 경고
            alerts.append(SafetyAlert(
                severity=Severity.ALERT,
                category="SQUAWK_EMRG",
                title=f"SQUAWK {squawk} — {code_name}: {cs}",
                message=f"{cs} {desc}. 우선권 즉시 부여. 경로 회피 필요",
                aircraft_involved=[cs],
                time_to_event_sec=0,
                recommendation=f"{cs}에게 즉시 우선권 부여. 진로 방해 금지",
                position=(emrg_ac.lat, emrg_ac.lon),
                event_key=f"SQUAWK_EMRG:{cs}:{squawk}",
                ackable=False,
            ))

            # 아군 전투기가 근처에 있으면 추가 경고
            for uac in user_aircraft:
                d = calculate_distance(uac.lat, uac.lon, emrg_ac.lat, emrg_ac.lon)
                if d < 25:
                    alerts.append(SafetyAlert(
                        severity=Severity.ALERT,
                        category="SQUAWK_EMRG",
                        title=f"{code_name} 근접: {cs} ↔ {uac.callsign}",
                        message=(f"{desc} {cs}가 {d:.1f}NM 거리. "
                                 f"즉시 경로 분리 필요"),
                        aircraft_involved=[uac.callsign, cs],
                        time_to_event_sec=0,
                        recommendation="비상 항공기 경로에서 즉시 벗어나세요",
                        position=(emrg_ac.lat, emrg_ac.lon),
                        event_key=f"SQUAWK_EMRG:{uac.callsign}:{cs}:{squawk}",
                        ackable=False,
                    ))

        return alerts

    # ── ATS 항로 진입 경고 ──

    def _check_ats_route(self, uac):
        """
        전투기가 ATS 항로 위에서 along-track 비행하면 경고 (ACK 불가).
        Crossing은 허용 — 항로를 따라(along) 연속 비행할 때만 경고.
        """
        alerts = []
        if not ATS_ROUTES:
            return alerts

        for route_name, route_data in ATS_ROUTES.items():
            # route_data: list of [lat, lon, id] or dict with 'waypoints'
            if isinstance(route_data, list):
                waypoints = route_data
            else:
                waypoints = route_data.get('waypoints', [])
            if len(waypoints) < 2:
                continue

            # 가장 가까운 항로 세그먼트와의 거리
            min_dist = float('inf')
            closest_seg_bearing = None

            for i in range(len(waypoints) - 1):
                wp_a = waypoints[i]
                wp_b = waypoints[i + 1]
                # Support both [lat, lon, id] list and {'lat':, 'lon':} dict
                if isinstance(wp_a, (list, tuple)):
                    a_lat, a_lon = wp_a[0], wp_a[1]
                else:
                    a_lat, a_lon = wp_a['lat'], wp_a['lon']
                if isinstance(wp_b, (list, tuple)):
                    b_lat, b_lon = wp_b[0], wp_b[1]
                else:
                    b_lat, b_lon = wp_b['lat'], wp_b['lon']

                # 항공기에서 세그먼트까지 대략 거리
                d_a = calculate_distance(uac.lat, uac.lon, a_lat, a_lon)
                d_b = calculate_distance(uac.lat, uac.lon, b_lat, b_lon)
                seg_len = calculate_distance(a_lat, a_lon, b_lat, b_lon)

                if seg_len < 0.1:
                    continue
                if d_a > seg_len + ATS_ROUTE_BUFFER_NM and d_b > seg_len + ATS_ROUTE_BUFFER_NM:
                    continue

                # 세그먼트까지 수직 거리 근사
                seg_brg = calculate_bearing(a_lat, a_lon, b_lat, b_lon)
                ac_brg = calculate_bearing(a_lat, a_lon, uac.lat, uac.lon)
                angle_diff = math.radians(ac_brg - seg_brg)
                cross_dist = abs(d_a * math.sin(angle_diff))
                along = d_a * math.cos(angle_diff)

                if 0 <= along <= seg_len and cross_dist < min_dist:
                    min_dist = cross_dist
                    closest_seg_bearing = seg_brg

            if min_dist > ATS_ROUTE_BUFFER_NM or closest_seg_bearing is None:
                continue

            # 항로 위에 있음 — crossing인지 along인지 판단
            hdg = uac.hdg_current
            hdg_diff = abs((hdg - closest_seg_bearing + 180) % 360 - 180)
            is_along = hdg_diff < 30 or hdg_diff > 150  # 같은 방향 또는 반대 방향

            if is_along:
                alerts.append(SafetyAlert(
                    severity=Severity.WARNING,
                    category="ATS_ALONG",
                    title=f"항로 {route_name} 진입",
                    message=(f"ATS 항로 {route_name} 위에서 along-track 비행 중 "
                             f"(항로 중심선 {min_dist:.1f}NM). Crossing만 허용됩니다"),
                    aircraft_involved=[uac.callsign],
                    time_to_event_sec=0,
                    recommendation=f"항로 {route_name}에서 즉시 벗어나세요",
                    position=(uac.lat, uac.lon),
                    event_key=f"ATS_ALONG:{uac.callsign}:{route_name}",
                    ackable=False,
                ))
                break  # 하나만 보고

        return alerts

    @staticmethod
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

"""
Human ATC + AI Advisory 데모

사람이 항공기를 생성하고 관제(HDG/ALT/SPD 지시)하면,
AI가 Safety Advisory 경고만 표시하는 데모.

조작:
  [N] 항공기 생성 (PyQt5 GUI)
  [I] 선택한 항공기에 지시 (PyQt5 GUI)
  [C] 선택 해제
  클릭: 항공기 선택
  [F] 선택 항공기 추적
  [R] Conflict 분석 새로고침
  [A] Safety Alert ACK (선택 항공기 관련 경고)
  [1-3] 예측 범위 (1=2분, 2=5분, 3=10분)
  [SPACE] AI 예측 일시정지
  [ESC] 종료
"""
import sys
import os
import time
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import pygame

import random
from dataclasses import dataclass, field

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    BLACK, WHITE, GREEN, RED, CYAN, YELLOW, ORANGE, BLUE,
    MOA_LIST, AIRPORTS, STATIC_OBSTACLES, ATS_ROUTES, KADIZ_POLYGON,
    R_ZONE_LIST, LIGHT_GRAY, INITIAL_ZOOM_LEVEL,
    MILITARY_AIRBASES,
)
from core.simulation import Simulation
from core.airspace import AirspaceManager
from ai.safety_advisor import SafetyAdvisor, Severity
from ai.world_model.trajectory_predictor import TrajectoryPredictor
from ai.world_model.conflict_detector import ConflictDetector
from ai.world_model.dataset import STATE_DIM, NORM_MEAN, NORM_STD, MAX_NEIGHBORS, CONTEXT_DIM
from utils import render_text_with_simple_outline

# ── 설정 ──
MODEL_PATH = "models/world_model/best_model.pt"
DREAMER_PATH = "models/dreamer/latest.pt"
PREDICTION_INTERVAL = 3.0
CONFLICT_SCAN_INTERVAL = 5.0
MC_SAMPLES_VIS = 20
PAST_STEPS = 6
HISTORY_BUFFER = 20
RISK_SCAN_INTERVAL = 3.0


def load_model(model_path, device):
    if not os.path.exists(model_path):
        print(f"[ATC] Model not found: {model_path}")
        print("[ATC] Running in linear-extrapolation fallback mode")
        return None
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint.get('args', {})
    model = TrajectoryPredictor(
        hidden_dim=args.get('hidden_dim', 256),
        latent_dim=args.get('latent_dim', 64),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    epoch = checkpoint.get('epoch', '?')
    print(f"[ATC] World Model loaded (epoch={epoch})")
    return model



def linear_extrapolate(ac, steps=12, dt=10.0):
    from config import KNOTS_TO_MPS, M_TO_NM
    trajs = []
    lat, lon, alt = ac.lat, ac.lon, ac.alt_current
    spd_nm_s = ac.ground_speed_kt * KNOTS_TO_MPS * M_TO_NM
    hdg_rad = math.radians(ac.track_true_deg)
    vrate = getattr(ac, 'vertical_rate_ft_min', 0) or 0
    vrate /= 60.0
    for t in range(steps):
        dist = spd_nm_s * dt
        lat += dist * math.cos(hdg_rad) / 60.0
        cos_lat = math.cos(math.radians(lat))
        if abs(cos_lat) > 1e-6:
            lon += dist * math.sin(hdg_rad) / (60.0 * cos_lat)
        alt += vrate * dt
        trajs.append((lat, lon, alt))
    return np.array(trajs)


# ── 시나리오 생성 ──
# 두 가지 케이스:
#   departure: 공항(이륙) → 공역 진입 (목적지 공역/MOA)
#   arrival:   공역 → 공항(착륙) (목적지 공항 30NM, 7500-12500ft 도달 시 APP 인계)

SCENARIO_TYPES = [
    "departure",     # 공항 → 공역 (이륙 후 배정 공역으로)
    "arrival",       # 공역 → 공항 (착륙을 위해 목적지 접근)
]

# APP 인계 조건
APP_HANDOFF_DIST_NM = 30.0
APP_HANDOFF_ALT_MIN = 7500
APP_HANDOFF_ALT_MAX = 12500

@dataclass
class Scenario:
    """관제 시나리오"""
    type: str                # "departure" | "arrival"
    aircraft_callsign: str
    origin: dict             # {"icao": ..., "name": ..., "lat": ..., "lon": ...}
    destination: dict        # 동일 형식
    waypoints: list          # [(lat, lon, alt, label)]  — 목적지 1개
    current_wp_idx: int = 0
    complete: bool = False
    handoff_msg: str = ""    # 완료 시 메시지

    @property
    def current_waypoint(self):
        if self.current_wp_idx < len(self.waypoints):
            return self.waypoints[self.current_wp_idx]
        return None

    @property
    def remaining_waypoints(self):
        return self.waypoints[self.current_wp_idx:]

    def advance(self):
        self.current_wp_idx += 1
        if self.current_wp_idx >= len(self.waypoints):
            self.complete = True


def generate_scenario(ac):
    """
    항공기에 랜덤 시나리오 부여.

    departure: 공항(이륙) → 배정 공역(MOA)
      - 항공기 현재 위치 근처 기지에서 출발, 목적지는 MOA 중심점
      - MOA 공역 진입(10NM 이내) 시 완료

    arrival: 공역 → 공항(착륙)
      - 항공기 현재 위치에서 목적지 기지로 이동
      - 목적지 30NM 이내 + 고도 7500-12500ft 도달 시 APP/RPN 인계 → 완료
    """
    from utils.geo import calculate_distance

    bases = sorted(MILITARY_AIRBASES,
                   key=lambda b: calculate_distance(ac.lat, ac.lon, b["lat"], b["lon"]))
    origin = bases[0]
    stype = random.choice(SCENARIO_TYPES)

    if stype == "departure":
        # 공항 → 공역: 가까운 MOA로 배정
        moas = [m for m in MOA_LIST if m.get("vertices") or m.get("coords")]
        if not moas:
            moas = MOA_LIST[:3]
        moa = random.choice(moas)
        coords = moa.get("vertices") or moa.get("coords") or []
        if coords:
            moa_lat = sum(c[0] for c in coords) / len(coords)
            moa_lon = sum(c[1] for c in coords) / len(coords)
        else:
            moa_lat, moa_lon = 36.5, 127.5
        moa_name = moa.get("name", "MOA")
        cruise_alt = random.choice([15000, 20000, 25000])
        dest = {"icao": "", "name": moa_name, "lat": moa_lat, "lon": moa_lon, "mil": True}
        waypoints = [
            (moa_lat, moa_lon, cruise_alt, moa_name),
        ]
        return Scenario(
            type=stype,
            aircraft_callsign=ac.callsign,
            origin=origin,
            destination=dest,
            waypoints=waypoints,
        )

    else:  # arrival
        # 공역 → 공항: 다른 기지로 착륙 유도
        candidates = [b for b in bases[1:] if
                      calculate_distance(ac.lat, ac.lon, b["lat"], b["lon"]) > 40]
        if not candidates:
            candidates = bases[1:4]
        dest = random.choice(candidates)
        # 목적지 방향 접근 고도 (APP 인계 조건: 7500-12500ft)
        approach_alt = random.choice([8000, 10000, 12000])
        waypoints = [
            (dest["lat"], dest["lon"], approach_alt, f"{dest['name']}"),
        ]
        return Scenario(
            type=stype,
            aircraft_callsign=ac.callsign,
            origin=origin,
            destination=dest,
            waypoints=waypoints,
        )


def get_wp_heading_and_dist(ac_lat, ac_lon, wp_lat, wp_lon):
    """항공기에서 웨이포인트까지 방위/거리 계산"""
    from utils.geo import calculate_distance, calculate_bearing
    dist = calculate_distance(ac_lat, ac_lon, wp_lat, wp_lon)
    brg = calculate_bearing(ac_lat, ac_lon, wp_lat, wp_lon)
    return brg, dist


class AircraftHistory:
    def __init__(self):
        self.states = {}

    def update(self, icao, ac, ts):
        state = np.array([
            ac.lat, ac.lon, ac.alt_current,
            ac.ground_speed_kt, ac.track_true_deg,
            getattr(ac, 'vertical_rate_ft_min', 0) or 0,
            getattr(ac, 'ias_kt', 0) or 0,
            getattr(ac, 'mach', 0) or 0,
            getattr(ac, 'wind_direction_deg', 0) or 0,
            getattr(ac, 'wind_speed_kt', 0) or 0,
        ], dtype=np.float32)
        if icao not in self.states:
            self.states[icao] = []
        self.states[icao].append((ts, state))
        if len(self.states[icao]) > HISTORY_BUFFER:
            self.states[icao] = self.states[icao][-HISTORY_BUFFER:]

    def get_past(self, icao, steps=PAST_STEPS):
        if icao not in self.states:
            return None
        hist = self.states[icao]
        if len(hist) < 2:
            return None
        states = [s for _, s in hist[-steps:]]
        while len(states) < steps:
            states.insert(0, states[0].copy())
        return np.stack(states[-steps:])

    def get_context(self, target_icao):
        ctx = np.zeros((MAX_NEIGHBORS, CONTEXT_DIM), dtype=np.float32)
        if target_icao not in self.states or not self.states[target_icao]:
            return ctx
        t_state = self.states[target_icao][-1][1]
        t_lat, t_lon = t_state[0], t_state[1]
        neighbors = []
        for icao, hist in self.states.items():
            if icao == target_icao or not hist:
                continue
            s = hist[-1][1]
            dx = (s[1] - t_lon) * 60.0 * math.cos(math.radians(t_lat))
            dy = (s[0] - t_lat) * 60.0
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 50:
                continue
            dalt = (s[2] - t_state[2]) / 1000.0
            dgs = (s[3] - t_state[3]) / 100.0
            dtrack = ((s[4] - t_state[4] + 180) % 360 - 180) / 180.0
            dvrate = (s[5] - t_state[5]) / 1000.0
            neighbors.append((dist, [dx, dy, dalt, dgs, dtrack, dvrate]))
        neighbors.sort(key=lambda x: x[0])
        for i, (_, vec) in enumerate(neighbors[:MAX_NEIGHBORS]):
            ctx[i] = vec
        return ctx

    def cleanup(self, valid_icaos):
        stale = [k for k in self.states if k not in valid_icaos]
        for k in stale:
            del self.states[k]


def predict_trajectories(model, history, icao, device, future_steps=12, num_mc=MC_SAMPLES_VIS):
    past = history.get_past(icao, PAST_STEPS)
    if past is None:
        return None
    past_norm = (past - NORM_MEAN) / NORM_STD
    past_t = torch.from_numpy(past_norm).unsqueeze(0).to(device)
    ctx_list = [history.get_context(icao) for _ in range(PAST_STEPS)]
    ctx = np.stack(ctx_list)
    ctx_t = torch.from_numpy(ctx).unsqueeze(0).to(device)
    trajs = model.predict(past_t, ctx_t, num_samples=num_mc, future_steps=future_steps)
    return trajs[0, :, :, :3].cpu().numpy()


def main():
    import atexit
    import signal

    parser = argparse.ArgumentParser(description="Human ATC + AI Advisory Demo")
    parser.add_argument('--replay', default=None, help='ADS-B replay file (.jsonl or .db)')
    parser.add_argument('--speed', type=float, default=1.0, help='Replay speed multiplier')
    parser.add_argument('--live', action='store_true', help='Live ADS-B mode')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, device)

    # 데이터 소스 결정
    replay_src = args.replay
    if not replay_src and not args.live:
        rec_dir = os.path.join(os.path.dirname(__file__), "data", "recordings")
        if os.path.isdir(rec_dir):
            rec_files = sorted([f for f in os.listdir(rec_dir) if f.endswith(".jsonl")])
            if rec_files:
                replay_src = os.path.join(rec_dir, rec_files[-1])

    if replay_src:
        print(f"[ATC] Replay: {replay_src} (x{args.speed})")
        sim = Simulation(
            use_gui=True, use_pygame=True,
            replay_source=replay_src, replay_speed=args.speed, replay_loop=True,
        )
    else:
        print("[ATC] Live ADS-B mode")
        sim = Simulation(use_gui=True, use_pygame=True)

    advisor = SafetyAdvisor(
        world_model_path=MODEL_PATH if model else None,
        dreamer_path=DREAMER_PATH,
    )
    airspace_mgr = AirspaceManager()
    ac_history = AircraftHistory()
    clock = pygame.time.Clock()

    # ADS-B 대기
    print("[ATC] Waiting for ADS-B data...")
    t0 = time.time()
    while not sim.other_aircraft and time.time() - t0 < 10:
        sim.process_external_aircraft_queue()
        time.sleep(0.3)
    print(f"[ATC] {len(sim.other_aircraft)} aircraft loaded")

    # 상태
    selected_icao = None
    follow_aircraft = False
    pred_trajs = None
    last_pred_time = 0
    future_steps = 12
    panel_clickables = []
    paused = False
    scenarios = {}  # callsign -> Scenario

    print("=" * 55)
    print("  Human ATC + AI Safety Advisory")
    print("  [N] Create aircraft  [I] Instruct selected")
    print("  [C] Deselect  [F] Follow  [A] ACK alert")
    print("  [T] Assign scenario  [R] Refresh conflicts")
    print("  [1-3] Pred range  [SPACE] Pause AI  [ESC] Quit")
    print("=" * 55)

    # 비정상 종료 시에도 자식 프로세스 정리 보장
    _cleaned_up = False
    def _cleanup():
        nonlocal _cleaned_up
        if _cleaned_up:
            return
        _cleaned_up = True
        try:
            sim.stop()
        except Exception:
            pass

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda *_: (_cleanup(), exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), exit(0)))

    try:
      while sim.running:
        dt = clock.tick(30) / 1000.0
        now = time.time()

        # ── GUI 메시지 처리 (항공기 생성/지시) ──
        sim._check_gui_messages()

        # ── 이벤트 처리 ──
        for event in pygame.event.get():
            # 좌표 입력 대기 중이면 맵 클릭으로 좌표 전달
            if sim.awaiting_gui_coords and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                from utils import decimal_to_dms
                lat, lon = sim.map.screen_to_latlon(*event.pos)
                sim.parent_conn.send({"type": "coords", "data": {
                    "lat": decimal_to_dms(lat, False, 0), "lon": decimal_to_dms(lon, False, 1)}})
                sim.awaiting_gui_coords = False
                continue

            if sim.map.handle_event(event):
                continue

            if event.type == pygame.QUIT:
                sim.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sim.running = False
                elif event.key == pygame.K_n and sim.use_gui_process:
                    sim.parent_conn.send({"type": "action", "action": "create_aircraft"})
                elif event.key == pygame.K_i and sim.use_gui_process:
                    if sim.selected_aircraft and sim.selected_aircraft.is_user_controlled:
                        sim.parent_conn.send({"type": "action", "action": "instruction", "aircraft_data": {
                            "callsign": sim.selected_aircraft.callsign,
                            "alt": sim.selected_aircraft.alt_target,
                            "spd": sim.selected_aircraft.spd_target,
                            "hdg": sim.selected_aircraft.hdg_target,
                        }})
                    else:
                        print("[ATC] Select a user-controlled aircraft first (press N to create)")
                elif event.key == pygame.K_c:
                    sim.selected_aircraft = None
                    selected_icao = None
                elif event.key == pygame.K_f:
                    follow_aircraft = not follow_aircraft
                elif event.key == pygame.K_r:
                    last_conflict_time = 0
                elif event.key == pygame.K_a:
                    # ACK: 선택 항공기 관련 경고 확인
                    if selected_icao:
                        cs = getattr(sim.selected_aircraft, 'callsign', selected_icao) if sim.selected_aircraft else selected_icao
                        related = advisor.get_alerts_for_aircraft(cs)
                        for a in related:
                            if a.ackable:
                                advisor.acknowledge(a.event_key)
                        if related:
                            print(f"[ATC] ACK {len(related)} alerts for {cs}")
                elif event.key == pygame.K_t:
                    # 시나리오 부여
                    if sim.selected_aircraft and sim.selected_aircraft.is_user_controlled:
                        sc = generate_scenario(sim.selected_aircraft)
                        scenarios[sim.selected_aircraft.callsign] = sc
                        advisor.set_scenario(sim.selected_aircraft.callsign, sc)
                        wp = sc.current_waypoint
                        if sc.type == "departure":
                            print(f"[SCENARIO] DEPARTURE: {sc.origin['name']} -> {sc.destination['name']} (공역 진입 10NM)")
                        else:
                            print(f"[SCENARIO] ARRIVAL: {sc.origin['name']} -> {sc.destination['name']} (APP 인계 30NM, 7500-12500ft)")
                        wlat, wlon, walt, wlabel = wp
                        print(f"  Target: {wlabel} ({wlat:.2f}, {wlon:.2f}) FL{walt/100:.0f}")
                    else:
                        print("[ATC] Select a user-controlled aircraft first (press N to create)")
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_1:
                    future_steps = 12; pred_trajs = None
                elif event.key == pygame.K_2:
                    future_steps = 30; pred_trajs = None
                elif event.key == pygame.K_3:
                    future_steps = 60; pred_trajs = None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # 패널 클릭 확인
                panel_hit = None
                for rect, target in panel_clickables:
                    if rect.collidepoint(mx, my):
                        panel_hit = target
                        break
                if panel_hit:
                    if isinstance(panel_hit, tuple) and len(panel_hit) == 2:
                        hit_icao, hit_pos = panel_hit
                    else:
                        hit_icao, hit_pos = panel_hit, None
                    found_ac = all_tracked.get(hit_icao)
                    if not found_ac:
                        for k, ac in all_tracked.items():
                            if getattr(ac, 'callsign', '') == hit_icao:
                                found_ac = ac
                                hit_icao = k
                                break
                    if found_ac:
                        selected_icao = hit_icao
                        sim.selected_aircraft = found_ac
                        pred_trajs = None
                        last_pred_time = 0
                        follow_aircraft = True
                    elif hit_pos:
                        sim.map.center_lat = hit_pos[0]
                        sim.map.center_lon = hit_pos[1]
                else:
                    # 항공기 클릭
                    clicked = sim._find_aircraft_at(mx, my)
                    if clicked:
                        selected_icao = getattr(clicked, 'icao24', None) or clicked.callsign
                        sim.selected_aircraft = clicked
                        pred_trajs = None
                        last_pred_time = 0
            elif event.type == pygame.MOUSEMOTION and sim.font:
                from utils import decimal_to_dms
                lat, lon = sim.map.screen_to_latlon(*event.pos)
                sim.mouse_latlon_text = f"Lat: {decimal_to_dms(lat, True, 0)}, Lon: {decimal_to_dms(lon, True, 1)}"

        # ── 시뮬레이션 업데이트 ──
        sim.process_external_aircraft_queue()
        sim.map.process_tile_results()
        if now - sim._last_map_tile_check > 0.5:
            sim.map.update_needed_tiles()
            sim._last_map_tile_check = now

        for ac in sim.user_aircraft:
            ac.update(dt)
        for ac in list(sim.other_aircraft.values()):
            ac.update(dt)

        # ── 시나리오 완료 체크 ──
        for ac in sim.user_aircraft:
            sc = scenarios.get(ac.callsign)
            if sc and not sc.complete:
                wp = sc.current_waypoint
                if wp:
                    wp_lat, wp_lon, wp_alt, wp_label = wp
                    brg, dist_nm = get_wp_heading_and_dist(ac.lat, ac.lon, wp_lat, wp_lon)

                    if sc.type == "arrival":
                        # APP/RPN 인계 조건: 30NM 이내 + 7500-12500ft
                        alt = ac.alt_current
                        if dist_nm < APP_HANDOFF_DIST_NM and APP_HANDOFF_ALT_MIN <= alt <= APP_HANDOFF_ALT_MAX:
                            sc.complete = True
                            sc.handoff_msg = f"APP 인계 완료 ({dist_nm:.1f}NM, {alt:.0f}ft)"
                            print(f"[SCENARIO] {ac.callsign} -> {wp_label} APP/RPN HANDOFF ({dist_nm:.1f}NM, FL{alt/100:.0f})")
                        elif dist_nm < APP_HANDOFF_DIST_NM:
                            # 거리는 됐지만 고도가 안 맞음
                            if alt > APP_HANDOFF_ALT_MAX:
                                print(f"[SCENARIO] {ac.callsign} {wp_label} {dist_nm:.1f}NM - 고도 {alt:.0f}ft 너무 높음 (12500ft 이하로 강하)")
                            elif alt < APP_HANDOFF_ALT_MIN:
                                print(f"[SCENARIO] {ac.callsign} {wp_label} {dist_nm:.1f}NM - 고도 {alt:.0f}ft 너무 낮음 (7500ft 이상 유지)")

                    elif sc.type == "departure":
                        # 공역 진입: 10NM 이내 도달
                        if dist_nm < 10.0:
                            sc.complete = True
                            sc.handoff_msg = f"공역 진입 완료 ({wp_label})"
                            print(f"[SCENARIO] {ac.callsign} entered {wp_label} airspace ({dist_nm:.1f}NM) - COMPLETE")

        sim.check_tcas()
        sim.remove_stale_aircraft()
        airspace_mgr.update(dt)

        # 임무 전투기
        patrol_fighters = airspace_mgr.get_patrol_fighters()
        pf_aircraft = {}
        for pf in patrol_fighters:
            pf_key = f'PF_{pf.callsign}'
            pf_ac = type('PF', (), {
                'lat': pf.lat, 'lon': pf.lon,
                'alt_current': pf.alt, 'callsign': pf.callsign,
                'is_user_controlled': False,
                'is_military': True,
                'ground_speed_kt': pf.spd,
                'track_true_deg': pf.hdg,
                'vertical_rate_ft_min': 0,
                'hdg_current': pf.hdg, 'spd_current': pf.spd,
                'tcas_status': None, 'icao24': pf_key,
                'moa': pf.moa_name,
                'last_external_update': now,
                'instruction_active': False, 'instruction_pending': False,
                'color': (255, 180, 50),
            })()
            pf_ac.draw = lambda *a, **k: None
            pf_ac.update = lambda *a, **k: None
            pf_aircraft[pf_key] = pf_ac

        # 통합 항공기 (사용자 + ADS-B + 전투기)
        all_tracked = {**sim.other_aircraft, **pf_aircraft}
        # 사용자 항공기도 tracked에 추가 (AI 분석 대상)
        for uac in sim.user_aircraft:
            uid = getattr(uac, 'icao24', None) or f'USER_{uac.callsign}'
            all_tracked[uid] = uac

        # selected_icao 동기화
        if sim.selected_aircraft:
            sid = getattr(sim.selected_aircraft, 'icao24', None) or sim.selected_aircraft.callsign
            if sid.startswith('PF_') or sid in all_tracked:
                selected_icao = sid
            else:
                # user aircraft
                for k, v in all_tracked.items():
                    if v is sim.selected_aircraft:
                        selected_icao = k
                        break

        # 이력 업데이트
        for icao, ac in all_tracked.items():
            ac_history.update(icao, ac, now)
        ac_history.cleanup(set(all_tracked.keys()))

        # ── Safety Advisor (통합: AI+룰 충돌탐지 + 위험도 + 경고) ──
        if sim.user_aircraft:
            advisor.update(sim.user_aircraft, all_tracked)
            all_alerts = advisor.get_alerts(min_severity=Severity.CAUTION)
        else:
            all_alerts = []

        # alerts를 카테고리별 분리
        conflict_alerts = [a for a in all_alerts if a.category == "CONFLICT"]
        risk_alerts = [a for a in all_alerts if a.category == "AI_RISK"]
        other_alerts = [a for a in all_alerts if a.category not in ("CONFLICT", "AI_RISK")]
        alerts = all_alerts  # 전체 (HUD 카운트용)

        # ── AI 궤적 예측 (선택 항공기 시각화용) ──
        if not paused and selected_icao and selected_icao in all_tracked:
            if now - last_pred_time >= PREDICTION_INTERVAL:
                last_pred_time = now
                if model:
                    pred_trajs = predict_trajectories(
                        model, ac_history, selected_icao, device,
                        future_steps=future_steps, num_mc=MC_SAMPLES_VIS)
                else:
                    ac = all_tracked[selected_icao]
                    lin = linear_extrapolate(ac, steps=future_steps)
                    pred_trajs = lin[np.newaxis, :, :]

        # ══════════════════════════════════════
        #               렌더링
        # ══════════════════════════════════════
        sim.screen.fill(BLACK)
        panel_clickables = []

        # 카메라 추적
        if follow_aircraft and selected_icao and selected_icao in all_tracked:
            ac = all_tracked[selected_icao]
            sim.map.center_lat = ac.lat
            sim.map.center_lon = ac.lon

        sim.map.draw(sim.screen)

        # KADIZ
        kadiz_pts = [sim.map.latlon_to_screen(v[0], v[1]) for v in KADIZ_POLYGON]
        kadiz_int = [(int(p[0]), int(p[1])) for p in kadiz_pts]
        if len(kadiz_int) >= 3:
            pygame.draw.polygon(sim.screen, (200, 200, 0), kadiz_int, 2)

        # 공역
        moa_status = airspace_mgr.get_moa_status()
        rzone_status = airspace_mgr.get_rzone_status()
        all_airspaces = (
            [(moa, 'moa') for moa in MOA_LIST] +
            [(rz, 'rzone') for rz in R_ZONE_LIST] +
            [(obs, 'prd') for obs in STATIC_OBSTACLES]
        )
        for airspace, atype in all_airspaces:
            verts = airspace.get("vertices")
            if not verts or len(verts) < 3:
                continue
            screen_pts = [sim.map.latlon_to_screen(v[0], v[1]) for v in verts]
            avg_x = sum(p[0] for p in screen_pts) / len(screen_pts)
            avg_y = sum(p[1] for p in screen_pts) / len(screen_pts)
            if avg_x < -500 or avg_x > SCREEN_WIDTH + 500 or avg_y < -500 or avg_y > SCREEN_HEIGHT + 500:
                continue
            int_pts = [(int(p[0]), int(p[1])) for p in screen_pts]

            if atype == 'moa':
                is_hot = moa_status.get(airspace["name"], False)
                fill_color = (255, 50, 50, 60) if is_hot else (100, 100, 100, 30)
                border_color = (255, 80, 80) if is_hot else (120, 120, 120)
                label = f"{airspace['name']} {'HOT' if is_hot else 'COLD'}"
            elif atype == 'rzone':
                is_hot = rzone_status.get(airspace["name"], False)
                fill_color = (255, 100, 0, 60) if is_hot else (80, 80, 80, 20)
                border_color = (255, 130, 0) if is_hot else (100, 100, 100)
                label = f"{airspace['name']} {'HOT' if is_hot else 'COLD'}"
            else:
                fill_color = (255, 165, 0, 40)
                border_color = (255, 165, 0)
                label = airspace["name"]

            xs = [p[0] for p in int_pts]
            ys = [p[1] for p in int_pts]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            w, h = max_x - min_x + 1, max_y - min_y + 1
            if 0 < w < 3000 and 0 < h < 3000:
                poly_surf = pygame.Surface((w, h), pygame.SRCALPHA)
                shifted = [(x - min_x, y - min_y) for x, y in int_pts]
                try:
                    pygame.draw.polygon(poly_surf, fill_color, shifted)
                    sim.screen.blit(poly_surf, (min_x, min_y))
                except Exception:
                    pass
            if len(int_pts) >= 3:
                pygame.draw.polygon(sim.screen, border_color, int_pts, 2)
            s = render_text_with_simple_outline(sim.font, label, border_color, BLACK)
            sim.screen.blit(s, (int(avg_x) - s.get_width() // 2, int(avg_y) - 8))

        # ── Conflict 라인 (alert 기반) ──
        for alert in conflict_alerts:
            if len(alert.aircraft_involved) >= 2 and alert.position:
                cs_a, cs_b = alert.aircraft_involved[0], alert.aircraft_involved[1]
                ac_a = all_tracked.get(cs_a)
                ac_b = all_tracked.get(cs_b)
                if not ac_a:
                    for k, v in all_tracked.items():
                        if getattr(v, 'callsign', '') == cs_a:
                            ac_a = v; break
                if not ac_b:
                    for k, v in all_tracked.items():
                        if getattr(v, 'callsign', '') == cs_b:
                            ac_b = v; break
                if ac_a and ac_b:
                    ax, ay = sim.map.latlon_to_screen(ac_a.lat, ac_a.lon)
                    bx, by = sim.map.latlon_to_screen(ac_b.lat, ac_b.lon)
                    sev_col = {Severity.ALERT: RED, Severity.WARNING: ORANGE, Severity.CAUTION: YELLOW}
                    line_color = sev_col.get(alert.severity, YELLOW)
                    pygame.draw.line(sim.screen, line_color, (int(ax), int(ay)), (int(bx), int(by)), 2)
                    mx, my = (int(ax) + int(bx)) // 2, (int(ay) + int(by)) // 2
                    s = render_text_with_simple_outline(sim.font, alert.title, line_color, BLACK)
                    sim.screen.blit(s, (mx - s.get_width() // 2, my - 10))

        # ── 선택 항공기 예측 궤적 ──
        if pred_trajs is not None and selected_icao in all_tracked:
            mc, T, _ = pred_trajs.shape
            for s_idx in range(mc):
                pts = []
                for t in range(T):
                    lat, lon = pred_trajs[s_idx, t, 0], pred_trajs[s_idx, t, 1]
                    sx, sy = sim.map.latlon_to_screen(lat, lon)
                    pts.append((int(sx), int(sy)))
                if len(pts) >= 2:
                    for i in range(len(pts) - 1):
                        pygame.draw.line(sim.screen, (0, 255, 200), pts[i], pts[i+1], 1)

            if mc > 1:
                final_lats = pred_trajs[:, -1, 0]
                final_lons = pred_trajs[:, -1, 1]
                mean_lat = final_lats.mean()
                mean_lon = final_lons.mean()
                spread_lat = final_lats.std() * 60
                spread_lon = final_lons.std() * 60 * math.cos(math.radians(mean_lat))
                spread_nm = math.sqrt(spread_lat**2 + spread_lon**2)
                cx, cy = sim.map.latlon_to_screen(mean_lat, mean_lon)
                ref_x, ref_y = sim.map.latlon_to_screen(mean_lat + 1/60, mean_lon)
                px_per_nm = abs(ref_y - cy) if abs(ref_y - cy) > 0 else 1
                radius = max(5, int(spread_nm * px_per_nm))
                uncert_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(uncert_surf, (0, 255, 200, 40), (radius, radius), radius)
                sim.screen.blit(uncert_surf, (int(cx) - radius, int(cy) - radius))
                s = render_text_with_simple_outline(
                    sim.font, f"spread: {spread_nm:.1f}NM", (0, 255, 200), BLACK)
                sim.screen.blit(s, (int(cx) + radius + 5, int(cy) - 8))

        # ── 항공기 렌더링 ──
        for ac in list(sim.other_aircraft.values()):
            ac.draw(sim.screen, sim.map, sim.font)
        for ac in sim.user_aircraft:
            ac.draw(sim.screen, sim.map, sim.font)
        sim._highlight_selected()

        # ── Risk 오버레이 (AI_RISK 경고 기반) ──
        for alert in risk_alerts:
            cs = alert.aircraft_involved[0] if alert.aircraft_involved else None
            if not cs:
                continue
            ac = all_tracked.get(cs)
            if not ac:
                for k, v in all_tracked.items():
                    if getattr(v, 'callsign', '') == cs:
                        ac = v
                        break
            if not ac:
                continue
            sx, sy = sim.map.latlon_to_screen(ac.lat, ac.lon)
            sx, sy = int(sx), int(sy)
            if sx < -50 or sx > SCREEN_WIDTH + 50 or sy < -50 or sy > SCREEN_HEIGHT + 50:
                continue
            sev_col = {Severity.ALERT: RED, Severity.WARNING: ORANGE, Severity.CAUTION: YELLOW}
            col = sev_col.get(alert.severity, YELLOW)
            rtxt = render_text_with_simple_outline(sim.font, alert.title, col, BLACK)
            sim.screen.blit(rtxt, (sx + 18, sy - 10))

        # 임무 전투기 렌더링
        for pf in patrol_fighters:
            px, py = sim.map.latlon_to_screen(pf.lat, pf.lon)
            px, py = int(px), int(py)
            if -50 <= px < SCREEN_WIDTH + 50 and -50 <= py < SCREEN_HEIGHT + 50:
                sz = 8
                rad = math.radians(pf.hdg)
                pts = [
                    (px + sz * math.sin(rad), py - sz * math.cos(rad)),
                    (px + sz * 0.7 * math.sin(rad + 2.5), py - sz * 0.7 * math.cos(rad + 2.5)),
                    (px + sz * 0.7 * math.sin(rad - 2.5), py - sz * 0.7 * math.cos(rad - 2.5)),
                ]
                pts_int = [(int(p[0]), int(p[1])) for p in pts]
                pygame.draw.polygon(sim.screen, (255, 180, 50), pts_int)
                lbl = render_text_with_simple_outline(sim.font, pf.callsign, (255, 180, 50), BLACK)
                sim.screen.blit(lbl, (px - lbl.get_width() // 2, py + 12))

        font = sim.font

        # ── HUD (좌측 상단) ──
        sel_info = ""
        if sim.selected_aircraft:
            ac = sim.selected_aircraft
            cs = getattr(ac, 'callsign', selected_icao or '?')
            is_user = ac.is_user_controlled
            tag = "[CTRL]" if is_user else "[MIL]" if (selected_icao or '').startswith('PF_') else "[CIV]"
            risk = 0  # risk는 이제 alerts에서 표시
            risk_str = f"RISK {risk:.0%}" if risk > 0 else ""

            if is_user:
                # 사용자 항공기: 지시 상태 표시
                if ac.instruction_pending:
                    instr_str = f"Pending ({ac.delay_timer:.1f}s)"
                elif ac.instruction_active:
                    instr_str = "Active"
                else:
                    instr_str = "Idle"
                sel_info = (f"Selected: {tag} {cs} | ALT {ac.alt_current:.0f}"
                            f"({ac.alt_target:.0f})ft SPD {ac.spd_current:.0f}"
                            f"({ac.spd_target:.0f})kt HDG {ac.hdg_current:.0f}"
                            f"({ac.hdg_target:.0f}) | {instr_str} {risk_str}")
            else:
                sel_info = (f"Selected: {tag} {cs} | ALT {ac.alt_current:.0f}ft "
                            f"GS {ac.ground_speed_kt:.0f}kt HDG {ac.track_true_deg:.0f} "
                            f"{risk_str}")

        mode = "World Model" if model else "Linear Extrapolation"
        pred_range = {12: "2min", 30: "5min", 60: "10min"}.get(future_steps, f"{future_steps}steps")
        n_user = len(sim.user_aircraft)
        n_ext = len(sim.other_aircraft)
        hud_lines = [
            f"Human ATC + AI Advisory [{mode}] | User: {n_user} External: {n_ext}",
            f"Pred: {pred_range} | Conflicts: {len(conflict_alerts)} | Risks: {len(risk_alerts)} | Alerts: {len(other_alerts)}",
            sel_info,
            f"[N]Create [I]Instruct [C]Clear [F]{'Follow' if follow_aircraft else 'Free'} [A]ACK | {'PAUSED' if paused else 'Live'}",
        ]

        hud_h = len(hud_lines) * 20 + 10
        hud_bg = pygame.Surface((700, hud_h), pygame.SRCALPHA)
        hud_bg.fill((0, 0, 0, 180))
        sim.screen.blit(hud_bg, (5, 5))
        hud_y = 10
        for line in hud_lines:
            if line:
                s = render_text_with_simple_outline(font, line, WHITE, BLACK)
                sim.screen.blit(s, (10, hud_y))
            hud_y += 20

        # ── Conflict 패널 (우측 상단) ──
        sev_colors = {Severity.CAUTION: YELLOW, Severity.WARNING: ORANGE, Severity.ALERT: RED}
        panel_w = 420
        panel_x = SCREEN_WIDTH - panel_w - 10
        panel_top = 10

        if conflict_alerts:
            panel_h = min(len(conflict_alerts[:6]) * 40 + 30, 300)
            bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 200))
            sim.screen.blit(bg, (panel_x, panel_top))
            s = font.render("CONFLICT (AI+RULE)", True, WHITE)
            sim.screen.blit(s, (panel_x + 10, panel_top + 5))
            ay = panel_top + 26
            for alert in conflict_alerts[:6]:
                item_y = ay
                col = sev_colors.get(alert.severity, WHITE)
                s = font.render(f"[{alert.severity.name}] {alert.title}", True, col)
                sim.screen.blit(s, (panel_x + 10, ay))
                ay += 16
                msg = alert.message[:52] + "..." if len(alert.message) > 52 else alert.message
                s = font.render(msg, True, (180, 180, 180))
                sim.screen.blit(s, (panel_x + 15, ay))
                ay += 22
                click_rect = pygame.Rect(panel_x, item_y, panel_w, ay - item_y)
                target = alert.aircraft_involved[0] if alert.aircraft_involved else None
                if target:
                    panel_clickables.append((click_rect, (target, alert.position)))
            panel_top = panel_top + panel_h + 5

        # ── AI Risk 패널 (우측 중간) ──
        if risk_alerts:
            panel_h = min(len(risk_alerts[:5]) * 38 + 30, 250)
            bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 200))
            sim.screen.blit(bg, (panel_x, panel_top))
            s = font.render("RISK FACTORS", True, WHITE)
            sim.screen.blit(s, (panel_x + 10, panel_top + 5))
            ay = panel_top + 26
            for alert in risk_alerts[:5]:
                item_y = ay
                col = sev_colors.get(alert.severity, WHITE)
                s = font.render(f"[{alert.severity.name}] {alert.title}", True, col)
                sim.screen.blit(s, (panel_x + 10, ay))
                ay += 16
                # 세부 요인 표시 (message에 SEP/CONV/ALT/SPD/AI 포함)
                msg = alert.message[:56] + "..." if len(alert.message) > 56 else alert.message
                s = font.render(msg, True, (180, 180, 180))
                sim.screen.blit(s, (panel_x + 15, ay))
                ay += 20
                click_rect = pygame.Rect(panel_x, item_y, panel_w, ay - item_y)
                target = alert.aircraft_involved[0] if alert.aircraft_involved else None
                if target:
                    panel_clickables.append((click_rect, (target, alert.position)))
            panel_top = panel_top + panel_h + 5

        # ── Safety Alerts 패널 (우측 하단) ──
        if other_alerts:
            panel_h = min(len(other_alerts[:6]) * 40 + 30, 300)
            ay = SCREEN_HEIGHT - panel_h - 10
            bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 200))
            sim.screen.blit(bg, (panel_x, ay))
            s = font.render("SAFETY ALERTS", True, WHITE)
            sim.screen.blit(s, (panel_x + 10, ay + 5))
            ay += 26
            for alert in other_alerts[:6]:
                item_y = ay
                col = sev_colors.get(alert.severity, WHITE)
                s = font.render(f"[{alert.severity.name}] {alert.title}", True, col)
                sim.screen.blit(s, (panel_x + 10, ay))
                ay += 16
                msg = alert.message[:52] + "..." if len(alert.message) > 52 else alert.message
                s = font.render(msg, True, (200, 200, 200))
                sim.screen.blit(s, (panel_x + 15, ay))
                ay += 22
                click_rect = pygame.Rect(panel_x, item_y, panel_w, ay - item_y)
                target = alert.aircraft_involved[0] if alert.aircraft_involved else None
                if target:
                    panel_clickables.append((click_rect, (target, alert.position)))

        # ── 사용자 항공기 목록 (좌측 하단) ──
        if sim.user_aircraft:
            ua_panel_w = 400
            ua_panel_h = min(len(sim.user_aircraft) * 22 + 30, 200)
            ua_x = 5
            ua_y = SCREEN_HEIGHT - ua_panel_h - 10
            ua_bg = pygame.Surface((ua_panel_w, ua_panel_h), pygame.SRCALPHA)
            ua_bg.fill((0, 0, 40, 200))
            sim.screen.blit(ua_bg, (ua_x, ua_y))
            s = render_text_with_simple_outline(font, "YOUR AIRCRAFT", CYAN, BLACK)
            sim.screen.blit(s, (ua_x + 8, ua_y + 5))
            uy = ua_y + 26
            for uac in sim.user_aircraft:
                status = ""
                if uac.instruction_pending:
                    status = f"PENDING({uac.delay_timer:.0f}s)"
                elif uac.instruction_active:
                    status = "ACTIVE"
                else:
                    status = "IDLE"
                line = (f"{uac.callsign:8s}  ALT {uac.alt_current:.0f}ft "
                        f"SPD {uac.spd_current:.0f}kt HDG {uac.hdg_current:.0f} "
                        f"| {status}")
                col = GREEN if uac is sim.selected_aircraft else CYAN
                s = render_text_with_simple_outline(font, line, col, BLACK)
                sim.screen.blit(s, (ua_x + 8, uy))
                uy += 22

        # ── 시나리오 웨이포인트 경로 & 패널 ──
        for ac in sim.user_aircraft:
            sc = scenarios.get(ac.callsign)
            if not sc or sc.complete:
                continue
            # 웨이포인트 경로선 + Bearing/Range 그리기
            ac_sx, ac_sy = sim.map.latlon_to_screen(ac.lat, ac.lon)
            pts = [(int(ac_sx), int(ac_sy))]
            for wlat, wlon, walt, wlabel in sc.remaining_waypoints:
                wx, wy = sim.map.latlon_to_screen(wlat, wlon)
                pts.append((int(wx), int(wy)))
            if len(pts) >= 2:
                pygame.draw.lines(sim.screen, YELLOW, False, pts, 2)
                # 경로선 중간에 Bearing/Range 표시
                wp = sc.current_waypoint
                if wp:
                    brg, rng = get_wp_heading_and_dist(ac.lat, ac.lon, wp[0], wp[1])
                    mid_x = (pts[0][0] + pts[1][0]) // 2
                    mid_y = (pts[0][1] + pts[1][1]) // 2
                    br_text = f"BRG {brg:.0f} / {rng:.1f}NM"
                    s = render_text_with_simple_outline(font, br_text, YELLOW, BLACK)
                    sim.screen.blit(s, (mid_x - s.get_width() // 2, mid_y - 12))
            # 웨이포인트 마커
            for i, (wlat, wlon, walt, wlabel) in enumerate(sc.waypoints):
                sx, sy = sim.map.latlon_to_screen(wlat, wlon)
                color = GREEN if i < sc.current_wp_idx else (YELLOW if i == sc.current_wp_idx else LIGHT_GRAY)
                pygame.draw.circle(sim.screen, color, (int(sx), int(sy)), 6, 2)
                s = font.render(f"{wlabel} FL{walt/100:.0f}", True, color)
                sim.screen.blit(s, (int(sx) + 8, int(sy) - 8))

        # 선택 항공기 시나리오 패널 (좌측 중간)
        if sim.selected_aircraft:
            sc = scenarios.get(sim.selected_aircraft.callsign)
            if sc and not sc.complete:
                wp = sc.current_waypoint
                if wp:
                    wp_lat, wp_lon, wp_alt, wp_label = wp
                    brg, dist_nm = get_wp_heading_and_dist(
                        sim.selected_aircraft.lat, sim.selected_aircraft.lon, wp_lat, wp_lon)
                    sc_panel_w = 420
                    sc_panel_h = 80
                    sc_x, sc_y = 5, SCREEN_HEIGHT // 2 - sc_panel_h // 2
                    sc_bg = pygame.Surface((sc_panel_w, sc_panel_h), pygame.SRCALPHA)
                    sc_bg.fill((40, 40, 0, 200))
                    sim.screen.blit(sc_bg, (sc_x, sc_y))
                    if sc.type == "arrival":
                        alt = sim.selected_aircraft.alt_current
                        alt_ok = APP_HANDOFF_ALT_MIN <= alt <= APP_HANDOFF_ALT_MAX
                        dist_ok = dist_nm < APP_HANDOFF_DIST_NM
                        cond_color = GREEN if (alt_ok and dist_ok) else (YELLOW if dist_ok else WHITE)
                        title = f"ARRIVAL -> {wp_label} (APP 30NM, 7.5-12.5kft)"
                        cond = f"DIST {'OK' if dist_ok else f'{dist_nm:.0f}NM'} | ALT {'OK' if alt_ok else f'{alt:.0f}ft'}"
                    else:
                        cond_color = WHITE
                        title = f"DEPARTURE -> {wp_label} (10NM)"
                        cond = f"DIST {dist_nm:.0f}NM"
                    s = render_text_with_simple_outline(font, title, YELLOW, BLACK)
                    sim.screen.blit(s, (sc_x + 8, sc_y + 5))
                    s = render_text_with_simple_outline(font,
                        f"HDG {brg:.0f} | {dist_nm:.1f}NM | FL{wp_alt/100:.0f}",
                        WHITE, BLACK)
                    sim.screen.blit(s, (sc_x + 8, sc_y + 25))
                    route_str = f"{sc.origin['name']} -> {sc.destination['name']} | {cond}"
                    s = render_text_with_simple_outline(font, route_str, cond_color, BLACK)
                    sim.screen.blit(s, (sc_x + 8, sc_y + 45))

        pygame.display.flip()

    finally:
        _cleanup()
        print("[ATC] Done.")


if __name__ == "__main__":
    main()

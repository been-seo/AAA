"""
World Model Safety Advisory 데모

실시간 ADS-B 트래픽에 대해:
1. 선택한 항공기의 미래 궤적을 Monte Carlo로 예측 (불확실성 팬 시각화)
2. 항공기 쌍 간 conflict 확률 실시간 표시
3. Safety Advisory 경고 패널

조작:
  클릭: 항공기 선택 → 궤적 예측 표시
  [F] 선택 항공기 추적
  [R] conflict 분석 새로고침
  [1-3] 예측 스텝 수 (1=2분, 2=5분, 3=10분)
  [ESC] 종료
"""
import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import pygame

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    BLACK, WHITE, GREEN, RED, CYAN, YELLOW, ORANGE, BLUE,
    MOA_LIST, AIRPORTS, STATIC_OBSTACLES, ATS_ROUTES, KADIZ_POLYGON,
    R_ZONE_LIST, LIGHT_GRAY,
)
from core.flight_plan import FlightPlanExtractor
from core.simulation import Simulation
from core.airspace import AirspaceManager
from ai.safety_advisor import SafetyAdvisor, Severity
from ai.world_model.trajectory_predictor import TrajectoryPredictor
from ai.world_model.conflict_detector import ConflictDetector
from ai.world_model.dataset import STATE_DIM, NORM_MEAN, NORM_STD, MAX_NEIGHBORS, CONTEXT_DIM
from utils import calculate_distance, render_text_with_simple_outline


# ── 설정 ──
MODEL_PATH = "models/world_model/best_model.pt"
PREDICTION_INTERVAL = 3.0      # 예측 갱신 주기 (초)
CONFLICT_SCAN_INTERVAL = 5.0   # conflict 분석 주기 (초)
MC_SAMPLES_VIS = 20            # 시각화용 Monte Carlo 샘플 수
PAST_STEPS = 6
HISTORY_BUFFER = 20            # 항공기별 상태 이력 버퍼


def load_model(model_path, device):
    """학습된 모델 로드. 없으면 None."""
    if not os.path.exists(model_path):
        print(f"[WM-Demo] Model not found: {model_path}")
        print("[WM-Demo] Running in linear-extrapolation fallback mode")
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
    val = checkpoint.get('val_metrics', {})
    print(f"[WM-Demo] Model loaded (epoch={epoch}, "
          f"pos_err_final={val.get('pos_err_final', '?'):.2f}NM)")
    return model


def linear_extrapolate(ac, steps=12, dt=10.0):
    """모델 없을 때 선형 외삽 (fallback). Returns (steps, 3) — lat, lon, alt"""
    from config import KNOTS_TO_MPS, M_TO_NM
    trajs = []
    lat, lon, alt = ac.lat, ac.lon, ac.alt_current
    spd_nm_s = ac.ground_speed_kt * KNOTS_TO_MPS * M_TO_NM
    hdg_rad = math.radians(ac.track_true_deg)
    vrate = ac.vertical_rate_ft_min / 60.0

    for t in range(steps):
        dist = spd_nm_s * dt
        lat += dist * math.cos(hdg_rad) / 60.0
        cos_lat = math.cos(math.radians(lat))
        if abs(cos_lat) > 1e-6:
            lon += dist * math.sin(hdg_rad) / (60.0 * cos_lat)
        alt += vrate * dt
        trajs.append((lat, lon, alt))
    return np.array(trajs)


class AircraftHistory:
    """항공기별 상태 이력 관리"""

    def __init__(self):
        self.states = {}   # icao → list of (timestamp, state_10d)

    def update(self, icao, ac, ts):
        state = np.array([
            ac.lat, ac.lon, ac.alt_current,
            ac.ground_speed_kt, ac.track_true_deg,
            ac.vertical_rate_ft_min,
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
        """과거 시퀀스 반환 (steps, STATE_DIM). 부족하면 복제."""
        if icao not in self.states:
            return None
        hist = self.states[icao]
        if len(hist) < 2:
            return None
        states = [s for _, s in hist[-steps:]]
        while len(states) < steps:
            states.insert(0, states[0].copy())
        return np.stack(states[-steps:])

    def get_context(self, target_icao, step_idx=-1):
        """주변 항공기 context 생성"""
        ctx = np.zeros((MAX_NEIGHBORS, CONTEXT_DIM), dtype=np.float32)
        if target_icao not in self.states or not self.states[target_icao]:
            return ctx

        t_state = self.states[target_icao][step_idx][1]
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
        """유효하지 않은 항공기 제거"""
        stale = [k for k in self.states if k not in valid_icaos]
        for k in stale:
            del self.states[k]


def predict_trajectories(model, history, icao, device, future_steps=12, num_mc=MC_SAMPLES_VIS):
    """
    선택한 항공기의 미래 궤적을 Monte Carlo 예측.
    Returns: (num_mc, future_steps, 3) — lat, lon, alt (비정규화)
    """
    past = history.get_past(icao, PAST_STEPS)
    if past is None:
        return None

    # 정규화
    past_norm = (past - NORM_MEAN) / NORM_STD
    past_t = torch.from_numpy(past_norm).unsqueeze(0).to(device)  # (1, K, D)

    # Context
    ctx_list = []
    for i in range(PAST_STEPS):
        ctx_list.append(history.get_context(icao))
    ctx = np.stack(ctx_list)  # (K, MAX_N, CTX_D)
    ctx_t = torch.from_numpy(ctx).unsqueeze(0).to(device)  # (1, K, MAX_N, CTX_D)

    # Predict
    trajs = model.predict(past_t, ctx_t, num_samples=num_mc, future_steps=future_steps)
    # (1, MC, T, STATE_DIM) → (MC, T, 3) — lat, lon, alt만
    result = trajs[0, :, :, :3].cpu().numpy()
    return result


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, device)

    # Simulation (ADS-B replay 또는 live)
    rec_dir = os.path.join(os.path.dirname(__file__), "data", "recordings")
    rec_files = sorted([f for f in os.listdir(rec_dir) if f.endswith(".jsonl")]) if os.path.isdir(rec_dir) else []

    if rec_files:
        replay_src = os.path.join(rec_dir, rec_files[-1])
        print(f"[WM-Demo] Replaying: {replay_src}")
        sim = Simulation(use_gui=False, use_pygame=True,
                         replay_source=replay_src, replay_speed=5, replay_loop=True)
    else:
        print("[WM-Demo] Live ADS-B mode")
        sim = Simulation(use_gui=False, use_pygame=True)

    advisor = SafetyAdvisor(world_model_path=MODEL_PATH if model else None)
    airspace_mgr = AirspaceManager()
    ac_history = AircraftHistory()
    clock = pygame.time.Clock()

    # Flight Plan 사전 스캔 (레코딩 → 이륙 이벤트 추출)
    fp_extractor = FlightPlanExtractor(AIRPORTS)
    if rec_files:
        fp_extractor.scan_jsonl(os.path.join(rec_dir, rec_files[-1]))
        fp_extractor.summary()
    replay_t0 = None  # 리플레이 시작 시 기준 시각

    # Conflict detector
    if model:
        conflict_det = ConflictDetector(model, device=str(device),
                                        num_mc_samples=30, future_steps=12)
    else:
        conflict_det = None

    # ADS-B 대기
    print("[WM-Demo] Waiting for ADS-B data...")
    t0 = time.time()
    while not sim.other_aircraft and time.time() - t0 < 15:
        sim.process_external_aircraft_queue()
        time.sleep(0.3)
    print(f"[WM-Demo] {len(sim.other_aircraft)} aircraft loaded")

    # 상태
    selected_icao = None
    follow_aircraft = False
    pred_trajs = None          # (MC, T, 3) 선택 항공기 예측
    conflict_preds = []        # ConflictPrediction 리스트
    conflict_lines = []        # 시각화용 conflict 라인
    last_pred_time = 0
    last_conflict_time = 0
    future_steps = 12          # 기본 12스텝 (=2분 @10초 간격)
    # 패널 클릭 영역: [(pygame.Rect, target_icao_or_position)]
    panel_clickables = []
    paused = False

    print("=" * 55)
    print("  World Model Safety Advisory Demo")
    print("  Click aircraft to see trajectory prediction")
    print("  [F] Follow  [R] Refresh conflicts  [1-3] Prediction range")
    print("  [SPACE] Pause predictions  [ESC] Quit")
    print("=" * 55)

    while sim.running:
        dt = clock.tick(30) / 1000.0
        now = time.time()

        # 이벤트
        for event in pygame.event.get():
            if sim.map.handle_event(event):
                continue
            if event.type == pygame.QUIT:
                sim.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sim.running = False
                elif event.key == pygame.K_f:
                    follow_aircraft = not follow_aircraft
                elif event.key == pygame.K_r:
                    last_conflict_time = 0  # 즉시 새로고침
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
                # 패널 항목 클릭 감지 (conflict/alert)
                panel_hit = None
                for rect, target in panel_clickables:
                    if rect.collidepoint(mx, my):
                        panel_hit = target
                        break

                if panel_hit:
                    # target = (icao, position) or icao
                    if isinstance(panel_hit, tuple) and len(panel_hit) == 2:
                        hit_icao, hit_pos = panel_hit
                    else:
                        hit_icao, hit_pos = panel_hit, None

                    # icao로 항공기 찾기
                    found_ac = all_tracked.get(hit_icao)
                    if not found_ac:
                        # callsign으로 매칭 시도
                        for k, ac in all_tracked.items():
                            cs = getattr(ac, 'callsign', '')
                            if cs == hit_icao:
                                found_ac = ac
                                hit_icao = k
                                break

                    if found_ac:
                        selected_icao = hit_icao
                        sim.selected_aircraft = found_ac
                        pred_trajs = None
                        last_pred_time = 0
                        follow_aircraft = True
                    elif hit_pos and hit_pos != (0, 0):
                        # 항공기를 못 찾으면 위치로 이동
                        sim.map.center_lat = hit_pos[0]
                        sim.map.center_lon = hit_pos[1]
                else:
                    # 임무 전투기 클릭 감지
                    clicked_pf = None
                    for pf_key, pf_ac in pf_aircraft.items():
                        px, py = sim.map.latlon_to_screen(pf_ac.lat, pf_ac.lon)
                        if abs(mx - px) < 15 and abs(my - py) < 15:
                            clicked_pf = (pf_key, pf_ac)
                            break
                    if clicked_pf:
                        selected_icao = clicked_pf[0]
                        sim.selected_aircraft = clicked_pf[1]
                        pred_trajs = None
                        last_pred_time = 0
                    else:
                        # 일반 항공기 클릭
                        clicked = sim._find_aircraft_at(mx, my)
                        if clicked:
                            selected_icao = getattr(clicked, 'icao24', None) or clicked.callsign
                            sim.selected_aircraft = clicked
                            pred_trajs = None
                            last_pred_time = 0

        # 시뮬레이션 업데이트
        sim.process_external_aircraft_queue()
        sim.map.process_tile_results()
        if now - sim._last_map_tile_check > 0.5:
            sim.map.update_needed_tiles()
            sim._last_map_tile_check = now

        for ac in list(sim.other_aircraft.values()):
            ac.update(dt)
        sim.check_tcas()
        sim.remove_stale_aircraft()
        airspace_mgr.update(dt)

        # 임무 전투기를 가상 Aircraft로 등록 (선택/추적/분석 가능)
        patrol_fighters = airspace_mgr.get_patrol_fighters()
        pf_aircraft = {}
        for pf in patrol_fighters:
            pf_key = f'PF_{pf.callsign}'
            pf_ac = type('PF', (), {
                'lat': pf.lat, 'lon': pf.lon,
                'alt_current': pf.alt, 'callsign': pf.callsign,
                'is_user_controlled': False,
                'ground_speed_kt': pf.spd,
                'track_true_deg': pf.hdg,
                'vertical_rate_ft_min': 0,
                'hdg_current': pf.hdg, 'spd_current': pf.spd,
                'tcas_status': None, 'icao24': pf_key,
                'moa': pf.moa_name,  # 소속 공역 (같은 공역 전투기끼리는 conflict 스킵)
                'last_external_update': now,
                'instruction_active': False, 'instruction_pending': False,
                'color': (255, 180, 50),
            })()
            pf_ac.draw = lambda *a, **k: None  # 별도 렌더링
            pf_ac.update = lambda *a, **k: None
            pf_aircraft[pf_key] = pf_ac

        # 통합 항공기 딕셔너리 (민항기 + 전투기)
        all_tracked = {**sim.other_aircraft, **pf_aircraft}

        # 항공기 이력 업데이트
        for icao, ac in all_tracked.items():
            ac_history.update(icao, ac, now)
        valid_icaos = set(all_tracked.keys())
        ac_history.cleanup(valid_icaos)

        # Conflict detector 상태 업데이트
        if conflict_det:
            for icao, ac in all_tracked.items():
                cs = getattr(ac, 'callsign', '') or icao
                moa = getattr(ac, 'moa', None)
                conflict_det.update_state(icao, {
                    'lat': ac.lat, 'lon': ac.lon,
                    'baro_altitude_ft': ac.alt_current,
                    'ground_speed_kt': ac.ground_speed_kt,
                    'true_track_deg': ac.track_true_deg,
                    'vertical_rate_ft_min': ac.vertical_rate_ft_min,
                }, callsign=cs, moa=moa)

        # Safety Advisor — 감시 모드 (user_aircraft 없음)
        advisor.update([], all_tracked)
        alerts = advisor.get_alerts(min_severity=Severity.CAUTION)

        # ── 예측 갱신 ──
        if not paused and selected_icao and selected_icao in all_tracked:
            if now - last_pred_time >= PREDICTION_INTERVAL:
                last_pred_time = now
                if model:
                    pred_trajs = predict_trajectories(
                        model, ac_history, selected_icao, device,
                        future_steps=future_steps, num_mc=MC_SAMPLES_VIS)
                else:
                    ac = all_tracked[selected_icao]
                    lin = linear_extrapolate(ac, steps=future_steps, dt=10.0)
                    pred_trajs = lin[np.newaxis, :, :]  # (1, T, 3)

        # ── Conflict 분석 ──
        if conflict_det and not paused and now - last_conflict_time >= CONFLICT_SCAN_INTERVAL:
            last_conflict_time = now
            try:
                conflict_preds = conflict_det.detect(dt_sec=10.0, past_steps=PAST_STEPS, min_prob=0.05)
                conflict_lines = []
                for cp in conflict_preds[:10]:
                    ac_a = sim.other_aircraft.get(cp.icao_a)
                    ac_b = sim.other_aircraft.get(cp.icao_b)
                    if ac_a and ac_b:
                        conflict_lines.append((ac_a, ac_b, cp))
            except Exception:
                conflict_preds = []
                conflict_lines = []

        # ── 렌더링 ──
        sim.screen.fill(BLACK)
        panel_clickables = []

        # 카메라 추적
        if follow_aircraft and selected_icao and selected_icao in all_tracked:
            ac = all_tracked[selected_icao]
            sim.map.center_lat = ac.lat
            sim.map.center_lon = ac.lon

        sim.map.draw(sim.screen)

        # KADIZ 경계
        kadiz_pts = [sim.map.latlon_to_screen(v[0], v[1]) for v in KADIZ_POLYGON]
        kadiz_int = [(int(p[0]), int(p[1])) for p in kadiz_pts]
        if len(kadiz_int) >= 3:
            pygame.draw.polygon(sim.screen, (200, 200, 0), kadiz_int, 2)

        # 공역 시각화
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

        # ── Conflict 라인 시각화 ──
        for ac_a, ac_b, cp in conflict_lines:
            ax, ay = sim.map.latlon_to_screen(ac_a.lat, ac_a.lon)
            bx, by = sim.map.latlon_to_screen(ac_b.lat, ac_b.lon)
            ax, ay, bx, by = int(ax), int(ay), int(bx), int(by)

            # 확률에 따라 색상
            if cp.probability >= 0.7:
                line_color = (255, 0, 0)
            elif cp.probability >= 0.4:
                line_color = (255, 165, 0)
            elif cp.probability >= 0.2:
                line_color = (255, 255, 0)
            else:
                line_color = (100, 255, 100)

            thickness = max(1, int(cp.probability * 4))
            pygame.draw.line(sim.screen, line_color, (ax, ay), (bx, by), thickness)

            # 확률 라벨 (중간점)
            mx, my = (ax + bx) // 2, (ay + by) // 2
            prob_text = f"{cp.probability*100:.0f}%"
            s = render_text_with_simple_outline(sim.font, prob_text, line_color, BLACK)
            sim.screen.blit(s, (mx - s.get_width() // 2, my - 20))

            # 시간 라벨
            time_text = f"{cp.expected_time_sec:.0f}s"
            s2 = render_text_with_simple_outline(sim.font, time_text, (200, 200, 200), BLACK)
            sim.screen.blit(s2, (mx - s2.get_width() // 2, my + 2))

        # ── 선택 항공기 예측 궤적 시각화 ──
        if pred_trajs is not None and selected_icao in all_tracked:
            mc, T, _ = pred_trajs.shape
            for s in range(mc):
                pts = []
                for t in range(T):
                    lat, lon = pred_trajs[s, t, 0], pred_trajs[s, t, 1]
                    sx, sy = sim.map.latlon_to_screen(lat, lon)
                    pts.append((int(sx), int(sy)))
                if len(pts) >= 2:
                    # 투명도: 뒤쪽 샘플일수록 연하게
                    alpha = max(40, 200 - s * 8)
                    color = (0, 255, 200, alpha) if mc > 1 else (0, 255, 200)
                    if mc > 1:
                        # 반투명 라인 (Surface trick)
                        for i in range(len(pts) - 1):
                            pygame.draw.line(sim.screen, (0, 255, 200), pts[i], pts[i+1], 1)
                    else:
                        pygame.draw.lines(sim.screen, (0, 255, 200), False, pts, 2)

            # 불확실성 영역: 마지막 스텝의 산포
            if mc > 1:
                final_lats = pred_trajs[:, -1, 0]
                final_lons = pred_trajs[:, -1, 1]
                mean_lat = final_lats.mean()
                mean_lon = final_lons.mean()
                spread_lat = final_lats.std() * 60  # NM
                spread_lon = final_lons.std() * 60 * math.cos(math.radians(mean_lat))
                spread_nm = math.sqrt(spread_lat**2 + spread_lon**2)

                cx, cy = sim.map.latlon_to_screen(mean_lat, mean_lon)
                # 스프레드를 화면 픽셀로 근사
                ref_x, ref_y = sim.map.latlon_to_screen(mean_lat + 1/60, mean_lon)
                px_per_nm = abs(ref_y - cy) if abs(ref_y - cy) > 0 else 1
                radius = max(5, int(spread_nm * px_per_nm))

                uncert_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(uncert_surf, (0, 255, 200, 40), (radius, radius), radius)
                sim.screen.blit(uncert_surf, (int(cx) - radius, int(cy) - radius))

                # 불확실성 수치
                s = render_text_with_simple_outline(
                    sim.font, f"spread: {spread_nm:.1f}NM", (0, 255, 200), BLACK)
                sim.screen.blit(s, (int(cx) + radius + 5, int(cy) - 8))

        # 항공기 렌더링
        for ac in list(sim.other_aircraft.values()):
            ac.draw(sim.screen, sim.map, sim.font)
        sim._highlight_selected()

        # 임무 전투기
        patrol_fighters = airspace_mgr.get_patrol_fighters()
        for pf in patrol_fighters:
            px, py = sim.map.latlon_to_screen(pf.lat, pf.lon)
            px, py = int(px), int(py)
            if -50 <= px < SCREEN_WIDTH + 50 and -50 <= py < SCREEN_HEIGHT + 50:
                draw_hdg = pf.hdg
                sz = 8
                rad = math.radians(draw_hdg)
                pts = [
                    (px + sz * math.sin(rad), py - sz * math.cos(rad)),
                    (px + sz * 0.7 * math.sin(rad + 2.5), py - sz * 0.7 * math.cos(rad + 2.5)),
                    (px + sz * 0.7 * math.sin(rad - 2.5), py - sz * 0.7 * math.cos(rad - 2.5)),
                ]
                pts_int = [(int(p[0]), int(p[1])) for p in pts]
                pygame.draw.polygon(sim.screen, (255, 180, 50), pts_int)
                lbl = render_text_with_simple_outline(
                    sim.font, pf.callsign, (255, 180, 50), BLACK)
                sim.screen.blit(lbl, (px - lbl.get_width() // 2, py + 12))
                info = f"{int(pf.alt)}ft {int(pf.spd)}kt"
                info_s = render_text_with_simple_outline(
                    sim.font, info, (255, 180, 50), BLACK)
                sim.screen.blit(info_s, (px - info_s.get_width() // 2, py + 26))

        font = sim.font

        # ── Flight Plan 패널 (좌측 하단) ──
        replay_ts = getattr(sim._fetcher, 'current_replay_ts', 0)
        if replay_ts and fp_extractor.plans:
            upcoming_deps = fp_extractor.get_upcoming_departures(replay_ts, 600)
            airport_activity = fp_extractor.get_airport_activity(replay_ts, 600)

            if upcoming_deps:
                fp_panel_w = 420
                fp_panel_h = min(len(upcoming_deps[:6]) * 18 + 50, 200)
                fp_x = 5
                fp_y = SCREEN_HEIGHT - fp_panel_h - 10
                fp_bg = pygame.Surface((fp_panel_w, fp_panel_h), pygame.SRCALPHA)
                fp_bg.fill((0, 0, 40, 200))
                sim.screen.blit(fp_bg, (fp_x, fp_y))

                title = f"FLIGHT PLAN — Upcoming DEP ({len(upcoming_deps)})"
                s = render_text_with_simple_outline(font, title, CYAN, BLACK)
                sim.screen.blit(s, (fp_x + 8, fp_y + 5))

                fy = fp_y + 26
                for plan in upcoming_deps[:6]:
                    eta = int(plan.dep_time - replay_ts)
                    dest_str = plan.dest_airport or f"HDG{plan.heading_deg:.0f}"
                    if eta >= 0:
                        eta_str = f"T-{eta:>3d}s"
                    else:
                        eta_str = f" DEP'd "
                    line = (f"{eta_str}  {plan.callsign:8s}  "
                            f"{plan.dep_airport}->{dest_str:4s}  "
                            f"FL{plan.cruise_alt_ft/100:.0f}  "
                            f"{plan.cruise_spd_kt:.0f}kt")
                    col = YELLOW if 0 <= eta < 120 else LIGHT_GRAY
                    s = render_text_with_simple_outline(font, line, col, BLACK)
                    sim.screen.blit(s, (fp_x + 8, fy))
                    fy += 18

            # 공항 마커에 이륙 예정 수 표시
            for ap in AIRPORTS:
                ap_icao = ap['icao']
                if ap_icao in airport_activity:
                    name, cnt, _ = airport_activity[ap_icao]
                    ax, ay = sim.map.latlon_to_screen(ap['lat'], ap['lon'])
                    ax, ay = int(ax), int(ay)
                    if 0 <= ax < SCREEN_WIDTH and 0 <= ay < SCREEN_HEIGHT:
                        # 이륙 예정 공항 강조 원
                        pygame.draw.circle(sim.screen, CYAN, (ax, ay), 12, 2)
                        lbl = f"{ap_icao} DEP:{cnt}"
                        s = render_text_with_simple_outline(font, lbl, CYAN, BLACK)
                        sim.screen.blit(s, (ax + 14, ay - 8))

        # ── HUD ──
        sel_info = ""
        if selected_icao and selected_icao in all_tracked:
            ac = all_tracked[selected_icao]
            cs = getattr(ac, 'callsign', selected_icao)
            is_mil = selected_icao.startswith('PF_')
            tag = "[MIL]" if is_mil else "[CIV]"
            sel_info = (f"Selected: {tag} {cs} | ALT {ac.alt_current:.0f}ft "
                        f"GS {ac.ground_speed_kt:.0f}kt HDG {ac.track_true_deg:.0f}")

        mode = "World Model" if model else "Linear Extrapolation"
        pred_range = {12: "2min", 30: "5min", 60: "10min"}.get(future_steps, f"{future_steps}steps")
        hud_lines = [
            f"Safety Advisory Demo [{mode}] | Traffic: {len(sim.other_aircraft)}",
            f"Prediction: {pred_range} [1-3] | MC samples: {MC_SAMPLES_VIS} | Conflicts: {len(conflict_preds)}",
            sel_info,
            f"Camera: {'Follow [F]' if follow_aircraft else 'Free [F]'} | {'PAUSED' if paused else 'Live'}",
        ]

        hud_h = len(hud_lines) * 20 + 10
        hud_bg = pygame.Surface((600, hud_h), pygame.SRCALPHA)
        hud_bg.fill((0, 0, 0, 180))
        sim.screen.blit(hud_bg, (5, 5))
        hud_y = 10
        for line in hud_lines:
            if line:
                s = render_text_with_simple_outline(font, line, WHITE, BLACK)
                sim.screen.blit(s, (10, hud_y))
            hud_y += 20

        # ── Conflict 패널 (우측 상단) ──
        if conflict_preds:
            panel_w = 380
            panel_x = SCREEN_WIDTH - panel_w - 10
            ay = 10
            panel_h = min(len(conflict_preds[:8]) * 45 + 35, 400)
            bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 200))
            sim.screen.blit(bg, (panel_x, ay))
            s = font.render("CONFLICT PREDICTIONS", True, WHITE)
            sim.screen.blit(s, (panel_x + 10, ay + 5))
            ay += 28

            for cp in conflict_preds[:8]:
                if cp.probability >= 0.7:
                    col = RED
                elif cp.probability >= 0.4:
                    col = ORANGE
                elif cp.probability >= 0.2:
                    col = YELLOW
                else:
                    col = GREEN
                item_y = ay
                line1 = f"{cp.icao_a[:8]} \u2194 {cp.icao_b[:8]}  P={cp.probability*100:.0f}%"
                s = font.render(line1, True, col)
                sim.screen.blit(s, (panel_x + 10, ay))
                ay += 16
                line2 = (f"  {cp.expected_time_sec:.0f}s | "
                         f"min {cp.min_h_dist_nm:.1f}NM/{cp.min_v_dist_ft:.0f}ft | "
                         f"conf {cp.confidence*100:.0f}%")
                s2 = font.render(line2, True, (180, 180, 180))
                sim.screen.blit(s2, (panel_x + 10, ay))
                ay += 22
                # 클릭 영역 등록 (icao_a 기준, position으로 폴백)
                click_rect = pygame.Rect(panel_x, item_y, panel_w, ay - item_y)
                panel_clickables.append((click_rect, (cp.icao_a, cp.position)))

        # ── Safety Alerts 패널 (우측 하단) ──
        if alerts:
            panel_w = 400
            panel_x = SCREEN_WIDTH - panel_w - 10
            panel_h = min(len(alerts[:5]) * 50 + 30, 300)
            ay = SCREEN_HEIGHT - panel_h - 10
            bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 200))
            sim.screen.blit(bg, (panel_x, ay))
            s = font.render("SAFETY ALERTS", True, WHITE)
            sim.screen.blit(s, (panel_x + 10, ay + 5))
            ay += 26
            sev_colors = {Severity.CAUTION: YELLOW, Severity.WARNING: ORANGE, Severity.ALERT: RED}
            for alert in alerts[:5]:
                item_y = ay
                col = sev_colors.get(alert.severity, WHITE)
                s = font.render(f"[{alert.severity.name}] {alert.title}", True, col)
                sim.screen.blit(s, (panel_x + 10, ay))
                ay += 17
                msg = alert.message[:48] + "..." if len(alert.message) > 48 else alert.message
                s = font.render(msg, True, (200, 200, 200))
                sim.screen.blit(s, (panel_x + 15, ay))
                ay += 22
                # 클릭 영역 (첫 번째 관련 항공기로 이동)
                click_rect = pygame.Rect(panel_x, item_y, panel_w, ay - item_y)
                target_icao = alert.aircraft_involved[0] if alert.aircraft_involved else None
                if target_icao:
                    panel_clickables.append((click_rect, (target_icao, alert.position)))

        pygame.display.flip()

    sim.stop()
    print("[WM-Demo] Done.")


if __name__ == "__main__":
    main()

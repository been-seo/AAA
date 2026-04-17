"""
시뮬레이션 엔진
모든 하위 시스템(항공기, 맵, ADS-B, TCAS, GUI)을 통합하는 메인 클래스
"""
import math
import time
import os
import queue
import threading
import multiprocessing

import pygame

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, MAP_CENTER_LAT, MAP_CENTER_LON,
    INITIAL_ZOOM_LEVEL, KNOTS_TO_MPS, M_TO_NM,
    TCAS_SL_TABLE, STALE_AIRCRAFT_CUTOFF,
    BLACK, BLUE, CYAN, RED, YELLOW, GREEN, WHITE,
)
from core.aircraft import Aircraft
from core.map_renderer import TileCache, TileDownloader, Map
from core.adsb_fetcher import ADSBFetcher, ADSBReplayFetcher, ADSBRecorder
from utils import dms_to_decimal, decimal_to_dms, render_text_with_simple_outline


class Simulation:
    def __init__(self, use_gui=False, use_pygame=True,
                 replay_source=None, replay_speed=1.0, replay_loop=True,
                 record_dir=None):
        self.use_pygame = use_pygame and os.environ.get('SDL_VIDEODRIVER') != 'dummy'
        self.pygame_initialized = False

        # Pygame 초기화
        if self.use_pygame:
            try:
                pygame.init()
                pygame.display.set_caption("ATC-AI Simulator")
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font("C:/Windows/Fonts/malgunbd.ttf", 16)  # Bold
                self.pygame_initialized = True
            except pygame.error as e:
                print(f"[Sim] Pygame init failed: {e}")
                self.use_pygame = False
                self.screen = None
                self.clock = pygame.time.Clock()
                self.font = None
        else:
            self.screen = None
            self.clock = pygame.time.Clock()
            self.font = None

        # GUI 프로세스 (별도 PyQt 윈도우)
        self.use_gui_process = use_gui
        self.parent_conn = None
        self.child_conn = None
        self.gui_process = None
        self.awaiting_gui_coords = False
        if use_gui:
            try:
                from gui.control_panel import launch_gui_process
                self.parent_conn, self.child_conn = multiprocessing.Pipe()
                self.gui_process = multiprocessing.Process(
                    target=launch_gui_process, args=(self.child_conn,))
                self.gui_process.daemon = True
                self.gui_process.start()
            except Exception as e:
                print(f"[Sim] GUI process failed: {e}")
                self.use_gui_process = False

        # 타일 맵
        self.tile_cache = TileCache()
        self.tile_req_q = queue.Queue(maxsize=2000)
        self.tile_res_q = queue.Queue(maxsize=2000)
        self._tile_stop = threading.Event()
        self._tile_threads = []
        for _ in range(4):
            t = TileDownloader(self.tile_req_q, self.tile_res_q, self.tile_cache, self._tile_stop)
            t.start()
            self._tile_threads.append(t)

        self.map = Map(
            SCREEN_WIDTH, SCREEN_HEIGHT, MAP_CENTER_LAT, MAP_CENTER_LON,
            INITIAL_ZOOM_LEVEL, self.tile_cache, self.tile_req_q, self.tile_res_q,
        )

        # 항공기
        self.user_aircraft = []
        self.other_aircraft = {}     # icao24 → Aircraft
        self.selected_aircraft = None
        self.tcas_aircraft_list = {"TA": set(), "RA": set()}

        # ADS-B Fetcher (실시간 또는 재생)
        self._adsb_q = queue.Queue(maxsize=100)
        self._adsb_stop = threading.Event()
        self._recorder = None

        if replay_source:
            # 재생 모드: 녹화 파일에서 로드
            self._fetcher = ADSBReplayFetcher(
                self._adsb_q, self._adsb_stop,
                source=replay_source, speed=replay_speed, loop=replay_loop,
            )
        else:
            # 실시간 모드
            self._fetcher = ADSBFetcher(self._adsb_q, self._adsb_stop)
            # 녹화 활성화
            if record_dir:
                self._recorder = ADSBRecorder(record_dir)
                self._fetcher.recorder = self._recorder

        self._fetcher.start()

        self.running = True
        self._last_map_tile_check = time.time()
        self.mouse_latlon_text = ""

    # ── ADS-B 데이터 처리 ──

    def process_external_aircraft_queue(self):
        while True:
            try:
                states = self._adsb_q.get_nowait()
            except queue.Empty:
                break
            for s in states:
                try:
                    icao = s.get("icao24")
                    lat, lon = s.get("lat"), s.get("lon")
                    if not icao or not lat or not lon or s.get("on_ground", 0) == 1:
                        continue
                    if any(ac.icao24 == icao for ac in self.user_aircraft):
                        continue
                    cs = (s.get("callsign") or "").strip() or icao
                    hdg = s.get("true_track_deg", 0) or 0
                    alt = s.get("baro_altitude_ft", 0) or 0
                    spd = s.get("ground_speed_kt", 0) or 0
                    vrate = s.get("vertical_rate_ft_min", 0) or 0

                    if icao not in self.other_aircraft:
                        ac = Aircraft(lat, lon, cs, hdg, alt, spd, is_user_controlled=False)
                        ac.icao24 = icao
                        self.other_aircraft[icao] = ac
                    else:
                        ac = self.other_aircraft[icao]
                        ac.lat, ac.lon, ac.callsign = lat, lon, cs
                        ac.alt_current, ac.spd_current = float(alt), float(spd)

                    ac.ground_speed_kt = float(spd)
                    ac.track_true_deg = float(hdg)
                    ac.vertical_rate_ft_min = float(vrate)
                    ac.squawk = s.get("squawk")
                    ac.last_external_update = time.time()
                except Exception:
                    continue

    # ── TCAS II (DO-185B) ──

    @staticmethod
    def _get_sensitivity_level(own_alt_ft):
        """자기 고도로부터 Sensitivity Level 파라미터를 결정한다."""
        for row in TCAS_SL_TABLE:
            if own_alt_ft <= row[0]:
                return row
        return TCAS_SL_TABLE[-1]

    def _estimate_vrate(self, ac):
        """항공기의 수직 속도(ft/s)를 추정한다."""
        if not ac.is_user_controlled:
            return ac.vertical_rate_ft_min / 60.0
        # 유저 항공기: instruction 기반으로 추정
        if ac.instruction_active and abs(ac.alt_target - ac.alt_current) > 10:
            # alt_change_progress와 direction으로 현재 실제 수직률을 추산
            # _alt_change_direction: +1(상승), -1(하강), 0(수평)
            direction = getattr(ac, '_alt_change_direction', 0)
            if direction != 0:
                # 평균 약 2500fpm으로 근사 (PEAK 7500, 곡선형이므로 평균 ~1/3)
                return direction * 2500.0 / 60.0
        return 0.0

    def _compute_horiz_state(self, ac1, ac2):
        """두 항공기의 수평 상대 상태를 계산한다.
        Returns: (range_nm, range_rate_nm_s, tau_mod for given dmod)
        range_rate < 0 이면 접근 중.
        """
        s1 = ac1.spd_current * KNOTS_TO_MPS * M_TO_NM
        h1 = math.radians(ac1.hdg_current)
        vx1, vy1 = s1 * math.sin(h1), s1 * math.cos(h1)

        s2 = ac2.ground_speed_kt * KNOTS_TO_MPS * M_TO_NM
        h2 = math.radians(ac2.track_true_deg)
        vx2, vy2 = s2 * math.sin(h2), s2 * math.cos(h2)

        avg_lat = math.radians((ac1.lat + ac2.lat) / 2)
        cl = max(abs(math.cos(avg_lat)), 1e-6)
        dx = (ac2.lon - ac1.lon) * 60.0 * cl  # NM
        dy = (ac2.lat - ac1.lat) * 60.0        # NM

        range_nm = math.hypot(dx, dy)

        # 상대 속도
        dvx, dvy = vx2 - vx1, vy2 - vy1
        # range_rate = d(range)/dt = (r⃗ · v⃗_rel) / |r⃗|
        if range_nm < 1e-6:
            range_rate = 0.0
        else:
            range_rate = (dx * dvx + dy * dvy) / range_nm

        return range_nm, range_rate

    def _tau_mod(self, range_nm, range_rate, dmod):
        """Modified tau 계산 (DO-185B 공식).
        range_rate < 0 이면 접근 중.
        Returns: tau_mod (seconds), 접근하지 않으면 inf.
        """
        if range_nm < dmod:
            # 이미 DMOD 내부 — 즉각 위협
            return 0.0
        if range_rate >= 0:
            # 멀어지는 중 → 위협 아님
            return float('inf')
        # τ_mod = (range² - DMOD²) / (range × |closure_rate|)
        closure = -range_rate  # 양수로 변환
        numerator = range_nm * range_nm - dmod * dmod
        if numerator < 0:
            return 0.0
        return numerator / (range_nm * closure)

    def _tau_vert(self, dalt_ft, vert_closure_fps, zthr_ft):
        """수직 tau 계산.
        dalt_ft: |고도차|, vert_closure_fps: 수직 접근률 (ft/s, >0이면 접근)
        Returns: tau_vert (seconds).
        """
        if dalt_ft < zthr_ft:
            # 이미 ZTHR 내부
            return 0.0
        if vert_closure_fps <= 0:
            # 수직으로 벌어지는 중
            return float('inf')
        return (dalt_ft - zthr_ft) / vert_closure_fps

    def check_tcas(self):
        """TCAS II 로직: 수평 τ_mod + 수직 τ_vert 동시 충족 시에만 경고."""
        self.tcas_aircraft_list["RA"].clear()
        self.tcas_aircraft_list["TA"].clear()
        others = list(self.other_aircraft.values())

        for uac in self.user_aircraft:
            uac.tcas_status = None
            uac.tcas_ra_sense = None  # "CLIMB" | "DESCEND" | "MONITOR_VS" | None
            uac.color = BLUE
            is_ra = False

            sl = self._get_sensitivity_level(uac.alt_current)
            # sl: (ceil, tau_TA, dmod_TA, zthr_TA, tau_RA, dmod_RA, zthr_RA, alim_RA)
            tau_ta, dmod_ta, zthr_ta = sl[1], sl[2], sl[3]
            tau_ra, dmod_ra, zthr_ra, alim_ra = sl[4], sl[5], sl[6], sl[7]

            own_vrate = self._estimate_vrate(uac)

            for oac in others:
                if oac.tcas_status != "RA":
                    oac.tcas_status = None
                    oac.color = CYAN

                # 초벌 필터: 고도차 10,000ft 초과 → 확실히 무관
                dalt = abs(uac.alt_current - oac.alt_current)
                if dalt > 10000:
                    continue

                # 수평 상태
                range_nm, range_rate = self._compute_horiz_state(uac, oac)

                # 초벌 필터: 거리 30NM 이상이고 멀어지는 중이면 스킵
                if range_nm > 30 and range_rate >= 0:
                    continue

                # 수직 상태
                intr_vrate = oac.vertical_rate_ft_min / 60.0  # ft/s
                vert_closure = -(intr_vrate - own_vrate)  # 양수 = 수직 접근
                # 부호: own이 +2000fpm 상승, intruder가 -1000fpm 하강
                #   → own_vrate=+33.3, intr_vrate=-16.7
                #   → closure = -((-16.7) - 33.3) = 50.0 (접근) ... 만약 own이 위에 있고 intr가 아래면
                # 실제로는 고도 차이의 변화율이 중요:
                #   d(|dalt|)/dt 가 음수면 접근
                own_alt = uac.alt_current
                intr_alt = oac.alt_current
                # Δalt = own - intruder (부호 있음)
                delta_alt = own_alt - intr_alt
                # d(delta_alt)/dt = own_vrate - intr_vrate
                delta_alt_rate = own_vrate - intr_vrate
                # 수직 접근률: |Δalt|이 줄어드는 비율
                if delta_alt > 0:
                    vert_closure_fps = -delta_alt_rate  # own 위, 줄어들려면 rate<0
                elif delta_alt < 0:
                    vert_closure_fps = delta_alt_rate    # own 아래, 줄어들려면 rate>0
                else:
                    vert_closure_fps = abs(delta_alt_rate)

                # ── RA 판정 ──
                if tau_ra is not None:
                    h_tau = self._tau_mod(range_nm, range_rate, dmod_ra)
                    v_tau = self._tau_vert(dalt, vert_closure_fps, zthr_ra)

                    # 수평 조건: τ_mod ≤ τ_RA 또는 range < DMOD
                    h_threat = (h_tau <= tau_ra)
                    # 수직 조건: τ_vert ≤ τ_RA 또는 |Δalt| < ZTHR
                    v_threat = (v_tau <= tau_ra)

                    if h_threat and v_threat:
                        uac.tcas_status = "RA"
                        oac.tcas_status = "RA"
                        oac.color = RED
                        self.tcas_aircraft_list["RA"].add(oac.callsign)
                        is_ra = True

                        # RA sense 결정: ALIM 확보 방향으로
                        if own_alt >= intr_alt:
                            # 위에 있으면 상승 (분리 확대)
                            uac.tcas_ra_sense = "CLIMB"
                        else:
                            uac.tcas_ra_sense = "DESCEND"
                        continue

                # ── TA 판정 ──
                if not is_ra:
                    h_tau = self._tau_mod(range_nm, range_rate, dmod_ta)
                    v_tau = self._tau_vert(dalt, vert_closure_fps, zthr_ta)

                    h_threat = (h_tau <= tau_ta)
                    v_threat = (v_tau <= tau_ta)

                    if h_threat and v_threat:
                        if uac.tcas_status is None:
                            uac.tcas_status = "TA"
                        oac.tcas_status = "TA"
                        oac.color = YELLOW
                        self.tcas_aircraft_list["TA"].add(oac.callsign)

            if is_ra:
                uac.color = RED
            elif uac.tcas_status == "TA":
                uac.color = YELLOW

    # ── 오래된 항공기 제거 ──

    def remove_stale_aircraft(self, cutoff=STALE_AIRCRAFT_CUTOFF):
        now = time.time()
        stale = [k for k, ac in self.other_aircraft.items() if ac.last_external_update < now - cutoff]
        for k in stale:
            if self.selected_aircraft and getattr(self.selected_aircraft, 'icao24', None) == k:
                self.selected_aircraft = None
            del self.other_aircraft[k]

    # ── 통합 시뮬 스텝 ──

    def run_step(self, delta_time):
        """외부에서 호출하는 1 프레임 업데이트"""
        if self.use_gui_process:
            self._check_gui_messages()
        if self.use_pygame:
            self._handle_input()
        self.process_external_aircraft_queue()
        if self.use_pygame:
            self.map.process_tile_results()
            now = time.time()
            if now - self._last_map_tile_check > 0.5:
                self.map.update_needed_tiles()
                self._last_map_tile_check = now
        for ac in self.user_aircraft:
            ac.update(delta_time)
        for ac in list(self.other_aircraft.values()):
            ac.update(delta_time)
        self.check_tcas()
        self.remove_stale_aircraft()
        if self.use_pygame:
            self._render()

    # ── Pygame 입력 ──

    def _handle_input(self):
        if not self.use_pygame:
            return
        for event in pygame.event.get():
            if self.awaiting_gui_coords and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                lat, lon = self.map.screen_to_latlon(*event.pos)
                self.parent_conn.send({"type": "coords", "data": {
                    "lat": decimal_to_dms(lat, False, 0), "lon": decimal_to_dms(lon, False, 1)}})
                self.awaiting_gui_coords = False
                continue
            if self.map.handle_event(event):
                continue
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_n and self.use_gui_process:
                    self.parent_conn.send({"type": "action", "action": "create_aircraft"})
                elif event.key == pygame.K_i and self.use_gui_process and self.selected_aircraft and self.selected_aircraft.is_user_controlled:
                    self.parent_conn.send({"type": "action", "action": "instruction", "aircraft_data": {
                        "callsign": self.selected_aircraft.callsign,
                        "alt": self.selected_aircraft.alt_target,
                        "spd": self.selected_aircraft.spd_target,
                        "hdg": self.selected_aircraft.hdg_target,
                    }})
                elif event.key == pygame.K_c:
                    self.selected_aircraft = None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.selected_aircraft = self._find_aircraft_at(*event.pos)
            elif event.type == pygame.MOUSEMOTION and self.font:
                lat, lon = self.map.screen_to_latlon(*event.pos)
                self.mouse_latlon_text = f"Lat: {decimal_to_dms(lat, True, 0)}, Lon: {decimal_to_dms(lon, True, 1)}"

    def _find_aircraft_at(self, px, py):
        for ac in reversed(self.user_aircraft):
            if ac.contains_point(self.map, px, py):
                return ac
        for ac in reversed(list(self.other_aircraft.values())):
            if ac.contains_point(self.map, px, py):
                return ac
        return None

    # ── GUI 프로세스 통신 ──

    def _check_gui_messages(self):
        if not self.parent_conn:
            return
        try:
            while self.parent_conn.poll():
                msg = self.parent_conn.recv()
                if msg == "await_coords":
                    self.awaiting_gui_coords = True
                elif isinstance(msg, dict):
                    if msg["type"] == "create_aircraft":
                        ac = Aircraft(
                            dms_to_decimal(msg["lat"], 0), dms_to_decimal(msg["lon"], 1),
                            msg["callsign"], float(msg["hdg"]), float(msg["alt"]), float(msg["spd"]),
                        )
                        self.user_aircraft.append(ac)
                        self.selected_aircraft = ac
                    elif msg["type"] == "instruction" and self.selected_aircraft and self.selected_aircraft.is_user_controlled:
                        self.selected_aircraft.apply_instruction(
                            float(msg["hdg"]), float(msg["alt"]), float(msg["spd"]),
                        )
                        if msg.get("quick_alt"):
                            self.selected_aircraft.hdg_mode = "quick_alt"
        except (EOFError, BrokenPipeError):
            pass

    # ── 렌더링 ──

    def _render(self):
        if not self.pygame_initialized or not self.screen:
            return
        try:
            self.screen.fill(BLACK)
            self.map.draw(self.screen)
            for ac in list(self.other_aircraft.values()):
                ac.draw(self.screen, self.map, self.font)
            for ac in self.user_aircraft:
                ac.draw(self.screen, self.map, self.font)
            self._highlight_selected()
            label = render_text_with_simple_outline(self.font, self.mouse_latlon_text, BLUE, WHITE)
            self.screen.blit(label, (10, 10))
            pygame.display.flip()
        except pygame.error:
            pass

    def _highlight_selected(self):
        ac = self.selected_aircraft
        if not ac:
            return
        valid = (ac in self.user_aircraft) if ac.is_user_controlled else (getattr(ac, 'icao24', None) in self.other_aircraft)
        if not valid:
            self.selected_aircraft = None
            return
        sx, sy = self.map.latlon_to_screen(ac.lat, ac.lon)
        zs = self.map.get_current_zoom_level() / INITIAL_ZOOM_LEVEL
        r = int(max(8, min(40, ac._icon_size_base * zs)) * 2.0)
        if -r - 10 < sx < SCREEN_WIDTH + r + 10 and -r - 10 < sy < SCREEN_HEIGHT + r + 10:
            pygame.draw.circle(self.screen, GREEN, (int(sx), int(sy)), r, 3)

    # ── 종료 ──

    def stop(self):
        self.running = False
        print("[Sim] Shutting down...")
        self._adsb_stop.set()
        self._tile_stop.set()
        if self._recorder:
            self._recorder.close()
        if self.use_gui_process and self.parent_conn:
            try:
                self.parent_conn.send({"type": "action", "action": "shutdown"})
            except Exception:
                pass
        if self.gui_process:
            self.gui_process.join(timeout=2)
            if self.gui_process.is_alive():
                self.gui_process.terminate()
                self.gui_process.join(timeout=2)
            if self.gui_process.is_alive():
                self.gui_process.kill()
                self.gui_process.join(timeout=1)
        if self.parent_conn:
            self.parent_conn.close()
        if self.child_conn:
            self.child_conn.close()
        self._fetcher.join(timeout=2)
        for t in self._tile_threads:
            t.join(timeout=2)
        if self.use_pygame and self.pygame_initialized:
            pygame.quit()
        print("[Sim] Shutdown complete.")

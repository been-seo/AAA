"""
항공기 물리 모델
- 사용자 제어 항공기: hdg/alt/spd 지시 → 곡선형 가감속
- 외부 항공기 (ADS-B): 선형 외삽 위치 업데이트
"""
import math
import pygame

from config import (
    KNOTS_TO_MPS, M_TO_NM,
    HDG_RATE_NORMAL, SPD_RATE_NORMAL,
    PEAK_ALT_RATE_NORMAL_FPS, ALT_RATE_QUICK, INSTRUCTION_DELAY,
    BLUE, CYAN, RED, YELLOW, BLACK, WHITE,
    SCREEN_WIDTH, SCREEN_HEIGHT, INITIAL_ZOOM_LEVEL,
)
from utils import alt_normal_rate_factor, render_text_with_simple_outline


# ── 연료 모델 (F-16급 단발 전투기 기준) ──
FUEL_CAPACITY_LBS = 7000.0      # 내부 연료 (lbs)
FUEL_BASE_FLOW_LBH = 4800.0     # 기준 연료유량 (lbs/hr, FL250 / 420kts 순항)
FUEL_BINGO_RESERVE_MIN = 20.0   # 비상 예비 연료 (분)


def fuel_flow_rate(alt_ft, spd_kts, turn_rate_deg_s):
    """
    고도·속도·기동에 따른 연료유량 (lbs/hr)

    모델:
      flow = base × alt_factor × spd_factor × maneuver_factor

    - alt_factor: 저고도일수록 공기밀도 높아 엔진 부하 ↑
      SL(0ft)=1.6, FL100=1.3, FL250=1.0, FL350=0.85, FL450=0.80
    - spd_factor: 항력 ∝ V², 기준 420kts=1.0
      300kts=0.60, 420kts=1.0, 550kts=1.65, 600kts=2.0
    - maneuver_factor: 선회 시 G-load 증가 → 유도항력 ↑
      직선=1.0, 1.5°/s=1.05, 3°/s=1.15, 6°/s=1.40
    """
    # 고도 팩터: 지수 감쇠 (해면=1.6, 고고도=0.8)
    alt_factor = 0.80 + 0.80 * math.exp(-alt_ft / 18000.0)

    # 속도 팩터: (V/V_ref)^2, 최소 0.4
    spd_ref = 420.0
    spd_ratio = max(spd_kts, 150.0) / spd_ref
    spd_factor = max(0.4, spd_ratio * spd_ratio)

    # 기동 팩터: 선회율 기반 (bank angle → load factor → drag)
    turn_abs = abs(turn_rate_deg_s)
    if turn_abs < 0.5:
        maneuver_factor = 1.0
    else:
        # 대략적: 3°/s 표준율 선회 ≈ 1.15, 6°/s ≈ 1.40
        maneuver_factor = 1.0 + turn_abs * 0.065

    return FUEL_BASE_FLOW_LBH * alt_factor * spd_factor * maneuver_factor


class Aircraft:
    def __init__(self, lat, lon, callsign, hdg=0, alt=0, spd=0, is_user_controlled=True):
        self.lat = lat
        self.lon = lon
        self.callsign = callsign

        self.hdg_current = hdg % 360
        self.alt_current = float(alt)
        self.spd_current = float(spd)

        self.hdg_target = hdg % 360
        self.alt_target = float(alt)
        self.spd_target = float(spd)

        self.hdg_mode = "normal"
        self.alt_mode = "normal"
        self.spd_mode = "normal"

        self.is_user_controlled = is_user_controlled
        self.color = (100, 255, 100) if is_user_controlled else CYAN  # 밝은 녹색 / 시안

        self.tcas_status = None       # None | "TA" | "RA"
        self.tcas_ra_sense = None     # None | "CLIMB" | "DESCEND" | "MONITOR_VS"
        self.is_blinking = False
        self._blink_timer = 0.0

        self.instruction_pending = False
        self.delay_timer = 0.0
        self.instruction_active = False
        self.alt_start = float(alt)
        self.alt_change_progress = 0.0
        self._alt_change_direction = 0

        self._base_icon_surface = None
        self._icon_size_base = 15

        # 군/민 구분: user_controlled = 군, ADS-B 외부 = 민
        self.is_military = is_user_controlled

        # 연료 (user_controlled 항공기만 시뮬레이션)
        self.fuel_lbs = FUEL_CAPACITY_LBS if is_user_controlled else 0.0
        self.fuel_flow_lbh = 0.0        # 현재 연료유량 (lbs/hr)
        self.fuel_used_lbs = 0.0        # 누적 소모량
        self._prev_hdg = hdg % 360      # 선회율 계산용

        # ADS-B 외부 데이터
        self.last_external_update = 0.0
        self.ground_speed_kt = float(spd)
        self.vertical_rate_ft_min = 0
        self.track_true_deg = hdg % 360
        self.squawk = None
        self.icao24 = None

    # ── 물리 업데이트 ──

    def update(self, delta_time):
        if delta_time <= 0:
            return
        if self.is_blinking:
            self._blink_timer += delta_time

        if not self.is_user_controlled:
            self._update_external(delta_time)
        else:
            self._update_controlled(delta_time)
            self._move(delta_time)
            self._update_fuel(delta_time)

    def _update_external(self, dt):
        """ADS-B 항공기: 선형 외삽"""
        hdg_rad = math.radians(self.track_true_deg)
        speed_nm_s = self.ground_speed_kt * KNOTS_TO_MPS * M_TO_NM
        dist = speed_nm_s * dt
        self.lat += dist * math.cos(hdg_rad) / 60.0
        cos_lat = math.cos(math.radians(self.lat))
        if abs(cos_lat) >= 1e-6:
            self.lon += dist * math.sin(hdg_rad) / (60.0 * cos_lat)
        self.lon = (self.lon + 180) % 360 - 180
        self.alt_current += (self.vertical_rate_ft_min / 60.0) * dt

    def _update_controlled(self, dt):
        """사용자/RL 항공기: 지시 추종"""
        if self.instruction_pending:
            self.delay_timer -= dt
            if self.delay_timer <= 0:
                self.instruction_pending = False
                self.instruction_active = True
                self.alt_start = self.alt_current
                self.alt_change_progress = 0.0
                diff = self.alt_target - self.alt_start
                self._alt_change_direction = math.copysign(1, diff) if abs(diff) > 1 else 0

        if not self.instruction_active:
            return

        # Heading
        target = self.hdg_target % 360
        diff = (target - self.hdg_current + 180) % 360 - 180
        if abs(diff) > 0.1:
            change = min(HDG_RATE_NORMAL * dt, abs(diff))
            self.hdg_current = (self.hdg_current + math.copysign(change, diff)) % 360
        else:
            self.hdg_current = target

        # Speed
        diff_spd = self.spd_target - self.spd_current
        if abs(diff_spd) > 0.1:
            change = min(SPD_RATE_NORMAL * dt, abs(diff_spd))
            self.spd_current = max(0.0, self.spd_current + math.copysign(change, diff_spd))
        else:
            self.spd_current = self.spd_target

        # Altitude (곡선형)
        if abs(self.alt_current - self.alt_target) > 1:
            total = self.alt_target - self.alt_start
            if abs(total) < 1:
                self.alt_change_progress = 1.0
                self._alt_change_direction = 0
            else:
                self.alt_change_progress = max(0.0, min(1.0, (self.alt_current - self.alt_start) / total))

            if self.alt_change_progress < 1.0:
                factor = alt_normal_rate_factor(self.alt_change_progress)
                rate = ALT_RATE_QUICK if self.hdg_mode == "quick_alt" else PEAK_ALT_RATE_NORMAL_FPS
                delta_alt = rate * factor * dt * self._alt_change_direction
                if (self._alt_change_direction > 0 and self.alt_current + delta_alt >= self.alt_target) or \
                   (self._alt_change_direction < 0 and self.alt_current + delta_alt <= self.alt_target):
                    self.alt_current = self.alt_target
                    self.alt_change_progress = 1.0
                    self._alt_change_direction = 0
                else:
                    self.alt_current += delta_alt
        else:
            self.alt_current = self.alt_target
            self.alt_change_progress = 1.0
            self._alt_change_direction = 0
            if self.hdg_mode == "quick_alt":
                self.hdg_mode = "normal"

        # 완료 체크
        if (abs(self.alt_current - self.alt_target) < 1
                and abs(self.hdg_current - (self.hdg_target % 360)) < 0.1
                and abs(self.spd_current - self.spd_target) < 0.1):
            self.instruction_active = False
            self.instruction_pending = False

    def _move(self, dt):
        """위치 이동 (hdg_current는 True 기준)"""
        speed_nm_s = self.spd_current * KNOTS_TO_MPS * M_TO_NM
        dist = speed_nm_s * dt
        hdg_rad = math.radians(self.hdg_current)
        self.lat += dist * math.cos(hdg_rad) / 60.0
        cos_lat = math.cos(math.radians(self.lat))
        if abs(cos_lat) >= 1e-6:
            self.lon += dist * math.sin(hdg_rad) / (60.0 * cos_lat)
        self.lon = (self.lon + 180) % 360 - 180
        # ADS-B 호환 필드 동기화
        self.track_true_deg = self.hdg_current
        self.ground_speed_kt = self.spd_current

    def _update_fuel(self, dt):
        """연료 소모 계산 (고도·속도·기동 함수)"""
        if self.fuel_lbs <= 0:
            return

        # 선회율 산출 (현재 vs 이전 헤딩)
        hdg_diff = (self.hdg_current - self._prev_hdg + 180) % 360 - 180
        turn_rate = abs(hdg_diff / dt) if dt > 0.01 else 0.0
        self._prev_hdg = self.hdg_current

        # 연료유량 계산
        self.fuel_flow_lbh = fuel_flow_rate(self.alt_current, self.spd_current, turn_rate)

        # 소모
        consumed = self.fuel_flow_lbh * (dt / 3600.0)
        self.fuel_lbs = max(0.0, self.fuel_lbs - consumed)
        self.fuel_used_lbs += consumed

    @property
    def fuel_pct(self):
        """연료 잔량 (%)"""
        return (self.fuel_lbs / FUEL_CAPACITY_LBS) * 100.0

    @property
    def fuel_endurance_min(self):
        """현재 유량 기준 잔여 체공 시간 (분)"""
        if self.fuel_flow_lbh < 1.0:
            return 999.0
        return (self.fuel_lbs / self.fuel_flow_lbh) * 60.0

    @property
    def fuel_bingo(self):
        """Bingo fuel 도달 여부"""
        return self.fuel_endurance_min <= FUEL_BINGO_RESERVE_MIN

    # ── 지시 적용 ──

    def apply_instruction(self, hdg, alt, spd, hdg_mode="normal", alt_mode="normal", spd_mode="normal"):
        self.hdg_target = hdg % 360
        self.alt_target = float(alt)
        self.spd_target = float(spd)
        self.hdg_mode = hdg_mode
        self.alt_mode = alt_mode
        self.spd_mode = spd_mode
        if (abs((self.hdg_current - self.hdg_target + 180) % 360 - 180) > 0.1
                or abs(self.alt_current - self.alt_target) > 1
                or abs(self.spd_current - self.spd_target) > 0.1):
            self.instruction_pending = True
            self.delay_timer = INSTRUCTION_DELAY
            self.instruction_active = False
        else:
            self.instruction_pending = False
            self.instruction_active = False

    # ── Pygame 렌더링 ──

    def _create_base_icon(self, color):
        sz = self._icon_size_base
        surf_size = sz * 4
        surface = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        cx, cy = surf_size // 2, surf_size // 2
        points = [(cx, cy - sz), (cx - sz * 0.7, cy + sz * 0.8), (cx + sz * 0.7, cy + sz * 0.8)]
        pygame.draw.polygon(surface, color, points)
        return surface

    def draw(self, screen, map_obj, font):
        should_draw = True
        if self.is_blinking and (self._blink_timer % 1.0) < 0.5:
            should_draw = False
        if not should_draw:
            return

        sx, sy = map_obj.latlon_to_screen(self.lat, self.lon)
        sx, sy = int(round(sx)), int(round(sy))
        pad = 50
        if not (-pad <= sx < SCREEN_WIDTH + pad and -pad <= sy < SCREEN_HEIGHT + pad):
            return

        zoom_scale = map_obj.get_current_zoom_level() / INITIAL_ZOOM_LEVEL
        icon_sz = max(8, min(40, self._icon_size_base * zoom_scale))

        draw_hdg = self.hdg_current if self.is_user_controlled else self.track_true_deg

        color = self.color
        if self.tcas_status == "RA":
            color = RED
        elif self.tcas_status == "TA":
            color = YELLOW

        # 외곽선: 어두운 색엔 흰 외곽, 밝은 색엔 검은 외곽
        brightness = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
        outline = BLACK if brightness > 128 else WHITE

        icon = self._create_base_icon(color)
        scaled = pygame.transform.scale(icon, (int(icon_sz * 2), int(icon_sz * 2)))
        rotated = pygame.transform.rotate(scaled, -draw_hdg)
        screen.blit(rotated, rotated.get_rect(center=(sx, sy)))

        # 콜사인
        label = render_text_with_simple_outline(font, self.callsign, color, outline)
        ty = int(icon_sz * 1.5) + 5
        screen.blit(label, label.get_rect(center=(sx, sy + ty)))

        # 고도/속도/연료
        iy = ty + int(font.get_height() * 0.7)
        if self.is_user_controlled:
            l1 = f"{int(self.alt_current)}ft"
            l2 = f"{int(self.spd_current)}kts"
            if self.instruction_pending or self.instruction_active:
                l1 += f" ({int(self.alt_target)})"
                l2 += f" ({int(self.spd_target)})"
            # 연료 표시
            fuel_color = color
            if self.fuel_bingo:
                fuel_color = RED
            elif self.fuel_pct < 40:
                fuel_color = YELLOW
            l3 = f"FUEL {self.fuel_pct:.0f}% {self.fuel_endurance_min:.0f}min"
        else:
            l1 = f"{int(self.alt_current)}ft {self.vertical_rate_ft_min:+.0f}"
            l2 = f"{int(self.ground_speed_kt)}kts"
            l3 = None
        lines = [l1, l2] + ([l3] if l3 else [])
        line_colors = [color, color] + ([fuel_color] if l3 else [])
        for i, (line, lc) in enumerate(zip(lines, line_colors)):
            s = render_text_with_simple_outline(font, line, lc, outline)
            screen.blit(s, s.get_rect(center=(sx, sy + iy + int(font.get_height() * 0.9 * i))))

    def contains_point(self, map_obj, px, py):
        sx, sy = map_obj.latlon_to_screen(self.lat, self.lon)
        zoom_scale = map_obj.get_current_zoom_level() / INITIAL_ZOOM_LEVEL
        r = max(8, min(40, self._icon_size_base * zoom_scale)) * 2.0
        return (px - sx) ** 2 + (py - sy) ** 2 <= r ** 2

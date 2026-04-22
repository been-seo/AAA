"""
ATC-AI 통합 진입점 (Safety Advisor + WM + Critic)

실행 모드:
  python main.py              → 실시간 시뮬레이터 + AI Safety Advisor
  python main.py --replay DIR → 녹화 재생
  python main.py --record     → 실시간 수신 + ADS-B 자동 녹화
  python main.py --headless   → GUI 없이 실행

학습은 전용 스크립트:
  python -m ai.world_model.trainer      (World Model)
  python train_dreamer.py               (Dreamer Critic/Actor)
  python record_adsb.py                 (데이터 녹화)
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, YELLOW, ORANGE, RED, GREEN
from core.simulation import Simulation
from ai.safety_advisor import SafetyAdvisor, Severity
from utils import render_text_with_simple_outline


_WM_PATH = "models/world_model/best_model.pt"
_DREAMER_PATH = "models/dreamer/best.pt"


def run_simulator(use_gui=True, replay_source=None, replay_speed=1.0,
                  record_dir=None):
    """수동 관제 시뮬레이터 + AI Safety Advisor"""
    sim = Simulation(
        use_gui=use_gui, use_pygame=True,
        replay_source=replay_source, replay_speed=replay_speed,
        record_dir=record_dir,
    )
    # AI 구성: 파일이 있으면 로드, 없으면 룰 기반만 사용
    wm_path = _WM_PATH if os.path.exists(_WM_PATH) else None
    dreamer_path = _DREAMER_PATH if os.path.exists(_DREAMER_PATH) else None
    advisor = SafetyAdvisor(
        world_model_path=wm_path,
        dreamer_path=dreamer_path,
    )
    print(f"[main] WM={'ON' if wm_path else 'OFF'} "
          f"Dreamer={'ON' if dreamer_path else 'OFF'}")

    clock = pygame.time.Clock()
    print("=" * 60)
    print("  ATC-AI Simulator")
    print("  [N] 항공기 생성  [I] 지시  [C] 선택 해제  [ESC] 종료")
    print("=" * 60)

    while sim.running:
        dt = clock.tick(30) / 1000.0
        sim.run_step(dt)

        # Safety Advisor: user + all aircraft
        all_tracked = {**sim.other_aircraft}
        for uac in sim.user_aircraft:
            uid = getattr(uac, 'icao24', None) or f'USER_{uac.callsign}'
            all_tracked[uid] = uac
        advisor.update(sim.user_aircraft, all_tracked)

        if sim.use_pygame and sim.screen and sim.font:
            _draw_safety_overlay(sim.screen, sim.font, advisor)
            pygame.display.flip()

    sim.stop()


def _draw_safety_overlay(screen, font, advisor):
    alerts = advisor.get_alerts(min_severity=Severity.CAUTION)
    if not alerts:
        return

    panel_w = 450
    panel_x = SCREEN_WIDTH - panel_w - 10
    panel_y = 10
    line_h = 20
    header_h = 30

    panel_h = header_h + len(alerts) * (line_h * 3 + 10) + 10
    bg = pygame.Surface((panel_w, min(panel_h, SCREEN_HEIGHT - 20)), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 180))
    screen.blit(bg, (panel_x, panel_y))

    title = font.render("SAFETY ADVISOR", True, WHITE)
    screen.blit(title, (panel_x + 10, panel_y + 5))
    y = panel_y + header_h

    SEVERITY_COLORS = {
        Severity.INFO: GREEN,
        Severity.CAUTION: YELLOW,
        Severity.WARNING: ORANGE,
        Severity.ALERT: RED,
    }

    for alert in alerts[:8]:
        if y + line_h * 3 > SCREEN_HEIGHT - 10:
            break
        color = SEVERITY_COLORS.get(alert.severity, WHITE)
        sev_name = alert.severity.name
        header_text = f"[{sev_name}] {alert.title}"
        s = font.render(header_text, True, color)
        screen.blit(s, (panel_x + 10, y))
        y += line_h
        msg = alert.message[:60] + "..." if len(alert.message) > 60 else alert.message
        s = font.render(msg, True, (200, 200, 200))
        screen.blit(s, (panel_x + 15, y))
        y += line_h
        if alert.recommendation:
            rec = alert.recommendation[:55] + "..." if len(alert.recommendation) > 55 else alert.recommendation
            s = font.render(f"> {rec}", True, (150, 255, 150))
            screen.blit(s, (panel_x + 15, y))
        y += line_h + 5
        if alert.time_to_event_sec > 0:
            time_str = f"T-{alert.time_to_event_sec:.0f}s"
            ts = font.render(time_str, True, color)
            screen.blit(ts, (panel_x + panel_w - 70, y - line_h * 3 - 5))


def main():
    parser = argparse.ArgumentParser(
        description="ATC-AI Simulator (with Safety Advisor)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py                              실시간 시뮬레이터
  python main.py --record                     실시간 + ADS-B 녹화
  python main.py --replay data/recordings     녹화 재생
  python main.py --replay data/rec.jsonl --speed 10   10배속 재생

학습은 전용 스크립트로:
  python -m ai.world_model.trainer        World Model 학습
  python train_dreamer.py                 Dreamer (Critic/Actor) 학습
  python record_adsb.py                   ADS-B 녹화
""")
    parser.add_argument("--no-gui-panel", action="store_true",
                        help="PyQt GUI 패널 비활성화")
    parser.add_argument("--replay", type=str, default=None,
                        help="녹화 파일/디렉토리 경로")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="재생 배속 (0=최대, 1=실시간)")
    parser.add_argument("--record", action="store_true",
                        help="실시간 수신 중 ADS-B 자동 녹화")
    parser.add_argument("--record-dir", type=str, default=None,
                        help="녹화 저장 디렉토리")
    args = parser.parse_args()

    record_dir = None
    if args.record:
        record_dir = args.record_dir or os.path.join(
            os.path.dirname(__file__), "data", "recordings")
    run_simulator(
        use_gui=not args.no_gui_panel,
        replay_source=args.replay,
        replay_speed=args.speed,
        record_dir=record_dir,
    )


if __name__ == "__main__":
    main()

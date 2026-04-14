"""
ATC-AI 통합 진입점

실행 모드:
  python main.py              → 수동 관제 시뮬레이터 (Pygame + GUI + Safety Advisor)
  python main.py --train      → RL 에이전트 학습
  python main.py --test       → 학습된 에이전트 테스트 (시각화)
  python main.py --headless   → GUI 없이 시뮬레이터 실행 (ADS-B 모니터링)
"""
import sys
import os
import time
import argparse

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, YELLOW, ORANGE, RED, GREEN
from core.simulation import Simulation
from ai.safety_advisor import SafetyAdvisor, Severity
from utils import render_text_with_simple_outline


def run_simulator(use_gui=True, replay_source=None, replay_speed=1.0,
                  record_dir=None):
    """수동 관제 시뮬레이터 실행 (Safety Advisor 포함)"""
    sim = Simulation(
        use_gui=use_gui, use_pygame=True,
        replay_source=replay_source, replay_speed=replay_speed,
        record_dir=record_dir,
    )
    advisor = SafetyAdvisor()
    clock = pygame.time.Clock()

    print("=" * 60)
    print("  ATC-AI Simulator")
    print("  [N] 항공기 생성  [I] 지시  [C] 선택 해제  [ESC] 종료")
    print("=" * 60)

    while sim.running:
        dt = clock.tick(30) / 1000.0
        sim.run_step(dt)

        # Safety Advisor 업데이트
        advisor.update(sim.user_aircraft, sim.other_aircraft)

        # 경고 오버레이 렌더링
        if sim.use_pygame and sim.screen and sim.font:
            _draw_safety_overlay(sim.screen, sim.font, advisor)
            pygame.display.flip()

    sim.stop()


def _draw_safety_overlay(screen, font, advisor):
    """Safety Advisor 경고를 화면 우측에 오버레이"""
    alerts = advisor.get_alerts(min_severity=Severity.CAUTION)
    if not alerts:
        return

    # 경고 패널 배경
    panel_w = 450
    panel_x = SCREEN_WIDTH - panel_w - 10
    panel_y = 10
    line_h = 20
    header_h = 30

    # 반투명 배경
    panel_h = header_h + len(alerts) * (line_h * 3 + 10) + 10
    bg = pygame.Surface((panel_w, min(panel_h, SCREEN_HEIGHT - 20)), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 180))
    screen.blit(bg, (panel_x, panel_y))

    # 헤더
    title = font.render("SAFETY ADVISOR", True, WHITE)
    screen.blit(title, (panel_x + 10, panel_y + 5))
    y = panel_y + header_h

    SEVERITY_COLORS = {
        Severity.INFO: GREEN,
        Severity.CAUTION: YELLOW,
        Severity.WARNING: ORANGE,
        Severity.ALERT: RED,
    }

    for alert in alerts[:8]:  # 최대 8개 표시
        if y + line_h * 3 > SCREEN_HEIGHT - 10:
            break

        color = SEVERITY_COLORS.get(alert.severity, WHITE)
        sev_name = alert.severity.name

        # 심각도 + 제목
        header_text = f"[{sev_name}] {alert.title}"
        s = font.render(header_text, True, color)
        screen.blit(s, (panel_x + 10, y))
        y += line_h

        # 메시지 (축약)
        msg = alert.message[:60] + "..." if len(alert.message) > 60 else alert.message
        s = font.render(msg, True, (200, 200, 200))
        screen.blit(s, (panel_x + 15, y))
        y += line_h

        # 권고사항
        if alert.recommendation:
            rec = alert.recommendation[:55] + "..." if len(alert.recommendation) > 55 else alert.recommendation
            s = font.render(f"> {rec}", True, (150, 255, 150))
            screen.blit(s, (panel_x + 15, y))
        y += line_h + 5

        # 시간
        if alert.time_to_event_sec > 0:
            time_str = f"T-{alert.time_to_event_sec:.0f}s"
            ts = font.render(time_str, True, color)
            screen.blit(ts, (panel_x + panel_w - 70, y - line_h * 3 - 5))


def run_train():
    """RL 에이전트 학습"""
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    except ImportError:
        print("stable-baselines3가 필요합니다: pip install stable-baselines3")
        return

    from ai.atc_env import GuidedAirControlEnv
    from config import (
        TRAIN_TOTAL_TIMESTEPS, TRAIN_BUFFER_SIZE, TRAIN_LEARNING_STARTS,
        TRAIN_BATCH_SIZE, TRAIN_GAMMA, TRAIN_LEARNING_RATE, TRAIN_CHECKPOINT_FREQ,
    )

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = GuidedAirControlEnv(render_mode='none')

    # 기존 모델 이어서 학습
    latest_model = os.path.join(model_dir, "atc_dqn_latest.zip")
    if os.path.exists(latest_model):
        print(f"[Train] Resuming from {latest_model}")
        model = DQN.load(latest_model, env=env)
    else:
        print("[Train] Starting fresh training")
        model = DQN(
            "MlpPolicy", env,
            buffer_size=TRAIN_BUFFER_SIZE,
            learning_starts=TRAIN_LEARNING_STARTS,
            batch_size=TRAIN_BATCH_SIZE,
            gamma=TRAIN_GAMMA,
            learning_rate=TRAIN_LEARNING_RATE,
            verbose=1,
            tensorboard_log=log_dir,
        )

    callbacks = [
        CheckpointCallback(save_freq=TRAIN_CHECKPOINT_FREQ, save_path=model_dir, name_prefix="atc_dqn"),
    ]

    try:
        model.learn(total_timesteps=TRAIN_TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        print("\n[Train] Interrupted. Saving...")
    finally:
        model.save(latest_model)
        print(f"[Train] Model saved to {latest_model}")
        env.close()


def run_test(model_path=None):
    """학습된 에이전트 테스트 (시각화)"""
    try:
        from stable_baselines3 import DQN
    except ImportError:
        print("stable-baselines3가 필요합니다: pip install stable-baselines3")
        return

    from ai.atc_env import GuidedAirControlEnv

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    if model_path is None:
        model_path = os.path.join(model_dir, "atc_dqn_latest.zip")

    if not os.path.exists(model_path):
        print(f"[Test] Model not found: {model_path}")
        return

    env = GuidedAirControlEnv(render_mode='human')
    model = DQN.load(model_path)

    obs, _ = env.reset()
    total_reward = 0
    steps = 0

    while not env.done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            print(f"[Test] Episode done. Steps={steps}, Reward={total_reward:.0f}")
            break

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="ATC-AI Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                              실시간 시뮬레이터
  python main.py --record                     실시간 + ADS-B 자동 녹화
  python main.py --replay data/recordings     녹화 재생 (1배속)
  python main.py --replay data/rec.jsonl --speed 10   10배속 재생
  python main.py --replay data/recordings --speed 0   최대 속도 재생
  python main.py --train                      RL 학습
  python main.py --test                       학습된 모델 테스트
""")
    parser.add_argument("--train", action="store_true", help="RL 에이전트 학습")
    parser.add_argument("--test", action="store_true", help="학습된 에이전트 테스트")
    parser.add_argument("--model", type=str, default=None, help="테스트할 모델 경로")
    parser.add_argument("--no-gui-panel", action="store_true", help="PyQt GUI 패널 비활성화")
    parser.add_argument("--replay", type=str, default=None,
                        help="녹화 파일/디렉토리 경로 (재생 모드)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="재생 배속 (0=최대, 1=실시간, 10=10배속)")
    parser.add_argument("--no-loop", action="store_true",
                        help="재생 시 반복하지 않음")
    parser.add_argument("--record", action="store_true",
                        help="실시간 수신 중 ADS-B 데이터를 자동 녹화")
    parser.add_argument("--record-dir", type=str, default=None,
                        help="녹화 저장 디렉토리 (기본: data/recordings)")
    args = parser.parse_args()

    if args.train:
        run_train()
    elif args.test:
        run_test(args.model)
    else:
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

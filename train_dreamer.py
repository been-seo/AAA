"""
Dreamer MBPO 학습: World Model + RL Policy 동시 학습

World Model이 환경 역할을 하면서 RL이 관제를 꿈꾸며 학습.
돌발 이벤트(RTB 전투기, WX deviation, 7700 등) 랜덤 주입.
Value Function이 "이 상황이 얼마나 위험한지" 학습 → Safety Advisor 위험도 엔진.

사용법:
    python train_dreamer.py
    python train_dreamer.py --wm-path models/world_model/best_model.pt
"""
import os
import sys
import time
import argparse
import sqlite3

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.world_model.physics_wm import PhysicsWM
from ai.world_model.dataset import TrajectoryDataset, STATE_DIM, NORM_MEAN, NORM_STD
from ai.world_model.dreamer_policy import DreamerTrainer


def create_db_logger(db_path):
    """학습 로그 DB"""
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dreamer_steps (
            step INTEGER PRIMARY KEY,
            timestamp REAL,
            actor_loss REAL,
            critic_loss REAL,
            mean_reward REAL,
            mean_value REAL,
            crashes INTEGER,
            entropy REAL,
            total_episodes INTEGER,
            total_crashes INTEGER,
            total_safe INTEGER
        )
    """)
    # 축별 컬럼 추가 (기존 DB 호환)
    for col in ['v_safety', 'v_efficiency', 'v_mission', 'w_safety', 'w_efficiency', 'w_mission',
                 'r_safety', 'r_efficiency', 'r_mission', 'v_crash', 'v_safe',
                 'paving_regroups', 'paving_max_inter_cos', 'regrouped',
                 'reached', 'fuel_empty', 'avg_ep_len',
                 'min_gap', 'auc_5', 'auc_10', 'auc_20', 'auc_any']:
        try:
            conn.execute(f"ALTER TABLE dreamer_steps ADD COLUMN {col} REAL DEFAULT NULL")
        except Exception:
            pass  # 이미 존재
    conn.commit()
    return conn


def load_world_model(path, device):
    """학습된 (또는 학습 중인) World Model 로드"""
    if not os.path.exists(path):
        print(f"[Dreamer] WM not found: {path}, waiting...")
        while not os.path.exists(path):
            time.sleep(10)
        print(f"[Dreamer] WM found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get('args', {})
    model = PhysicsWM(
        hidden_dim=args.get('hidden_dim', 256),
        past_steps=args.get('past_steps', 6),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    epoch = ckpt.get('epoch', '?')
    print(f"[Dreamer] WM loaded (epoch={epoch})")
    return model


def reload_world_model_if_newer(model, path, device, last_mtime):
    """WM이 업데이트되면 자동 리로드 (동시 학습)"""
    try:
        mtime = os.path.getmtime(path)
        if mtime > last_mtime:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            epoch = ckpt.get('epoch', '?')
            print(f"[Dreamer] WM reloaded (epoch={epoch})")
            return model, mtime
    except Exception:
        pass
    return model, last_mtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm-path', default='models/world_model/best_model.pt')
    parser.add_argument('--data-dir', default='data/recordings')
    parser.add_argument('--save-dir', default='models/dreamer')
    parser.add_argument('--log-db', default='models/dreamer/train_log.db')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--horizon', type=int, default=120,
                        help='Max episode horizon (cap). Episodes terminate on '
                             'reached-destination or fuel-empty before this.')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--total-steps', type=int, default=100000)
    parser.add_argument('--wm-reload-interval', type=int, default=50,
                        help='WM 리로드 체크 주기 (스텝)')
    parser.add_argument('--save-interval', type=int, default=200)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--kst-hours', type=str, default=None,
                        help='KST 시간 필터 "start-end" 형식. 예: "8-16" = '
                             '08:00~16:00 KST. 미지정 시 전체 시간대.')
    parser.add_argument('--weekday-only', action='store_true',
                        help='주말(토/일) 제외, 평일만 사용.')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"[Dreamer] Device: {device}")

    # World Model 로드
    wm = load_world_model(args.wm_path, device)
    wm_mtime = os.path.getmtime(args.wm_path)

    # 데이터셋 (초기 상태를 뽑을 용도)
    rec_dir = args.data_dir
    rec_files = sorted([f for f in os.listdir(rec_dir)
                        if f.endswith('.jsonl') or f.endswith('.db')])
    if not rec_files:
        print("[Dreamer] No recording files found!")
        return

    # 학습 데이터 로드 + 메모리 프리로드 (--kst-hours로 시간대 필터링 가능)
    time_filter = None
    tf_tag = ''
    if args.kst_hours:
        try:
            h_s, h_e = args.kst_hours.split('-')
            time_filter = (int(h_s), int(h_e))
            tf_tag = f' (KST {time_filter[0]:02d}-{time_filter[1]:02d})'
        except (ValueError, IndexError):
            print(f"[Dreamer] --kst-hours 형식 오류 '{args.kst_hours}', 무시")
    dataset = TrajectoryDataset(rec_dir, past_steps=6, future_steps=12,
                                stride=2, time_filter=time_filter,
                                weekday_only=args.weekday_only)
    wk_tag = ' + weekday' if args.weekday_only else ''
    print(f"[Dreamer] Dataset: {len(dataset)} samples{tf_tag}{wk_tag}")
    dataset.preload()

    if len(dataset) == 0:
        print("[Dreamer] Empty dataset!")
        return

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=True)

    # Dreamer trainer
    trainer = DreamerTrainer(
        wm, device=args.device,
        imagination_horizon=args.horizon,
    )

    # 체크포인트 복원
    ckpt_path = os.path.join(args.save_dir, 'latest.pt')
    start_step = 0
    if os.path.exists(ckpt_path):
        trainer.load(ckpt_path)
        # step 복원은 DB에서
        print(f"[Dreamer] Resumed (episodes={trainer.total_episodes}, "
              f"crashes={trainer.total_crashes})")

    # DB 로거
    db = create_db_logger(args.log_db)

    print(f"[Dreamer] Starting training (horizon={args.horizon}, "
          f"batch={args.batch_size})")
    print(f"[Dreamer] WM will auto-reload every {args.wm_reload_interval} steps")
    print("=" * 60)

    step = start_step
    data_iter = iter(loader)

    while step < args.total_steps:
        # 배치 가져오기
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        past, future, ctx, msk, future_raw = batch[:5]
        past_states = past.to(device)           # (B, K, D)
        past_contexts = ctx[:, :past.shape[1]].to(device)  # (B, K, MAX_N, CTX_D)

        # Dreamer train step
        try:
            metrics = trainer.train_step(past_states, past_contexts)
        except (RuntimeError, ValueError) as e:
            # CUDA OOM, NaN 등 복구 가능한 에러만 잡음
            print(f"[Dreamer] Step {step} error: {type(e).__name__}: {e}")
            step += 1
            continue

        step += 1

        # 로깅
        if step % args.log_interval == 0:
            db.execute(
                """INSERT OR REPLACE INTO dreamer_steps
                   (step, timestamp, actor_loss, critic_loss, mean_reward, mean_value,
                    crashes, entropy, total_episodes, total_crashes, total_safe,
                    v_safety, v_efficiency, v_mission, w_safety, w_efficiency, w_mission,
                    r_safety, r_efficiency, r_mission, v_crash, v_safe,
                    paving_regroups, paving_max_inter_cos, regrouped,
                    reached, fuel_empty, avg_ep_len,
                    min_gap, auc_5, auc_10, auc_20, auc_any)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (step, time.time(),
                 metrics['actor_loss'], metrics['critic_loss'],
                 metrics['mean_reward'], metrics['mean_value'],
                 metrics['crashes'], metrics['entropy'],
                 trainer.total_episodes, trainer.total_crashes,
                 trainer.total_safe,
                 metrics.get('v_safety', 0), metrics.get('v_efficiency', 0),
                 metrics.get('v_mission', 0), metrics.get('w_safety', 0),
                 metrics.get('w_efficiency', 0), metrics.get('w_mission', 0),
                 metrics.get('r_safety', 0), metrics.get('r_efficiency', 0),
                 metrics.get('r_mission', 0),
                 metrics.get('v_crash', None),  # None → NULL
                 metrics.get('v_safe', None),
                 metrics.get('paving_regroups', 0),
                 metrics.get('paving_max_inter_cos', 0),
                 1 if metrics.get('regrouped', False) else 0,
                 metrics.get('reached', 0),
                 metrics.get('fuel_empty', 0),
                 metrics.get('avg_ep_len', 0),
                 metrics.get('min_gap', None),
                 metrics.get('auc_5', None),
                 metrics.get('auc_10', None),
                 metrics.get('auc_20', None),
                 metrics.get('auc_any', None)))
            db.commit()

            crash_rate = (trainer.total_crashes / max(trainer.total_episodes, 1)) * 100
            vs = f"V=[S{metrics.get('v_safety',0):.1f}/E{metrics.get('v_efficiency',0):.1f}/M{metrics.get('v_mission',0):.1f}]"
            rs = f"R=[S{metrics.get('r_safety',0):.1f}/E{metrics.get('r_efficiency',0):.1f}/M{metrics.get('r_mission',0):.1f}]"
            cl = f"CL=[S{metrics.get('c_safety',0):.0f}/E{metrics.get('c_efficiency',0):.0f}/M{metrics.get('c_mission',0):.0f}]"
            vc = metrics.get('v_crash', None)
            vf = metrics.get('v_safe', None)
            cos = metrics.get('max_cos_sim', 0)
            cert = metrics.get('certificate_h', 0)
            reached = metrics.get('reached', 0)
            fuel_empty = metrics.get('fuel_empty', 0)
            ep_len = metrics.get('avg_ep_len', 0)
            stage = metrics.get('curriculum_stage', 1)
            def _fmt_auc(v):
                return f"{v:.3f}" if v is not None else "----"
            def _fmt_v(v):
                return f"{v:+.1f}" if v is not None else "--"

            # Prediction accuracy: "Critic가 매 순간 위험 상태를 정상 상태보다
            # 더 위험하게 평가한 비율" — 5-step lookahead AUC
            pred_acc = metrics.get('auc_5')
            mg = metrics.get('min_gap')
            mg_str = f"{mg:+.1f}" if mg is not None else "--"
            print(f"[{step:6d}] PRED={_fmt_auc(pred_acc)} "
                  f"mingap={mg_str} "
                  f"Vc={_fmt_v(vc)}/Vs={_fmt_v(vf)} "
                  f"crash={metrics['crashes']} reach={reached} "
                  f"fuel0={fuel_empty} len={ep_len:.0f} "
                  f"ent={metrics['entropy']:.3f} "
                  f"| {trainer.total_episodes}ep "
                  f"{crash_rate:.1f}%")

        # WM 리로드 (동시 학습)
        if step % args.wm_reload_interval == 0:
            wm, wm_mtime = reload_world_model_if_newer(
                wm, args.wm_path, device, wm_mtime)

        # 저장
        if step % args.save_interval == 0:
            trainer.save(ckpt_path)
            # best도 저장 (crash rate 기준)
            best_path = os.path.join(args.save_dir, 'best.pt')
            trainer.save(best_path)

    trainer.save(ckpt_path)
    db.close()
    print(f"[Dreamer] Done. {trainer.total_episodes} episodes, "
          f"{trainer.total_crashes} crashes, {trainer.total_safe} safe")


if __name__ == '__main__':
    main()

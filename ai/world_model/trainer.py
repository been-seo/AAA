"""
World Model 학습 파이프라인 (Physics-Informed)

ADS-B 녹화 데이터로 TrajectoryPredictor 학습:
1. 데이터 로드 + 전처리 (dataset.py)
2. Physics envelope 기반 ratio 예측 학습
3. 검증: MC predict → pos_err (NM)
4. 체크포인트 저장

사용법:
    python -u -m ai.world_model.trainer --data-dir data/recordings --epochs 200
"""
import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .dataset import TrajectoryDataset, STATE_DIM, NORM_MEAN, NORM_STD
from .trajectory_predictor import TrajectoryPredictor


# ── 학습 결과 DB ──

_TRAIN_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL,
    args TEXT,
    device TEXT,
    param_count INTEGER,
    train_size INTEGER,
    val_size INTEGER
);
CREATE TABLE IF NOT EXISTS epochs (
    run_id INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    train_loss REAL,
    train_recon REAL,
    train_kl REAL,
    val_loss REAL,
    val_recon REAL,
    val_kl REAL,
    pos_err_1 REAL,
    pos_err_5 REAL,
    pos_err_final REAL,
    lr REAL,
    duration_sec REAL,
    is_best INTEGER DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
"""


class TrainLogger:
    """학습 결과를 SQLite에 기록"""

    def __init__(self, db_path):
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_TRAIN_DB_SCHEMA)
        self._conn.commit()
        self.run_id = None

    def start_run(self, args_dict, device, param_count, train_size, val_size):
        cur = self._conn.execute(
            "INSERT INTO runs (started_at, args, device, param_count, train_size, val_size) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), str(args_dict), str(device), param_count, train_size, val_size))
        self.run_id = cur.lastrowid
        self._conn.commit()

    def log_epoch(self, epoch, train_m, val_m, lr, duration, is_best=False):
        self._conn.execute(
            "INSERT INTO epochs (run_id, epoch, train_loss, train_recon, train_kl, "
            "val_loss, val_recon, val_kl, pos_err_1, pos_err_5, pos_err_final, "
            "lr, duration_sec, is_best) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.run_id, epoch,
             train_m['loss'], train_m['recon'], train_m['kl'],
             val_m['loss'], val_m['recon'], val_m['kl'],
             val_m.get('pos_err_1step', 0), val_m.get('pos_err_5step', 0),
             val_m.get('pos_err_final', 0), lr, duration, int(is_best)))
        self._conn.commit()

    def close(self):
        self._conn.close()


# ── Kalman Filter LR Scheduler ──

class KalmanLRScheduler:
    """
    칼만 필터 기반 LR 스케줄러.

    val_loss의 노이즈를 필터링하여 실제 개선률(delta)을 추정.
    delta가 크면 LR 유지, 0에 수렴하면 LR 감소.
    """

    def __init__(self, optimizer, lr_init, lr_min=1e-6,
                 process_noise=0.1, measurement_noise=2.0,
                 decay_rate=0.002):
        self.optimizer = optimizer
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.lr = lr_init
        self.decay_rate = decay_rate

        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 100.0
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.array([[process_noise * 0.1, 0.0], [0.0, process_noise]])
        self.R = np.array([[measurement_noise]])
        self._initialized = False
        self._epoch = 0

    def step(self, val_loss):
        self._epoch += 1
        z = np.array([val_loss])

        if not self._initialized:
            self.x[0] = val_loss
            self.x[1] = 0.0
            self._initialized = True
            return self.lr

        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + (K @ y).flatten()
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        filtered_delta = self.x[1]
        delta_uncertainty = np.sqrt(self.P[1, 1])
        improvement = max(0.0, -filtered_delta)

        if improvement > 1e-6:
            lr_scale = min(1.0, improvement / (improvement + self.decay_rate))
        else:
            lr_scale = 0.5

        confidence = 1.0 / (1.0 + delta_uncertainty)
        lr_scale = lr_scale * confidence + (1.0 - confidence) * (self.lr / self.lr_init)

        self.lr = max(self.lr_min, self.lr_init * lr_scale)
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.lr
        return self.lr

    def state_dict(self):
        return {
            'x': self.x.copy(), 'P': self.P.copy(),
            'lr': self.lr, 'epoch': self._epoch,
            'initialized': self._initialized,
        }

    def load_state_dict(self, d):
        self.x = d['x'].copy()
        self.P = d['P'].copy()
        self.lr = d['lr']
        self._epoch = d['epoch']
        self._initialized = d['initialized']
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.lr


# ── Train / Validate ──

def train_epoch(model, data, train_idx, batch_size, optimizer, device,
                kl_weight=0.1, grad_clip=100.0):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    total_recon = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    n_batches = 0

    idx = train_idx[torch.randperm(len(train_idx))]
    n_total = (len(idx) + batch_size - 1) // batch_size
    has_world = 'world' in data

    # PAVING state (passed via trainer attributes)
    paving_mgr = getattr(model, '_paving_mgr', None)
    paving_itm = getattr(model, '_paving_itm', None)
    paving_ctrl = getattr(model, '_paving_ctrl', None)
    task_weights = getattr(model, '_task_weights', None)
    paving_every = 50  # compute gram every N batches

    for b in range(n_total):
        bi = idx[b * batch_size : (b + 1) * batch_size]
        past = data['past'][bi].to(device, non_blocking=True)
        future = data['future'][bi].to(device, non_blocking=True)
        ctx = data['ctx'][bi].to(device, non_blocking=True)
        world = data['world'][bi].to(device, non_blocking=True) if has_world else None

        losses = model.compute_loss(past, future, ctx,
                                    kl_weight=kl_weight, world=world,
                                    task_weights=task_weights)

        optimizer.zero_grad()
        losses['total_loss'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += losses['total_loss'].detach()
        total_recon += losses['recon_loss'].detach()
        total_kl += losses['kl_loss'].detach()
        n_batches += 1

        # ── PAVING: periodic controller step ──
        if paving_ctrl and paving_itm and n_batches % paving_every == 0:
            # Compute fresh task losses (no aggregation)
            losses_p = model.compute_loss(past, future, ctx,
                                          kl_weight=kl_weight, world=world)
            tl = losses_p['task_losses']
            try:
                gram, grad_norms = paving_itm.compute_gram(model, tl)
                kappa = paving_itm.condition_number(gram)
                tl_scalar = {k: float(v.item()) for k, v in tl.items()}
                h = sum(v**2 for v in tl_scalar.values()) / len(tl_scalar)

                current_lr = optimizer.param_groups[0]['lr']
                decision = paving_ctrl.step(
                    kappa=kappa, h=h, task_losses=tl_scalar,
                    gram=gram, task_names=paving_itm.task_names,
                    current_lr=current_lr, grouping_mgr=paving_mgr,
                    verbose=True)

                # Apply decision
                if decision['action'] == 'lr_cut':
                    for pg in optimizer.param_groups:
                        pg['lr'] = decision['new_lr']
                    print(f"[PAVING] LR cut: {current_lr:.2e} → {decision['new_lr']:.2e}", flush=True)
                elif decision['action'] == 'lr_recover':
                    for pg in optimizer.param_groups:
                        pg['lr'] = decision['new_lr']
                    print(f"[PAVING] LR recover: {decision['new_lr']:.2e}", flush=True)
                elif decision['action'] == 'rebalance':
                    model._task_weights = decision['new_weights']
                    task_weights = model._task_weights
                    print(f"[PAVING] Rebalanced weights: "
                          f"{ {k:f'{v:.2f}' for k,v in decision['new_weights'].items()} }", flush=True)
                elif decision['action'] == 'regroup':
                    pair = decision['merge_pair']
                    ok = paving_mgr.merge(pair[0], pair[1])
                    print(f"[PAVING] Regroup {pair}, cos={decision['merge_cos']:.2f}, "
                          f"merged={ok}, groups={paving_mgr.summary()['n_groups']}", flush=True)
            except Exception as e:
                print(f"[PAVING] Error: {e}", flush=True)

        if n_batches % 20 == 0:
            avg = total_loss.item() / n_batches
            print(f"  step {n_batches}/{n_total} loss={avg:.4f}", flush=True)

    return {
        'loss': total_loss.item() / max(n_batches, 1),
        'recon': total_recon.item() / max(n_batches, 1),
        'kl': total_kl.item() / max(n_batches, 1),
    }


@torch.no_grad()
def validate(model, data, val_idx, batch_size, device, kl_weight=0.1):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_recon = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    n_batches = 0
    position_errors = []
    eval_metrics = {}  # coverage, sharpness 등

    n_total = (len(val_idx) + batch_size - 1) // batch_size
    has_world = 'world' in data
    nm = torch.from_numpy(NORM_MEAN).to(device)
    ns = torch.from_numpy(NORM_STD).to(device)

    for b in range(n_total):
        bi = val_idx[b * batch_size : (b + 1) * batch_size]
        past = data['past'][bi].to(device, non_blocking=True)
        future = data['future'][bi].to(device, non_blocking=True)
        ctx = data['ctx'][bi].to(device, non_blocking=True)
        future_raw = data['future_raw'][bi].to(device, non_blocking=True)
        world = data['world'][bi].to(device, non_blocking=True) if has_world else None

        losses = model.compute_loss(past, future, ctx,
                                    kl_weight=kl_weight, world=world)

        total_loss += losses['total_loss']
        total_recon += losses['recon_loss']
        total_kl += losses['kl_loss']

        # 위치 평가: MC 샘플 분포 기반 calibration + pos_err
        # GPU 메모리 절약: 소량 랜덤 샘플링 + zero 필터링
        if b == 0:
            valid_mask = future_raw.abs().sum(dim=(1, 2)) > 1e-6
            valid_past = past[valid_mask]
            valid_fr = future_raw[valid_mask]
            valid_ctx = ctx[valid_mask]
            valid_world = None
            if has_world:
                w_batch = data['world'][bi].to(device, non_blocking=True)
                valid_world = w_batch[valid_mask]
            n_eval = min(128, valid_past.shape[0])
            if n_eval > 0:
                rand_idx = torch.randperm(valid_past.shape[0], device=device)[:n_eval]
                eval_past = valid_past[rand_idx]
                eval_ctx = valid_ctx[rand_idx, :past.shape[1]]
                eval_world = None
                if valid_world is not None:
                    eval_world = valid_world[rand_idx, :past.shape[1]]
                traj = model.predict(eval_past, eval_ctx,
                                     num_samples=20,
                                     future_steps=valid_fr.shape[1],
                                     world_past=eval_world)
                # traj: (n, 20, T, D) 비정규화
                fr = valid_fr[rand_idx]  # (n, T, D) — 이미 GPU

                # 각 샘플의 위치 오차 (NM)
                cos_lat = torch.cos(torch.deg2rad(fr[:, :, 0])).clamp(min=0.5)
                # MC 샘플별 오차: (n, 20, T)
                lat_err_mc = (traj[:, :, :, 0] - fr[:, None, :, 0]) * 60.0
                lon_err_mc = (traj[:, :, :, 1] - fr[:, None, :, 1]) * 60.0 * cos_lat[:, None, :]
                pos_err_mc = torch.sqrt(lat_err_mc**2 + lon_err_mc**2)  # (n, 20, T)

                # 1) pos_err: MC 중앙값 기준
                pred_median = traj.median(dim=1).values  # (n, T, D)
                lat_err = (pred_median[:, :, 0] - fr[:, :, 0]) * 60.0
                lon_err = (pred_median[:, :, 1] - fr[:, :, 1]) * 60.0 * cos_lat
                pos_err = torch.sqrt(lat_err**2 + lon_err**2)
                position_errors.append(pos_err.cpu().numpy())

                # 시간축별 오차 분포 → 예측 범위 상하한
                err_q10 = pos_err.quantile(0.1, dim=0)   # (T,) 하한 (최소 오차)
                err_q50 = pos_err.quantile(0.5, dim=0)   # (T,) 중앙값
                err_q90 = pos_err.quantile(0.9, dim=0)   # (T,) 상한 (최대 오차)
                eval_metrics['err_bounds'] = {
                    'q10': err_q10.cpu().tolist(),  # 하한
                    'q50': err_q50.cpu().tolist(),  # 중앙
                    'q90': err_q90.cpu().tolist(),  # 상한
                }

                # 2) Coverage: MC 분포 범위가 실제 위치를 포함하는 비율
                # MC 샘플들의 위치 분포에서 90% 반경 계산
                # 각 (항공기, 타임스텝)별 MC 샘플 중심 = median
                mc_center_lat = traj[:, :, :, 0].median(dim=1).values  # (n, T)
                mc_center_lon = traj[:, :, :, 1].median(dim=1).values  # (n, T)
                # MC 샘플들의 중심으로부터 거리
                mc_lat_d = (traj[:, :, :, 0] - mc_center_lat[:, None, :]) * 60.0
                mc_lon_d = (traj[:, :, :, 1] - mc_center_lon[:, None, :]) * 60.0 * cos_lat[:, None, :]
                mc_dist_from_center = torch.sqrt(mc_lat_d**2 + mc_lon_d**2)  # (n, 20, T)
                # 90% 반경: MC 샘플의 90th percentile 거리
                radius_90 = mc_dist_from_center.quantile(0.9, dim=1)  # (n, T)

                # 실제 위치의 MC 중심으로부터 거리
                actual_lat_d = (fr[:, :, 0] - mc_center_lat) * 60.0
                actual_lon_d = (fr[:, :, 1] - mc_center_lon) * 60.0 * cos_lat
                actual_dist = torch.sqrt(actual_lat_d**2 + actual_lon_d**2)  # (n, T)

                # Coverage: 실제가 90% 반경 안에 있는 비율
                coverage_90 = (actual_dist <= radius_90).float().mean()

                # 3) Sharpness: 90% 반경 평균 (작을수록 좋음)
                sharpness = radius_90.mean()

                eval_metrics['coverage_90'] = coverage_90.item()
                eval_metrics['sharpness_nm'] = sharpness.item()

        n_batches += 1

    pos_err_all = np.concatenate(position_errors, axis=0) if position_errors else np.array([[0]])

    result = {
        'loss': total_loss.item() / max(n_batches, 1),
        'recon': total_recon.item() / max(n_batches, 1),
        'kl': total_kl.item() / max(n_batches, 1),
        'pos_err_1step': float(pos_err_all[:, 0].mean()) if pos_err_all.shape[1] > 0 else 0,
        'pos_err_5step': float(pos_err_all[:, min(4, pos_err_all.shape[1]-1)].mean()),
        'pos_err_final': float(pos_err_all[:, -1].mean()),
        'coverage_90': eval_metrics.get('coverage_90', 0),
        'sharpness_nm': eval_metrics.get('sharpness_nm', 0),
        'err_bounds': eval_metrics.get('err_bounds', None),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="World Model Trainer (Physics-Informed)")
    parser.add_argument('--data-dir', type=str, default='data/recordings')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--past-steps', type=int, default=6)
    parser.add_argument('--future-steps', type=int, default=12)
    parser.add_argument('--kl-weight', type=float, default=0.1)
    parser.add_argument('--save-dir', type=str, default='models/world_model')
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--log-db', type=str, default='models/world_model/train_log.db')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋
    dataset = TrajectoryDataset(
        data_dir=args.data_dir,
        past_steps=args.past_steps,
        future_steps=args.future_steps,
        stride=4,
        max_files=args.max_files,
    )
    if len(dataset) == 0:
        return

    print(f"[Trainer] Preloading {len(dataset)} samples...")
    sys.stdout.flush()
    dataset.preload()
    print(f"[Trainer] Preload done. has_world={dataset.has_world}")
    sys.stdout.flush()

    # Train/Val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    perm = torch.randperm(len(dataset))
    train_idx = perm[val_size:]
    val_idx = perm[:val_size]

    data = {
        'past': dataset._cache_past,
        'future': dataset._cache_future,
        'ctx': dataset._cache_ctx,
        'future_raw': dataset._cache_future_raw,
    }
    if dataset.has_world:
        data['world'] = dataset._cache_world

    # 모델
    model = TrajectoryPredictor(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    # DB 로거
    db_log = TrainLogger(args.log_db)
    db_log.start_run(vars(args), device, param_count, train_size, val_size)

    # Optimizer + Kalman LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = KalmanLRScheduler(optimizer, lr_init=args.lr)

    # ── PAVING MTL controller (현재 비활성화: 관찰만) ──
    # Inner task 간 redundancy로 κ 매우 크고, weight rebalance가
    # KL을 과도 regularization시켜 posterior collapse 유발.
    # 수동 균형 가중치로 고정 학습 후, 안정화되면 재활성화.
    #
    # 수동 가중치: 위치 우선, KL은 free_nats 높여서 posterior 다양성 유지
    from .trajectory_predictor import TrajectoryPredictor as _TP
    task_names = _TP.INNER_TASKS
    model._task_weights = {
        'pos_lat':      3.0, 'pos_lon':    3.0,   # 위치 강조
        'alt':          2.0, 'vrate':      1.0,
        'gs':           1.5, 'ias':        0.5, 'mach':  0.5,  # 속도 (gs 우선)
        'track_sin':    2.0, 'track_cos':  2.0,   # 방향
        'heading_circ': 1.0,
        'kl':           0.1,   # KL 약하게 (posterior 다양성 확보)
    }
    model._paving_ctrl = None  # 비활성화 → trainer에서 skip
    model._paving_itm = None
    model._paving_mgr = None
    print(f"[WM] Using fixed task weights, PAVING controller DISABLED", flush=True)

    # Resume
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt and 'x' in ckpt['scheduler_state_dict']:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('val_loss', float('inf'))
            print(f"[Trainer] Resumed from {ckpt_path} (epoch {ckpt['epoch']})")
            sys.stdout.flush()

    def kl_schedule(epoch):
        if epoch < 10:
            return args.kl_weight * (epoch / 10)
        return args.kl_weight

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_train_batches = (train_size + args.batch_size - 1) // args.batch_size
    print(f"[Trainer] Starting: {args.epochs} epochs (from {start_epoch}), "
          f"{n_train_batches} batches/epoch, batch_size={args.batch_size}, "
          f"device={device}")
    sys.stdout.flush()

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        kl_w = kl_schedule(epoch)

        train_metrics = train_epoch(model, data, train_idx, args.batch_size,
                                    optimizer, device, kl_weight=kl_w)
        val_metrics = validate(model, data, val_idx, args.batch_size,
                               device, kl_weight=kl_w)

        scheduler.step(val_metrics['loss'])
        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        is_best = val_metrics['loss'] < best_val_loss
        db_log.log_epoch(epoch, train_metrics, val_metrics, lr, dt, is_best)
        cov = val_metrics.get('coverage_90', 0)
        shrp = val_metrics.get('sharpness_nm', 0)
        bounds = val_metrics.get('err_bounds')
        bounds_str = ""
        if bounds:
            # 최종 스텝(120초)의 상하한
            q10_f = bounds['q10'][-1]
            q50_f = bounds['q50'][-1]
            q90_f = bounds['q90'][-1]
            bounds_str = f"range=[{q10_f:.1f}/{q50_f:.1f}/{q90_f:.1f}]NM "
        print(f"[Epoch {epoch}/{args.epochs}] loss={train_metrics['loss']:.4f} "
              f"val={val_metrics['loss']:.4f} pos_err={val_metrics.get('pos_err_final',0):.2f}NM "
              f"cov90={cov:.0%} shrp={shrp:.1f}NM {bounds_str}"
              f"lr={lr:.1e} dt={dt:.0f}s {'*BEST*' if is_best else ''}")
        sys.stdout.flush()

        if is_best:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'args': vars(args),
            }, save_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[Trainer] Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
            }, save_dir / f'checkpoint_ep{epoch}.pt')

    # 최종 저장
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': best_val_loss,
        'val_metrics': val_metrics,
        'args': vars(args),
    }, save_dir / 'final_model.pt')

    db_log.close()
    print(f"[Trainer] Done. Best val_loss={best_val_loss:.4f}")


if __name__ == '__main__':
    main()

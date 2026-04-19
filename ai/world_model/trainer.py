"""
World Model 학습 파이프라인

ADS-B 녹화 데이터로 TrajectoryPredictor 학습:
1. 데이터 로드 + 전처리 (dataset.py)
2. RSSM 모델 학습 (reconstruction + KL loss)
3. 검증: conflict prediction 정확도
4. 체크포인트 저장

v2: World context (웨이포인트 피처) 지원

사용법:
    python -m ai.world_model.trainer --data-dir data/recordings --epochs 100
"""
import argparse
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .dataset import TrajectoryDataset, STATE_DIM
from .trajectory_predictor import TrajectoryPredictor
from .conflict_detector import ConflictDetector


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


def _unpack_batch(batch, device):
    """배치 언팩 — world context 유무에 따라 처리."""
    if len(batch) == 6:
        past, future, ctx, mask, future_raw, world = [x.to(device) for x in batch]
    else:
        past, future, ctx, mask, future_raw = [x.to(device) for x in batch]
        world = None
    return past, future, ctx, mask, future_raw, world


def train_epoch(model, loader, optimizer, device, kl_weight=0.1, grad_clip=100.0):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0

    for batch in loader:
        past, future, ctx, mask, future_raw, world = _unpack_batch(batch, device)

        losses = model.compute_loss(past, future, ctx, future_raw,
                                    world=world, kl_weight=kl_weight)

        optimizer.zero_grad()
        losses['total_loss'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += losses['total_loss'].item()
        total_recon += losses['recon_loss'].item()
        total_kl += losses['kl_loss'].item()
        n_batches += 1

    return {
        'loss': total_loss / max(n_batches, 1),
        'recon': total_recon / max(n_batches, 1),
        'kl': total_kl / max(n_batches, 1),
    }


@torch.no_grad()
def validate(model, loader, device, kl_weight=0.1):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0

    position_errors = []

    for batch in loader:
        past, future, ctx, mask, future_raw, world = _unpack_batch(batch, device)

        losses = model.compute_loss(past, future, ctx, future_raw,
                                    world=world, kl_weight=kl_weight)

        total_loss += losses['total_loss'].item()
        total_recon += losses['recon_loss'].item()
        total_kl += losses['kl_loss'].item()

        # 위치 오차 계산 (비정규화 공간)
        output = model(past, future, ctx, world=world)
        pred_mean = output['future_pred_mean']
        pred_raw = pred_mean * model.norm_std + model.norm_mean
        lat_err = (pred_raw[:, :, 0] - future_raw[:, :, 0]) * 60.0
        lon_err = (pred_raw[:, :, 1] - future_raw[:, :, 1]) * 60.0 * \
                  torch.cos(torch.deg2rad(future_raw[:, :, 0])).clamp(min=0.5)
        pos_err = torch.sqrt(lat_err**2 + lon_err**2)
        position_errors.append(pos_err.cpu().numpy())

        n_batches += 1

    pos_err_all = np.concatenate(position_errors, axis=0) if position_errors else np.array([[0]])

    return {
        'loss': total_loss / max(n_batches, 1),
        'recon': total_recon / max(n_batches, 1),
        'kl': total_kl / max(n_batches, 1),
        'pos_err_1step': float(pos_err_all[:, 0].mean()) if pos_err_all.shape[1] > 0 else 0,
        'pos_err_5step': float(pos_err_all[:, min(4, pos_err_all.shape[1]-1)].mean()),
        'pos_err_final': float(pos_err_all[:, -1].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="World Model Trainer")
    parser.add_argument('--data-dir', type=str, default='data/recordings',
                        help='ADS-B 녹화 디렉토리')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--past-steps', type=int, default=6,
                        help='입력 시퀀스 길이 (과거 스텝)')
    parser.add_argument('--future-steps', type=int, default=12,
                        help='예측 시퀀스 길이 (미래 스텝)')
    parser.add_argument('--kl-weight', type=float, default=0.1)
    parser.add_argument('--save-dir', type=str, default='models/world_model')
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--log-db', type=str, default='models/world_model/train_log.db',
                        help='학습 로그 DB 경로')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 로드
    dataset = TrajectoryDataset(
        data_dir=args.data_dir,
        past_steps=args.past_steps,
        future_steps=args.future_steps,
        max_files=args.max_files,
    )

    if len(dataset) == 0:
        return

    # Train/Val 분할
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # 모델
    model = TrajectoryPredictor(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        world_dim=dataset.world_dim,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Trainer] Model params: {param_count:,}")
    print(f"[Trainer] World context: {'enabled' if dataset.has_world else 'disabled'} "
          f"(dim={dataset.world_dim})")

    # DB 로거
    db_log = TrainLogger(args.log_db)
    db_log.start_run(vars(args), device, param_count, train_size, val_size)

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    def kl_schedule(epoch):
        if epoch < 10:
            return args.kl_weight * (epoch / 10)
        return args.kl_weight

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        kl_w = kl_schedule(epoch)

        train_metrics = train_epoch(model, train_loader, optimizer, device, kl_weight=kl_w)
        val_metrics = validate(model, val_loader, device, kl_weight=kl_w)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        is_best = val_metrics['loss'] < best_val_loss
        db_log.log_epoch(epoch, train_metrics, val_metrics, lr, dt, is_best)

        if is_best:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'args': vars(args),
                'world_dim': dataset.world_dim,
            }, save_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'world_dim': dataset.world_dim,
            }, save_dir / f'checkpoint_ep{epoch}.pt')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_metrics': val_metrics,
        'world_dim': dataset.world_dim,
    }, save_dir / 'final_model.pt')

    db_log.close()


if __name__ == '__main__':
    main()

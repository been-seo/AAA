"""
WM 예측 실패 사례 분석 도구

held-out 데이터에서 예측 수행 → 오차 분포 + 실패 패턴 파악
- 최악 사례 (pos_err q99)
- 헤딩 오차 분포
- 고도 오차 분포
- MC calibration (cov90, sharpness)
- 실패 유형별 패턴 (선회 구간, 고도변경, 공항 근처 등)
"""
import sys
import os
import json
import math
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.world_model.trajectory_predictor import TrajectoryPredictor
from ai.world_model.dataset import (
    TrajectoryDataset, STATE_DIM, NORM_MEAN, NORM_STD, MAX_NEIGHBORS, CONTEXT_DIM
)


def analyze(model_path, data_dir='data/recordings', num_samples_to_analyze=500,
            num_mc=30, future_steps=12):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Analyze] Device: {device}")

    # Load model
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    args = ckpt.get('args', {})
    model = TrajectoryPredictor(
        hidden_dim=args.get('hidden_dim', 256),
        latent_dim=args.get('latent_dim', 64),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[Analyze] Model loaded (epoch={ckpt.get('epoch', '?')})")

    # Load dataset
    ds = TrajectoryDataset(data_dir, past_steps=6, future_steps=future_steps, stride=8)
    ds.preload()
    print(f"[Analyze] Dataset: {len(ds):,} samples")

    # Sample subset
    np.random.seed(42)
    n = min(num_samples_to_analyze, len(ds))
    indices = np.random.choice(len(ds), n, replace=False)

    pos_errors = []        # (n, T) median err
    hdg_errors = []        # (n, T) heading err
    alt_errors = []        # (n, T) alt err
    mc_spreads = []        # (n, T) MC 90% radius
    coverages = []         # (n, T) 1 if covered else 0

    # Context info for failure analysis
    init_states = []       # (n, STATE_DIM) first past state (raw)
    true_finals = []       # (n, STATE_DIM) true final position
    pred_medians = []      # (n, T, STATE_DIM) median prediction

    with torch.no_grad():
        for i, idx in enumerate(indices):
            items = ds[idx]
            past, future, ctx, mask, future_raw = items[:5]
            has_world = len(items) > 5
            world = items[5] if has_world else None

            # Skip zero-padded samples
            if future_raw.abs().sum() < 1e-6:
                continue

            past = past.unsqueeze(0).to(device)
            ctx = ctx.unsqueeze(0).to(device)
            future_raw = future_raw.to(device)
            world_past = None
            if world is not None:
                world_past = world[:past.shape[1]].unsqueeze(0).to(device)

            traj = model.predict(past, ctx[:, :past.shape[1]],
                                 num_samples=num_mc,
                                 future_steps=future_steps,
                                 world_past=world_past)
            # traj: (1, MC, T, D)
            traj = traj[0]  # (MC, T, D)

            # Denormalize past for context info
            past_raw = past[0].cpu().numpy() * NORM_STD + NORM_MEAN
            init_state = past_raw[-1]  # last past state

            # Compute errors
            cos_lat = np.cos(np.radians(future_raw[:, 0].cpu().numpy())).clip(min=0.5)
            pred_med = traj.median(dim=0).values.cpu().numpy()  # (T, D)
            true = future_raw.cpu().numpy()  # (T, D)

            lat_err = (pred_med[:, 0] - true[:, 0]) * 60
            lon_err = (pred_med[:, 1] - true[:, 1]) * 60 * cos_lat
            pos_err = np.sqrt(lat_err**2 + lon_err**2)  # (T,)

            hdg_err = np.abs((pred_med[:, 4] - true[:, 4] + 180) % 360 - 180)
            alt_err = np.abs(pred_med[:, 2] - true[:, 2])

            # MC spread (90% radius from MC mean)
            mc_np = traj.cpu().numpy()  # (MC, T, D)
            mc_center_lat = np.median(mc_np[:, :, 0], axis=0)
            mc_center_lon = np.median(mc_np[:, :, 1], axis=0)
            mc_lat_d = (mc_np[:, :, 0] - mc_center_lat[None, :]) * 60
            mc_lon_d = (mc_np[:, :, 1] - mc_center_lon[None, :]) * 60 * cos_lat[None, :]
            mc_dist = np.sqrt(mc_lat_d**2 + mc_lon_d**2)  # (MC, T)
            q90_radius = np.quantile(mc_dist, 0.9, axis=0)  # (T,)

            # Coverage: is true within 90% radius?
            actual_lat_d = (true[:, 0] - mc_center_lat) * 60
            actual_lon_d = (true[:, 1] - mc_center_lon) * 60 * cos_lat
            actual_dist = np.sqrt(actual_lat_d**2 + actual_lon_d**2)
            covered = (actual_dist <= q90_radius).astype(float)  # (T,)

            pos_errors.append(pos_err)
            hdg_errors.append(hdg_err)
            alt_errors.append(alt_err)
            mc_spreads.append(q90_radius)
            coverages.append(covered)
            init_states.append(init_state)
            true_finals.append(true[-1])
            pred_medians.append(pred_med)

            if (i + 1) % 50 == 0:
                print(f"[Analyze] {i+1}/{n} samples processed")

    pos_errors = np.array(pos_errors)  # (N, T)
    hdg_errors = np.array(hdg_errors)
    alt_errors = np.array(alt_errors)
    mc_spreads = np.array(mc_spreads)
    coverages = np.array(coverages)
    init_states = np.array(init_states)
    true_finals = np.array(true_finals)
    pred_medians = np.array(pred_medians)

    N = pos_errors.shape[0]
    print(f"\n[Analyze] Analyzed {N} samples\n")

    # ═══════════════════════════════════════
    # OVERALL STATS
    # ═══════════════════════════════════════
    print("=" * 60)
    print("OVERALL ERROR STATISTICS")
    print("=" * 60)

    for t_idx, t_sec in [(0, 10), (5, 60), (11, 120)]:
        if t_idx >= pos_errors.shape[1]:
            continue
        pe = pos_errors[:, t_idx]
        he = hdg_errors[:, t_idx]
        ae = alt_errors[:, t_idx]
        ms = mc_spreads[:, t_idx]
        cv = coverages[:, t_idx]
        print(f"\nt+{t_sec}s:")
        print(f"  pos_err: q10={np.quantile(pe, 0.1):.1f}  q50={np.quantile(pe, 0.5):.1f}  "
              f"q90={np.quantile(pe, 0.9):.1f}  q99={np.quantile(pe, 0.99):.1f} NM")
        print(f"  hdg_err: q50={np.quantile(he, 0.5):.1f}  q90={np.quantile(he, 0.9):.1f}  "
              f"q99={np.quantile(he, 0.99):.1f} deg")
        print(f"  alt_err: q50={np.quantile(ae, 0.5):.0f}  q90={np.quantile(ae, 0.9):.0f}  "
              f"q99={np.quantile(ae, 0.99):.0f} ft")
        print(f"  MC spread (q90 radius): mean={ms.mean():.2f}NM")
        print(f"  Coverage: {cv.mean()*100:.0f}% (target 90%)")

    # ═══════════════════════════════════════
    # WORST CASES (top 20)
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("WORST 20 CASES (at t+120s)")
    print("=" * 60)

    worst_idx = np.argsort(pos_errors[:, -1])[-20:][::-1]
    print(f"{'idx':>4} {'pos_err':>8} {'hdg_err':>8} {'alt_err':>8} "
          f"{'init_alt':>8} {'init_gs':>6} {'init_vrate':>10} {'init_track':>10}")
    for i in worst_idx:
        pe = pos_errors[i, -1]
        he = hdg_errors[i, -1]
        ae = alt_errors[i, -1]
        s = init_states[i]
        print(f"{i:>4} {pe:>8.1f} {he:>8.1f} {ae:>8.0f} "
              f"{s[2]:>8.0f} {s[3]:>6.0f} {s[5]:>10.0f} {s[4]:>10.0f}")

    # ═══════════════════════════════════════
    # FAILURE PATTERNS BY CONTEXT
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("FAILURE PATTERNS BY CONTEXT (t+120s)")
    print("=" * 60)

    final_pe = pos_errors[:, -1]
    hdg_vrate = init_states[:, 5]  # vertical rate

    # Climbing vs Cruising vs Descending
    climbing = np.abs(hdg_vrate) > 500
    climb_pos = hdg_vrate > 500
    desc = hdg_vrate < -500
    cruise = np.abs(hdg_vrate) <= 500

    print(f"\nBy vertical state:")
    print(f"  Climbing   (n={climb_pos.sum()}): q50={np.quantile(final_pe[climb_pos], 0.5):.1f}  q90={np.quantile(final_pe[climb_pos], 0.9):.1f} NM")
    print(f"  Descending (n={desc.sum()}): q50={np.quantile(final_pe[desc], 0.5):.1f}  q90={np.quantile(final_pe[desc], 0.9):.1f} NM")
    print(f"  Cruising   (n={cruise.sum()}): q50={np.quantile(final_pe[cruise], 0.5):.1f}  q90={np.quantile(final_pe[cruise], 0.9):.1f} NM")

    # By altitude
    init_alt = init_states[:, 2]
    high = init_alt > 25000
    mid = (init_alt >= 10000) & (init_alt <= 25000)
    low = init_alt < 10000

    print(f"\nBy altitude:")
    print(f"  High  (>25k ft, n={high.sum()}): q50={np.quantile(final_pe[high], 0.5):.1f}  q90={np.quantile(final_pe[high], 0.9):.1f} NM")
    print(f"  Mid   (10-25k, n={mid.sum()}):   q50={np.quantile(final_pe[mid], 0.5):.1f}  q90={np.quantile(final_pe[mid], 0.9):.1f} NM")
    print(f"  Low   (<10k, n={low.sum()}):     q50={np.quantile(final_pe[low], 0.5):.1f}  q90={np.quantile(final_pe[low], 0.9):.1f} NM")

    # By speed
    init_gs = init_states[:, 3]
    fast = init_gs > 350
    slow = init_gs < 200
    med = (init_gs >= 200) & (init_gs <= 350)

    print(f"\nBy ground speed:")
    print(f"  Fast (>350kt, n={fast.sum()}): q50={np.quantile(final_pe[fast], 0.5):.1f}  q90={np.quantile(final_pe[fast], 0.9):.1f} NM")
    print(f"  Med  (200-350, n={med.sum()}): q50={np.quantile(final_pe[med], 0.5):.1f}  q90={np.quantile(final_pe[med], 0.9):.1f} NM")
    print(f"  Slow (<200kt, n={slow.sum()}): q50={np.quantile(final_pe[slow], 0.5):.1f}  q90={np.quantile(final_pe[slow], 0.9):.1f} NM")

    # Direction error analysis
    final_he = hdg_errors[:, -1]
    print(f"\nDirection error distribution:")
    print(f"  q10 (best): {np.quantile(final_he, 0.1):.1f} deg")
    print(f"  q50: {np.quantile(final_he, 0.5):.1f} deg")
    print(f"  q90: {np.quantile(final_he, 0.9):.1f} deg")
    print(f"  q99 (worst): {np.quantile(final_he, 0.99):.1f} deg")
    print(f"  Fraction hdg_err > 30°: {(final_he > 30).mean()*100:.0f}%")
    print(f"  Fraction hdg_err > 60°: {(final_he > 60).mean()*100:.0f}%")

    # Save results
    out_path = 'models/world_model/failure_analysis.json'
    summary = {
        'n_samples': int(N),
        'final_pos_err': {
            'q10': float(np.quantile(final_pe, 0.1)),
            'q50': float(np.quantile(final_pe, 0.5)),
            'q90': float(np.quantile(final_pe, 0.9)),
            'q99': float(np.quantile(final_pe, 0.99)),
        },
        'final_hdg_err': {
            'q50': float(np.quantile(final_he, 0.5)),
            'q90': float(np.quantile(final_he, 0.9)),
            'q99': float(np.quantile(final_he, 0.99)),
        },
        'coverage_120s': float(coverages[:, -1].mean()),
        'mc_spread_120s_nm': float(mc_spreads[:, -1].mean()),
        'hdg_err_gt_30_pct': float((final_he > 30).mean() * 100),
        'hdg_err_gt_60_pct': float((final_he > 60).mean() * 100),
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Analyze] Summary saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='models/world_model/best_model.pt')
    parser.add_argument('--data-dir', default='data/recordings')
    parser.add_argument('--n', type=int, default=500)
    args = parser.parse_args()
    analyze(args.model_path, args.data_dir, args.n)

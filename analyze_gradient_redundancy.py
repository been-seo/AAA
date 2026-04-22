"""
Inner task gradient redundancy 분석 (PAVING §8.9 확장)

heading 그룹의 track_sin, track_cos, heading_circ 3 task는 이론상
같은 각도 정보의 redundant 인코딩:
  - track_sin² + track_cos² = 1 (2차원이면 원 전체 표현 가능)
  - heading_circ = phase wrap 버전

따라서 이들의 gradient vector Gram 행렬은 rank-deficient해야 함.
구체적으로:
  1. heading 3 task 간 |cos(g_i, g_j)| 매우 큼 (≈ 1에 가까움)
  2. 이들로 이루어진 3x3 부분 Gram의 최소 eigenvalue가 다른 task
     pair 대비 1~2 order 작음

이를 측정하여 K 축소 정당화.

사용:
  python analyze_gradient_redundancy.py
  python analyze_gradient_redundancy.py --model-path models/world_model/best_model.pt
"""
import sys
import os
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.world_model.trajectory_predictor import TrajectoryPredictor
from ai.world_model.dataset import (
    TrajectoryDataset, STATE_DIM, NORM_MEAN, NORM_STD,
)
from ai.world_model.paving import InnerTaskManager


def analyze(model_path, data_dir='data/recordings', batch_size=256,
            num_batches=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Analyze] Device: {device}")
    print(f"[Analyze] Model: {model_path}")

    # Load model
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    args = ckpt.get('args', {})
    model = TrajectoryPredictor(
        hidden_dim=args.get('hidden_dim', 256),
        latent_dim=args.get('latent_dim', 64),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.train()  # gradient 계산 위해 train 모드
    print(f"[Analyze] Loaded (epoch={ckpt.get('epoch', '?')})")

    # Dataset
    ds = TrajectoryDataset(data_dir, past_steps=6, future_steps=12, stride=8)
    ds.preload()
    print(f"[Analyze] Dataset: {len(ds):,} samples")

    task_names = TrajectoryPredictor.INNER_TASKS
    K = len(task_names)
    itm = InnerTaskManager(task_names)

    # 여러 배치에 걸쳐 Gram 행렬 평균
    gram_accum = np.zeros((K, K))
    n_batches_done = 0

    np.random.seed(0)
    indices_all = np.random.permutation(len(ds))

    for b in range(num_batches):
        bi = indices_all[b * batch_size:(b + 1) * batch_size]
        if len(bi) == 0:
            break

        # 배치 조립
        past_list, future_list, ctx_list, world_list = [], [], [], []
        for idx in bi:
            items = ds[int(idx)]
            past_list.append(items[0])
            future_list.append(items[1])
            ctx_list.append(items[2])
            if len(items) > 5:
                world_list.append(items[5])

        past = torch.stack(past_list).to(device)
        future = torch.stack(future_list).to(device)
        ctx = torch.stack(ctx_list).to(device)
        world = torch.stack(world_list).to(device) if world_list else None

        # Forward + per-task losses
        losses = model.compute_loss(past, future, ctx, world=world)
        tl = losses['task_losses']

        # Per-task gradient Gram
        try:
            gram, _ = itm.compute_gram(model, tl)
            gram_accum += gram
            n_batches_done += 1
            print(f"[Analyze] Batch {b+1}/{num_batches} done")
        except Exception as e:
            print(f"[Analyze] Batch {b+1} failed: {e}")

    if n_batches_done == 0:
        print("[Analyze] No batches succeeded")
        return

    gram = gram_accum / n_batches_done
    print(f"\n[Analyze] Averaged over {n_batches_done} batches\n")

    # ═══════════════════════════════════════════
    # 1. Full K×K Gram (cosine)
    # ═══════════════════════════════════════════
    print("=" * 72)
    print("FULL GRAM MATRIX (cosine similarity between task gradients)")
    print("=" * 72)
    header = "       " + " ".join(f"{n[:6]:>7}" for n in task_names)
    print(header)
    for i, name in enumerate(task_names):
        row = " ".join(f"{gram[i,j]:+.3f}" for j in range(K))
        print(f"{name[:6]:>7} {row}")

    # ═══════════════════════════════════════════
    # 2. Heading trio 분석
    # ═══════════════════════════════════════════
    heading_tasks = ['track_sin', 'track_cos', 'heading_circ']
    heading_idx = [task_names.index(t) for t in heading_tasks if t in task_names]

    heading_sub = None
    heading_eig = None
    if len(heading_idx) == 3:
        print("\n" + "=" * 72)
        print("HEADING TRIO ANALYSIS (track_sin, track_cos, heading_circ)")
        print("=" * 72)
        heading_sub = gram[np.ix_(heading_idx, heading_idx)].copy()
        print(f"3×3 sub-Gram:")
        for i, name in enumerate(heading_tasks):
            print(f"  {name:>14} {heading_sub[i]}")

        heading_eig = np.linalg.eigvalsh(heading_sub)
        print(f"\n  Eigenvalues: {heading_eig}")
        print(f"  min λ:       {heading_eig.min():.4f}")
        print(f"  max λ:       {heading_eig.max():.4f}")
        print(f"  κ (condition): {heading_eig.max() / max(heading_eig.min(), 1e-9):.2e}")
        print(f"  rank(≥1e-3):   {(heading_eig > 1e-3).sum()}/3")

        # Pairwise |cos|
        print(f"\n  Pairwise |cos|:")
        for i in range(3):
            for j in range(i + 1, 3):
                c = abs(heading_sub[i, j])
                print(f"    |cos({heading_tasks[i]}, {heading_tasks[j]})| = {c:.3f}")

    # ═══════════════════════════════════════════
    # 3. 비교: 다른 모든 (i,j) pair의 최소 eigenvalue 분포
    # ═══════════════════════════════════════════
    print("\n" + "=" * 72)
    print("MIN EIGENVALUE OF 3×3 SUB-GRAM (heading trio vs random triples)")
    print("=" * 72)

    all_triples = []
    for i in range(K):
        for j in range(i + 1, K):
            for k in range(j + 1, K):
                sub = gram[np.ix_([i, j, k], [i, j, k])]
                eigs = np.linalg.eigvalsh(sub)
                all_triples.append((
                    (task_names[i], task_names[j], task_names[k]),
                    eigs.min(),
                    eigs.max(),
                ))

    # 정렬 (min eig 작은 순)
    all_triples.sort(key=lambda x: x[1])

    print(f"\nTop 10 most rank-deficient triples (by min eigenvalue):")
    print(f"{'Triple':<55} {'min λ':>10} {'max λ':>10}")
    for trip, mn, mx in all_triples[:10]:
        name = ', '.join(trip)
        marker = " ←heading" if set(trip) == set(heading_tasks) else ""
        print(f"  {name:<55} {mn:>10.4f} {mx:>10.4f}{marker}")

    # heading trio 랭킹
    heading_set = set(heading_tasks)
    for rank, (trip, mn, mx) in enumerate(all_triples):
        if set(trip) == heading_set:
            print(f"\nHeading trio rank: {rank + 1} / {len(all_triples)}")
            print(f"  min λ = {mn:.4f} (median of all triples: "
                  f"{np.median([t[1] for t in all_triples]):.4f})")
            break

    # ═══════════════════════════════════════════
    # 4. Pairwise |cos| 전체 분포
    # ═══════════════════════════════════════════
    print("\n" + "=" * 72)
    print("ALL PAIRWISE |cos| DISTRIBUTION")
    print("=" * 72)

    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            pairs.append(((task_names[i], task_names[j]), abs(gram[i, j])))
    pairs.sort(key=lambda x: -x[1])

    print(f"\nTop 10 most correlated pairs:")
    for (a, b), c in pairs[:10]:
        marker = ""
        if {a, b} <= heading_set:
            marker = " ←heading redundancy"
        print(f"  |cos({a:<15}, {b:<15})| = {c:.3f}{marker}")

    # ═══════════════════════════════════════════
    # 5. Overall Gram condition number (K×K)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 72)
    print(f"FULL K={K} GRAM CONDITION NUMBER")
    print("=" * 72)
    full_eigs = np.linalg.eigvalsh(gram)
    full_eigs_clipped = np.clip(full_eigs, 1e-9, None)
    print(f"  Eigenvalues: {full_eigs}")
    print(f"  σ_min: {full_eigs_clipped.min():.4f}")
    print(f"  σ_max: {full_eigs_clipped.max():.4f}")
    print(f"  κ(G):  {full_eigs_clipped.max() / full_eigs_clipped.min():.2f}")
    print(f"  (PAVING target κ ≈ 1 for orthogonal task decomposition)")

    # 사후 검증: PAVING τ=0.5 기준 위반 pair 체크
    print("\n" + "=" * 72)
    print("PAVING TAU=0.5 VIOLATIONS (|cos| > 0.5)")
    print("=" * 72)
    violations = [(p, c) for p, c in pairs if c > 0.5]
    if violations:
        for (a, b), c in violations:
            print(f"  ⚠ |cos({a}, {b})| = {c:.3f} > 0.5")
        print(f"  → K reduction needed or PAVING controller regroup")
    else:
        print(f"  ✓ All {len(pairs)} pairs within τ=0.5")
        print(f"  → PAVING controller safe to enable")

    # Save to JSON for paper
    import json
    out = {
        'model': model_path,
        'n_batches': n_batches_done,
        'task_names': task_names,
        'K': K,
        'gram': gram.tolist(),
        'full_gram_kappa': float(full_eigs_clipped.max() / full_eigs_clipped.min()),
        'tau_violations': [
            {'pair': list(p), 'abs_cos': float(c)} for p, c in violations
        ],
        'heading_trio': {
            'tasks': heading_tasks,
            'sub_gram': heading_sub.tolist() if heading_sub is not None else None,
            'eigenvalues': heading_eig.tolist() if heading_eig is not None else None,
        } if len(heading_idx) == 3 else None,
        'top_correlated_pairs': [
            {'pair': list(p), 'abs_cos': float(c)} for p, c in pairs[:10]
        ],
        'top_rank_deficient_triples': [
            {'triple': list(t), 'min_eig': float(m), 'max_eig': float(mx)}
            for t, m, mx in all_triples[:10]
        ],
    }
    out_path = 'models/world_model/gradient_redundancy.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[Analyze] Saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',
                        default='models/world_model/best_model.pt')
    parser.add_argument('--data-dir', default='data/recordings')
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    analyze(args.model_path, args.data_dir, args.batch_size, args.batches)

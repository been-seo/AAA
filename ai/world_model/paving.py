"""
PAVING MTL Controller for World Model training.

Implements Algorithm 2 from "Paving the Loss Landscape" (Seo 2026):
  - Monitor κ(G) = condition number of gradient Gram matrix
  - Monitor h = (1/K) Σ L_k² (squared-norm certificate)
  - Escalation on degradation:
      Level 1: LR halving           (symptom mitigation)
      Level 2: Task weight rebalance (Gram balancing)
      Level 3: Regroup correlated tasks (structural)

Periodic per-task gradient computation (every N steps) via K backward passes.
"""
import math
from collections import deque

import numpy as np
import torch


class InnerTaskManager:
    """
    Holds K inner tasks, computes per-task gradients on demand.

    Usage:
        itm = InnerTaskManager(task_names)
        # In training loop:
        task_losses = model.compute_loss(...)['task_losses']
        if step % compute_grad_every == 0:
            G = itm.compute_gram(model, task_losses)  # K x K gram matrix
    """
    def __init__(self, task_names):
        self.task_names = list(task_names)
        self.K = len(task_names)

    def compute_gram(self, model, task_losses, normalize=True):
        """
        Compute K×K Gram matrix of gradient cosines.

        Cost: K backward passes (expensive, call periodically).
        :return: gram matrix (K, K), gradient norms (K,)
        """
        # Collect shared parameters (for gradient comparison)
        params = [p for p in model.parameters() if p.requires_grad]

        grads = []  # list of K flat gradient vectors
        for k, name in enumerate(self.task_names):
            loss_k = task_losses[name]
            # Retain graph for all but last
            retain = (k < self.K - 1)
            g_list = torch.autograd.grad(
                loss_k, params, retain_graph=retain, allow_unused=True)
            # Flatten to single vector
            flat = torch.cat([
                g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                for g, p in zip(g_list, params)
            ])
            grads.append(flat)

        G = torch.zeros(self.K, self.K, device=grads[0].device)
        norms = torch.zeros(self.K, device=grads[0].device)
        for i in range(self.K):
            norms[i] = grads[i].norm() + 1e-12
        for i in range(self.K):
            for j in range(i, self.K):
                dot = (grads[i] * grads[j]).sum()
                if normalize:
                    G[i, j] = dot / (norms[i] * norms[j])
                else:
                    G[i, j] = dot
                G[j, i] = G[i, j]

        return G.cpu().numpy(), norms.cpu().numpy()

    def condition_number(self, gram):
        """κ(G) = σ_max / σ_min of gram matrix"""
        try:
            eigvals = np.linalg.eigvalsh(gram)
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvals.max() / eigvals.min()
        except Exception:
            return 1e6

    def max_cos_pair(self, gram):
        """Find (i, j) with max |cos|, i ≠ j"""
        K = gram.shape[0]
        max_abs_cos = -1
        best_pair = (0, 1)
        for i in range(K):
            for j in range(i + 1, K):
                c = abs(gram[i, j])
                if c > max_abs_cos:
                    max_abs_cos = c
                    best_pair = (i, j)
        return best_pair, max_abs_cos


class GroupingManager:
    """
    Manages task → group assignment, supports dynamic regrouping.

    Groups are named sets of task indices. Aggregation uses group sum.
    """
    def __init__(self, task_names, initial_groups):
        """
        :param task_names: list of K task names
        :param initial_groups: dict[group_name -> list of task names]
        """
        self.task_names = list(task_names)
        self.K = len(task_names)
        self.task_to_idx = {n: i for i, n in enumerate(task_names)}
        self.groups = {g: [self.task_to_idx[t] for t in tasks]
                       for g, tasks in initial_groups.items()}

    def merge(self, task_i_name, task_j_name):
        """Merge groups containing task_i and task_j."""
        gi = self._find_group(task_i_name)
        gj = self._find_group(task_j_name)
        if gi == gj or gi is None or gj is None:
            return False
        # Merge gj into gi
        self.groups[gi].extend(self.groups[gj])
        del self.groups[gj]
        return True

    def _find_group(self, task_name):
        idx = self.task_to_idx.get(task_name)
        if idx is None:
            return None
        for g, task_ids in self.groups.items():
            if idx in task_ids:
                return g
        return None

    def compute_weights_proportional(self, task_losses, w_min=0.05, w_max=5.0):
        """
        PAVING CTRL_w: proportional weighting w_k = K/(L_k · Σ L_j^-1)
        Preserves task-count balance (original objective proportions).
        Clamp to [w_min, w_max] to prevent pathological scales from
        dominating gradient budget.
        """
        # Normalize losses to [mean=1] first to reduce scale disparity
        raw = {k: max(float(task_losses[k]), 1e-6) for k in self.task_names}
        mean_loss = sum(raw.values()) / max(len(raw), 1)
        losses = {k: v / max(mean_loss, 1e-6) for k, v in raw.items()}
        inv_sum = sum(1.0 / v for v in losses.values())
        K = self.K
        weights = {k: K / (losses[k] * inv_sum) for k in self.task_names}
        # Clamp
        weights = {k: min(max(v, w_min), w_max) for k, v in weights.items()}
        return weights

    def summary(self):
        inv = {v: k for k, tasks in self.groups.items() for v in tasks}
        return {
            'n_groups': len(self.groups),
            'groups': {g: [self.task_names[i] for i in ids]
                       for g, ids in self.groups.items()},
        }


class CertificateController:
    """
    PAVING Algorithm 2: Certificate-Based Training Controller.

    Escalation levels:
      Level 1: LR halving        (O(1), symptom mitigation)
      Level 2: Weight rebalance  (O(K), Gram balancing)
      Level 3: Regroup           (O(K²d), structural)
    """
    def __init__(self, kappa_max=10.0, W_lr=3, W_alpha=3,
                 kappa_window=5):
        self.kappa_max = kappa_max
        self.W_lr = W_lr
        self.W_alpha = W_alpha
        self.persist = 0
        self.level = 0
        self.kappa_history = deque(maxlen=kappa_window)
        self.h_history = deque(maxlen=kappa_window)
        self._original_lr = None

    def step(self, kappa, h, task_losses, gram, task_names,
             current_lr=None, grouping_mgr=None, verbose=False):
        """
        Called periodically (e.g., every 100 train steps).

        :return: decision dict with actions to take
          {
            'action': 'none' | 'lr_cut' | 'rebalance' | 'regroup',
            'new_lr': float or None,
            'new_weights': dict or None,
            'merge_pair': (i, j) or None,
          }
        """
        self.kappa_history.append(kappa)
        self.h_history.append(h)

        # Trend checks
        kappa_trend = 0
        if len(self.kappa_history) >= 3:
            recent = list(self.kappa_history)
            kappa_trend = recent[-1] - recent[0]
        h_trend = 0
        if len(self.h_history) >= 3:
            recent = list(self.h_history)
            h_trend = recent[-1] - recent[0]

        decision = {
            'action': 'none',
            'new_lr': None,
            'new_weights': None,
            'merge_pair': None,
            'kappa': kappa, 'h': h,
            'persist': self.persist, 'level': self.level,
        }

        if kappa > self.kappa_max and (kappa_trend > 0 or h_trend > 0):
            self.persist += 1

            if self.persist <= self.W_lr:
                # Level 1: LR halving
                if current_lr is not None:
                    new_lr = current_lr * 0.5
                    decision['action'] = 'lr_cut'
                    decision['new_lr'] = new_lr
                    self.level = 1
            elif self.persist <= self.W_lr + self.W_alpha:
                # Level 2: Proportional weight rebalance
                if grouping_mgr is not None:
                    new_weights = grouping_mgr.compute_weights_proportional(
                        task_losses)
                    decision['action'] = 'rebalance'
                    decision['new_weights'] = new_weights
                    self.level = 2
            else:
                # Level 3: Regroup (merge most correlated pair)
                if gram is not None:
                    K = gram.shape[0]
                    max_c = -1
                    best = (0, 1)
                    for i in range(K):
                        for j in range(i + 1, K):
                            c = abs(gram[i, j])
                            if c > max_c:
                                max_c = c
                                best = (i, j)
                    decision['action'] = 'regroup'
                    decision['merge_pair'] = (
                        task_names[best[0]], task_names[best[1]])
                    decision['merge_cos'] = float(max_c)
                    self.persist = 0
                    self.level = 0
        elif kappa < self.kappa_max:
            # Healthy: recover if previously cut
            if self.persist > 0:
                self.persist = 0
                if self.level == 1 and self._original_lr is not None:
                    decision['action'] = 'lr_recover'
                    decision['new_lr'] = self._original_lr
                self.level = 0

        if verbose and decision['action'] != 'none':
            print(f"[PAVING] action={decision['action']} "
                  f"κ={kappa:.2f} h={h:.4f} "
                  f"persist={self.persist} level={self.level}")

        return decision


# ── Default grouping for WM inner tasks ──
DEFAULT_GROUPS = {
    'lateral_pos': ['pos_lat', 'pos_lon'],
    'vertical':    ['alt', 'vrate'],
    'dynamics':    ['gs', 'ias', 'mach'],
    'heading':     ['track_sin', 'track_cos', 'heading_circ'],
    'latent':      ['kl'],
}

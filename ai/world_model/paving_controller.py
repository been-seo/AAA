"""
PAVING Regroup Controller (Seo 2026)

Inner task gradient Gram 행렬을 EMA로 추적하고, 축 간 |cos|이 τ를 지속
초과하면 hierarchical clustering으로 inner task를 재그룹핑한다.

핵심:
  - measure(per_task_grads): per-task 그래디언트 벡터 받아 Gram EMA 갱신
  - check_and_regroup(): τ 위반 지속 시 새 그룹 dict 반환, 아니면 None
  - get_groups(): 현재 그룹핑 (외부에서 reward 집계용)

재그룹핑 알고리즘:
  1. D = 1 - |cos(G_ij)| 거리 행렬
  2. Scipy linkage (average) → dendrogram
  3. fcluster로 n_groups개 클러스터 절단
"""
import numpy as np


class PAVINGController:
    def __init__(self, inner_tasks, initial_groups, n_groups=3,
                 tau=0.5, measure_interval=100, violation_persist=5,
                 ema_alpha=0.1, cooldown=500):
        """
        :param inner_tasks: list of inner task names
        :param initial_groups: dict group_name → list of task names
        :param n_groups: target number of groups (clustering k)
        :param tau: CANON threshold (|cos| upper limit)
        :param measure_interval: steps between gradient measurements
        :param violation_persist: consecutive measurements required to regroup
        :param ema_alpha: Gram EMA update rate (0~1)
        :param cooldown: steps after a regroup during which another is blocked
        """
        self.inner_tasks = list(inner_tasks)
        self.K = len(inner_tasks)
        self.task_to_idx = {t: i for i, t in enumerate(self.inner_tasks)}
        self.n_groups = n_groups
        self.tau = tau
        self.measure_interval = measure_interval
        self.violation_persist = violation_persist
        self.ema_alpha = ema_alpha
        self.cooldown = cooldown

        # (K, K) identity (self-cos = 1)
        self.gram_ema = np.eye(self.K, dtype=np.float32)
        self.measurements = 0
        self.violation_streak = 0
        self.regroup_count = 0
        self.last_regroup_step = -cooldown  # allow first regroup after warmup
        self.groups = {g: list(tasks) for g, tasks in initial_groups.items()}
        self.group_names = list(initial_groups.keys())
        # task_name → group_name
        self.task_to_group = {}
        for g, tasks in self.groups.items():
            for t in tasks:
                self.task_to_group[t] = g
        self.history = []  # list of (step, max_inter_cos, regrouped)

    def measure(self, per_task_grads):
        """per_task_grads: dict[task_name] -> 1D tensor or np array."""
        if set(per_task_grads.keys()) != set(self.inner_tasks):
            missing = set(self.inner_tasks) - set(per_task_grads.keys())
            extra = set(per_task_grads.keys()) - set(self.inner_tasks)
            raise ValueError(f"grad keys mismatch. missing={missing} extra={extra}")

        # Build (K, D) matrix, normalize rows
        grads = []
        for t in self.inner_tasks:
            g = per_task_grads[t]
            if hasattr(g, 'detach'):
                g = g.detach().cpu().numpy()
            grads.append(np.asarray(g, dtype=np.float32))
        G = np.stack(grads, axis=0)  # (K, D)
        norms = np.linalg.norm(G, axis=1, keepdims=True).clip(min=1e-8)
        G_n = G / norms
        cos = G_n @ G_n.T  # (K, K), symmetric, diag=1

        alpha = self.ema_alpha
        self.gram_ema = (1 - alpha) * self.gram_ema + alpha * cos
        self.measurements += 1

    def max_inter_group_cos(self):
        """Return the largest |cos| between any two tasks in different groups."""
        max_c = 0.0
        pair = None
        for g1, t1s in self.groups.items():
            for g2, t2s in self.groups.items():
                if g1 >= g2:  # unordered pair
                    continue
                for a in t1s:
                    for b in t2s:
                        i = self.task_to_idx[a]
                        j = self.task_to_idx[b]
                        c = abs(float(self.gram_ema[i, j]))
                        if c > max_c:
                            max_c = c
                            pair = (a, b, g1, g2)
        return max_c, pair

    def check_and_regroup(self, current_step):
        """
        Call after each measure(). Returns new groups dict if regrouped else None.
        Logs violation streak and triggers on persist threshold.
        """
        if self.measurements < self.violation_persist:
            return None  # warmup

        if current_step - self.last_regroup_step < self.cooldown:
            return None  # cooldown block

        max_c, pair = self.max_inter_group_cos()
        self.history.append((current_step, max_c, False))

        if max_c > self.tau:
            self.violation_streak += 1
            if self.violation_streak >= self.violation_persist:
                new_groups = self._cluster()
                if self._groups_changed(new_groups):
                    old = {g: list(v) for g, v in self.groups.items()}
                    self.groups = new_groups
                    self.task_to_group = {}
                    for g, tasks in new_groups.items():
                        for t in tasks:
                            self.task_to_group[t] = g
                    self.violation_streak = 0
                    self.regroup_count += 1
                    self.last_regroup_step = current_step
                    # mark last history entry as regrouped
                    if self.history:
                        s, m, _ = self.history[-1]
                        self.history[-1] = (s, m, True)
                    return {'old': old, 'new': new_groups, 'trigger': pair,
                            'max_cos': max_c}
                # clustering returned same grouping — reset streak to avoid looping
                self.violation_streak = 0
        else:
            self.violation_streak = max(0, self.violation_streak - 1)
        return None

    def _cluster(self):
        """
        |cos| 유사도 기반 hierarchical clustering.
        거리 = 1 - |cos|, linkage=average, cut at n_groups.
        """
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
        except ImportError:
            return self._cluster_greedy()

        sim = np.abs(self.gram_ema)
        sim = np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(sim, 1.0)
        dist = 1.0 - sim
        # symmetric + zero diag
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0.0)
        try:
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method='average')
            labels = fcluster(Z, t=self.n_groups, criterion='maxclust')
        except Exception:
            return self._cluster_greedy()

        # Build groups dict, preserve group_names order by mean similarity
        clusters = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(self.inner_tasks[i])

        # Assign canonical group_names to clusters in order of smallest index
        sorted_labels = sorted(clusters.keys(),
                                key=lambda L: min(
                                    self.task_to_idx[t] for t in clusters[L]))
        new_groups = {}
        for rank, lab in enumerate(sorted_labels):
            name = self.group_names[rank] if rank < len(self.group_names) \
                else f'group_{rank}'
            new_groups[name] = clusters[lab]
        # Ensure all canonical names exist (if clustering produced fewer)
        for name in self.group_names:
            if name not in new_groups:
                new_groups[name] = []
        return new_groups

    def _cluster_greedy(self):
        """scipy 없을 때 fallback: 가장 가까운 쌍부터 병합."""
        K = self.K
        sim = np.abs(self.gram_ema).copy()
        np.fill_diagonal(sim, -1.0)
        clusters = [[i] for i in range(K)]
        while len(clusters) > self.n_groups:
            best = (-1, -1, -1.0)
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    s = max(sim[a, b] for a in clusters[i] for b in clusters[j])
                    if s > best[2]:
                        best = (i, j, s)
            i, j, _ = best
            clusters[i] = clusters[i] + clusters[j]
            del clusters[j]

        new_groups = {}
        sorted_c = sorted(clusters, key=lambda c: min(c))
        for rank, c in enumerate(sorted_c):
            name = self.group_names[rank] if rank < len(self.group_names) \
                else f'group_{rank}'
            new_groups[name] = [self.inner_tasks[i] for i in c]
        return new_groups

    def _groups_changed(self, new_groups):
        for g in new_groups:
            if g not in self.groups:
                return True
            if set(new_groups[g]) != set(self.groups[g]):
                return True
        return False

    def get_groups(self):
        return {g: list(v) for g, v in self.groups.items()}

    def summary(self):
        max_c, pair = self.max_inter_group_cos()
        return {
            'measurements': self.measurements,
            'regroup_count': self.regroup_count,
            'max_inter_cos': max_c,
            'max_inter_pair': pair,
            'violation_streak': self.violation_streak,
            'groups': self.get_groups(),
        }

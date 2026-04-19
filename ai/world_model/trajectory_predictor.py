"""
Delta 기반 항공기 궤적 예측 모델

핵심: 다음 상태의 절대 좌표가 아닌 Δ(변화량)를 직접 예측.
  - Δlat, Δlon, Δalt, Δgs, Δtrack, Δvrate, ...
  - 다음 상태 = 현재 상태 + Δ
  - 물리 clamp: 비현실적 Δ는 사후에 잘라냄
  - envelope/ratio/residual 없음 → 단순하고 빠름

구조:
  1. State Encoder: state(10D) → sin/cos 치환 12D → 128D
  2. Neighbor Attention: cross-attention으로 주변 항공기 interaction
  3. RSSM Core: GRU(deterministic) + VAE(stochastic)
  4. Delta Decoder: (hidden, z) → Δstate (unbounded)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .dataset import STATE_DIM, CONTEXT_DIM, MAX_NEIGHBORS, NORM_MEAN, NORM_STD, WORLD_DIM

# 물리 clamp 한계 (10초 간격 기준, 비정규화 공간)
# [Δlat_max, Δlon_max, Δalt_max, Δgs_max, Δtrack_max, Δvrate_max, ...]
_DELTA_CLAMP_RAW = np.array([
    0.05,    # lat: ~3NM
    0.05,    # lon: ~3NM
    3333,    # alt: 20000fpm * 10s
    200,     # gs: ±200kt
    180,     # track: ±180° (전방위)
    10000,   # vrate: ±10000fpm
    200,     # ias
    0.3,     # mach
    360,     # wind_dir
    60,      # wind_spd
], dtype=np.float32)

# 정규화 공간에서의 clamp
_DELTA_CLAMP_NORM = _DELTA_CLAMP_RAW / NORM_STD


class NeighborAttention(nn.Module):
    """주변 항공기와의 interaction을 cross-attention으로 모델링"""

    def __init__(self, query_dim, context_dim=CONTEXT_DIM, num_heads=2, out_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.q_proj = nn.Linear(query_dim, out_dim)
        self.k_proj = nn.Linear(context_dim, out_dim)
        self.v_proj = nn.Linear(context_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, query, context):
        B, N, _ = context.shape
        Q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        ctx_norm = context.norm(dim=-1)
        mask = (ctx_norm < 1e-6).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, -1)
        out = self.out_proj(out)
        return self.layer_norm(out)


class TrajectoryPredictor(nn.Module):
    """
    Delta 기반 RSSM 궤적 예측 모델

    디코더: Δstate를 직접 출력 (정규화 공간).
    다음 상태 = 현재 상태 + Δ, 물리 clamp 적용.
    """

    def __init__(self, hidden_dim=256, latent_dim=64, interaction_dim=64,
                 num_gru_layers=2, dropout=0.1, use_world=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_world = use_world

        # State encoder (10D → sin/cos 치환 12D → 128D)
        self._track_mean = NORM_MEAN[4]
        self._track_std = NORM_STD[4]
        self._wind_mean = NORM_MEAN[8]
        self._wind_std = NORM_STD[8]
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM - 2 + 4, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
        )

        # Neighbor interaction
        self.neighbor_attn = NeighborAttention(
            query_dim=128, context_dim=CONTEXT_DIM,
            num_heads=2, out_dim=interaction_dim)

        # World context encoder
        if use_world:
            self.world_encoder = nn.Sequential(
                nn.Linear(WORLD_DIM, 64), nn.ELU(),
                nn.Linear(64, 64), nn.ELU(),
            )
        self.world_embed_dim = 64 if use_world else 0

        # GRU
        gru_input_dim = 128 + interaction_dim + latent_dim + self.world_embed_dim
        self.gru = nn.GRU(gru_input_dim, hidden_dim,
                          num_layers=num_gru_layers,
                          batch_first=True, dropout=dropout if num_gru_layers > 1 else 0)

        # Posterior & Prior
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + 128, 256), nn.ELU(),
            nn.Linear(256, latent_dim * 2),
        )
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ELU(),
            nn.Linear(256, latent_dim * 2),
        )

        # Delta decoder: (hidden, z) → Δstate (정규화 공간)
        self.delta_decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, STATE_DIM),
        )

        # Normalization params
        self.register_buffer('norm_mean', torch.from_numpy(NORM_MEAN))
        self.register_buffer('norm_std', torch.from_numpy(NORM_STD))
        self.register_buffer('delta_clamp', torch.from_numpy(_DELTA_CLAMP_NORM))

    # ── Encoder helpers ──

    def _deg_to_sincos(self, state):
        track_deg = state[:, 4:5] * self._track_std + self._track_mean
        wind_deg = state[:, 8:9] * self._wind_std + self._wind_mean
        track_rad = track_deg * (3.14159265 / 180.0)
        wind_rad = wind_deg * (3.14159265 / 180.0)
        return torch.cat([
            state[:, :4],
            torch.sin(track_rad), torch.cos(track_rad),
            state[:, 5:8],
            torch.sin(wind_rad), torch.cos(wind_rad),
            state[:, 9:],
        ], dim=-1)

    def _encode_state(self, state):
        return self.state_encoder(self._deg_to_sincos(state))

    def _encode_world(self, world_feat):
        if not self.use_world or world_feat is None:
            return None
        return self.world_encoder(world_feat)

    def _build_gru_input(self, state_emb, interaction, z, world_emb=None):
        parts = [state_emb, interaction, z]
        if self.use_world:
            if world_emb is None:
                world_emb = torch.zeros(state_emb.shape[0], self.world_embed_dim,
                                        device=state_emb.device)
            parts.append(world_emb)
        return torch.cat(parts, dim=-1)

    def _posterior(self, hidden, state_emb):
        params = self.posterior_net(torch.cat([hidden, state_emb], dim=-1))
        mean, log_std = params.chunk(2, dim=-1)
        return mean, log_std.clamp(-5, 2)

    def _prior(self, hidden):
        params = self.prior_net(hidden)
        mean, log_std = params.chunk(2, dim=-1)
        return mean, log_std.clamp(-5, 2)

    def _sample_z(self, mean, log_std):
        return mean + log_std.exp() * torch.randn_like(log_std)

    # ── Delta decoder ──

    def _decode_delta(self, hidden, z):
        """(hidden, z) → Δstate (정규화 공간, clamped)"""
        delta = self.delta_decoder(torch.cat([hidden, z], dim=-1))
        return delta.clamp(-self.delta_clamp, self.delta_clamp)

    # ── Forward (학습) ──

    def forward(self, past_states, future_states, contexts, world=None):
        """
        학습 시: posterior로 z 샘플링, delta 예측.
        모든 입력/출력은 정규화 공간.

        :param past_states: (B, K, STATE_DIM) 정규화
        :param future_states: (B, N, STATE_DIM) 정규화
        :param contexts: (B, K+N, MAX_NEIGHBORS, CONTEXT_DIM)
        :param world: (B, K+N, WORLD_DIM) optional
        """
        B, K, _ = past_states.shape
        N = future_states.shape[1]
        T = K + N
        device = past_states.device

        all_states = torch.cat([past_states, future_states], dim=1)

        h = torch.zeros(self.gru.num_layers, B, self.hidden_dim, device=device)
        z = torch.zeros(B, self.latent_dim, device=device)

        pred_deltas = []
        target_deltas = []
        kl_losses = []

        for t in range(T):
            state_t = all_states[:, t]
            ctx_t = contexts[:, t]

            state_emb = self._encode_state(state_t)
            interaction = self.neighbor_attn(state_emb, ctx_t)
            world_emb = None
            if world is not None:
                world_emb = self._encode_world(world[:, t])

            gru_in = self._build_gru_input(state_emb, interaction, z, world_emb).unsqueeze(1)
            gru_out, h = self.gru(gru_in, h)
            h_t = gru_out.squeeze(1)

            post_mean, post_logstd = self._posterior(h_t, state_emb)
            prior_mean, prior_logstd = self._prior(h_t)
            kl_losses.append(self._kl_divergence(post_mean, post_logstd,
                                                  prior_mean, prior_logstd))
            z = self._sample_z(post_mean, post_logstd)

            pred_delta = self._decode_delta(h_t, z)
            pred_deltas.append(pred_delta)

            # Target delta: state[t+1] - state[t] (정규화 공간)
            if t < T - 1:
                tgt_delta = all_states[:, t + 1] - all_states[:, t]
                target_deltas.append(tgt_delta)

        pred_deltas = torch.stack(pred_deltas, dim=1)      # (B, T, D)
        target_deltas = torch.stack(target_deltas, dim=1)    # (B, T-1, D)
        kl_losses = torch.stack(kl_losses, dim=1)

        return {
            'pred_deltas': pred_deltas[:, :-1],    # (B, T-1, D)
            'target_deltas': target_deltas,          # (B, T-1, D)
            'kl_loss': kl_losses,
        }

    # ── Predict (추론) ──

    @torch.no_grad()
    def predict(self, past_states, contexts_past,
                num_samples=50, future_steps=12, world_past=None):
        """
        추론 시: prior로 미래 궤적을 MC 샘플링.

        :param past_states: (B, K, STATE_DIM) 정규화
        :param contexts_past: (B, K, MAX_NEIGHBORS, CONTEXT_DIM)
        :return: (B, num_samples, future_steps, STATE_DIM) 비정규화 예측 궤적
        """
        B, K, _ = past_states.shape
        device = past_states.device

        h = torch.zeros(self.gru.num_layers, B, self.hidden_dim, device=device)
        z = torch.zeros(B, self.latent_dim, device=device)

        for t in range(K):
            state_t = past_states[:, t]
            ctx_t = contexts_past[:, t]
            state_emb = self._encode_state(state_t)
            interaction = self.neighbor_attn(state_emb, ctx_t)
            world_emb = None
            if world_past is not None:
                world_emb = self._encode_world(world_past[:, t])
            gru_in = self._build_gru_input(state_emb, interaction, z, world_emb).unsqueeze(1)
            gru_out, h = self.gru(gru_in, h)
            h_t = gru_out.squeeze(1)
            post_mean, post_logstd = self._posterior(h_t, state_emb)
            z = self._sample_z(post_mean, post_logstd)

        # MC 확장
        h_exp = h.unsqueeze(2).expand(-1, -1, num_samples, -1).reshape(
            self.gru.num_layers, B * num_samples, self.hidden_dim).contiguous()
        z_exp = z.unsqueeze(1).expand(-1, num_samples, -1).reshape(
            B * num_samples, self.latent_dim)

        current_state = past_states[:, -1].unsqueeze(1).expand(-1, num_samples, -1).reshape(
            B * num_samples, STATE_DIM)

        empty_ctx = torch.zeros(B * num_samples, MAX_NEIGHBORS, CONTEXT_DIM, device=device)

        trajectories = []

        for t in range(future_steps):
            state_emb = self._encode_state(current_state)
            interaction = self.neighbor_attn(state_emb, empty_ctx)
            gru_in = self._build_gru_input(state_emb, interaction, z_exp).unsqueeze(1)
            gru_out, h_exp = self.gru(gru_in, h_exp)
            h_t = gru_out.squeeze(1)

            prior_mean, prior_logstd = self._prior(h_t)
            z_exp = self._sample_z(prior_mean, prior_logstd)

            pred_delta = self._decode_delta(h_t, z_exp)
            current_state = current_state + pred_delta

            trajectories.append(current_state)

        traj = torch.stack(trajectories, dim=1)  # (B*S, future_steps, D)
        traj = traj.view(B, num_samples, future_steps, STATE_DIM)

        # 비정규화
        traj = traj * self.norm_std + self.norm_mean

        return traj

    # ── Loss ──

    @staticmethod
    def _kl_divergence(mean1, logstd1, mean2, logstd2):
        var1 = (2 * logstd1).exp()
        var2 = (2 * logstd2).exp()
        kl = 0.5 * (var1 / var2 + (mean2 - mean1).pow(2) / var2 - 1 + 2 * (logstd2 - logstd1))
        return kl.sum(dim=-1)

    def compute_loss(self, past_states, future_states, contexts,
                     kl_weight=0.1, free_nats_per_step=1.0, world=None):
        """
        학습 loss: delta MSE + KL.
        past_raw/future_raw 불필요 — 전부 정규화 공간에서 처리.
        """
        output = self.forward(past_states, future_states, contexts, world=world)

        pred = output['pred_deltas']       # (B, T-1, D)
        target = output['target_deltas']   # (B, T-1, D)

        dim_weights = torch.ones(STATE_DIM, device=past_states.device)
        dim_weights[0] = 3.0  # lat
        dim_weights[1] = 3.0  # lon
        dim_weights[2] = 2.0  # alt
        dim_weights[3] = 1.5  # gs
        dim_weights[4] = 1.5  # track

        delta_err = (pred - target).pow(2) * dim_weights
        recon_loss = delta_err.sum(dim=-1).mean()

        K = past_states.shape[1]
        kl_per_step = output['kl_loss'].mean(dim=0)
        T = kl_per_step.shape[0]
        kl = torch.clamp(kl_per_step.sum() - free_nats_per_step * T, min=0)

        total_loss = recon_loss + kl_weight * kl

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl,
        }

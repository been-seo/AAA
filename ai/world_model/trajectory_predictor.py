"""
RSSM 기반 항공기 궤적 예측 모델

Dreamer-v3 영감:
- Deterministic path (GRU) + Stochastic latent (Gaussian)
- 주변 항공기 interaction을 cross-attention으로 처리
- 출력: 미래 상태의 mean + std (확률적 예측)

입력: 과거 K스텝 상태 + 주변 항공기 context
출력: 미래 N스텝 상태 분포 (mean, log_std)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import STATE_DIM, CONTEXT_DIM, MAX_NEIGHBORS, NORM_MEAN, NORM_STD


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
        """
        :param query: (batch, query_dim) — 현재 항공기 latent
        :param context: (batch, max_neighbors, context_dim) — 주변 항공기
        :return: (batch, out_dim)
        """
        B, N, _ = context.shape

        Q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 빈 이웃 마스킹 (context가 0벡터이면 무시)
        ctx_norm = context.norm(dim=-1)  # (B, N)
        mask = (ctx_norm < 1e-6).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        scores = scores.masked_fill(mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, -1)
        out = self.out_proj(out)
        return self.layer_norm(out)


class TrajectoryPredictor(nn.Module):
    """
    RSSM 스타일 궤적 예측 모델

    Architecture:
    1. State Encoder: state(10d) → embedding(128d)
    2. Context Encoder: neighbor attention → interaction(64d)
    3. RSSM Core:
       - Deterministic: GRU(embedding + interaction → hidden)
       - Stochastic: hidden → (z_mean, z_std)
       - Prior: hidden → (z_mean_prior, z_std_prior)
    4. Decoder: (hidden, z) → state_mean, state_log_std
    """

    def __init__(self, hidden_dim=256, latent_dim=64, interaction_dim=64,
                 num_gru_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
        )

        # Neighbor interaction
        self.neighbor_attn = NeighborAttention(
            query_dim=128, context_dim=CONTEXT_DIM,
            num_heads=2, out_dim=interaction_dim)

        # GRU (deterministic path)
        gru_input_dim = 128 + interaction_dim + latent_dim
        self.gru = nn.GRU(gru_input_dim, hidden_dim,
                          num_layers=num_gru_layers,
                          batch_first=True, dropout=dropout if num_gru_layers > 1 else 0)

        # Posterior: (hidden, state_emb) → z
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + 128, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim * 2),  # mean + log_std
        )

        # Prior: hidden → z (for prediction / KL)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim * 2),
        )

        # State decoder: (hidden, z) → state prediction
        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.state_mean_head = nn.Linear(128, STATE_DIM)
        self.state_logstd_head = nn.Linear(128, STATE_DIM)

        # Normalization params (registered as buffers for device handling)
        self.register_buffer('norm_mean', torch.from_numpy(NORM_MEAN))
        self.register_buffer('norm_std', torch.from_numpy(NORM_STD))

    def _encode_state(self, state):
        """state (B, STATE_DIM) → embedding (B, 128)"""
        return self.state_encoder(state)

    def _posterior(self, hidden, state_emb):
        """Posterior q(z|h, x): 관측이 있을 때의 latent"""
        inp = torch.cat([hidden, state_emb], dim=-1)
        params = self.posterior_net(inp)
        mean, log_std = params.chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        return mean, log_std

    def _prior(self, hidden):
        """Prior p(z|h): 관측 없이 hidden만으로 예측"""
        params = self.prior_net(hidden)
        mean, log_std = params.chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        return mean, log_std

    def _sample_z(self, mean, log_std):
        """Reparameterization trick"""
        std = log_std.exp()
        eps = torch.randn_like(std)
        return mean + std * eps

    def _decode_state(self, hidden, z):
        """(hidden, z) → state distribution"""
        inp = torch.cat([hidden, z], dim=-1)
        feat = self.state_decoder(inp)
        mean = self.state_mean_head(feat)
        log_std = self.state_logstd_head(feat).clamp(-5, 2)
        return mean, log_std

    def forward(self, past_states, future_states, contexts):
        """
        학습 시: posterior로 z를 샘플링하면서 전체 시퀀스 처리

        :param past_states: (B, K, STATE_DIM) 정규화된 과거 상태
        :param future_states: (B, N, STATE_DIM) 정규화된 미래 상태
        :param contexts: (B, K+N, MAX_NEIGHBORS, CONTEXT_DIM)
        :return: dict with predictions, KL losses
        """
        B, K, _ = past_states.shape
        N = future_states.shape[1]
        T = K + N
        device = past_states.device

        all_states = torch.cat([past_states, future_states], dim=1)  # (B, T, D)

        # Initialize
        h = torch.zeros(self.gru.num_layers, B, self.hidden_dim, device=device)
        z = torch.zeros(B, self.latent_dim, device=device)

        pred_means = []
        pred_logstds = []
        kl_losses = []

        for t in range(T):
            state_t = all_states[:, t]  # (B, STATE_DIM)
            ctx_t = contexts[:, t]  # (B, MAX_NEIGHBORS, CONTEXT_DIM)

            # Encode
            state_emb = self._encode_state(state_t)  # (B, 128)
            interaction = self.neighbor_attn(state_emb, ctx_t)  # (B, 64)

            # GRU input
            gru_in = torch.cat([state_emb, interaction, z], dim=-1).unsqueeze(1)
            gru_out, h = self.gru(gru_in, h)
            h_t = gru_out.squeeze(1)  # (B, hidden_dim)

            # Posterior & Prior
            post_mean, post_logstd = self._posterior(h_t, state_emb)
            prior_mean, prior_logstd = self._prior(h_t)

            # KL divergence
            kl = self._kl_divergence(post_mean, post_logstd, prior_mean, prior_logstd)
            kl_losses.append(kl)

            # Sample z from posterior (training)
            z = self._sample_z(post_mean, post_logstd)

            # Decode
            pred_mean, pred_logstd = self._decode_state(h_t, z)
            pred_means.append(pred_mean)
            pred_logstds.append(pred_logstd)

        pred_means = torch.stack(pred_means, dim=1)  # (B, T, STATE_DIM)
        pred_logstds = torch.stack(pred_logstds, dim=1)
        kl_losses = torch.stack(kl_losses, dim=1)  # (B, T)

        return {
            'pred_mean': pred_means,        # (B, T, STATE_DIM)
            'pred_logstd': pred_logstds,    # (B, T, STATE_DIM)
            'kl_loss': kl_losses,           # (B, T)
            'future_pred_mean': pred_means[:, K:],   # (B, N, STATE_DIM)
            'future_pred_logstd': pred_logstds[:, K:],
        }

    @torch.no_grad()
    def predict(self, past_states, contexts_past, contexts_future_dummy=None,
                num_samples=50, future_steps=12):
        """
        추론 시: prior로 미래 궤적을 Monte Carlo 샘플링

        :param past_states: (B, K, STATE_DIM) 정규화된 과거 상태
        :param contexts_past: (B, K, MAX_NEIGHBORS, CONTEXT_DIM)
        :param num_samples: Monte Carlo 샘플 수
        :param future_steps: 예측할 미래 스텝 수
        :return: (B, num_samples, future_steps, STATE_DIM) 비정규화된 예측 궤적
        """
        B, K, _ = past_states.shape
        device = past_states.device

        # 과거 시퀀스 인코딩 (posterior 사용)
        h = torch.zeros(self.gru.num_layers, B, self.hidden_dim, device=device)
        z = torch.zeros(B, self.latent_dim, device=device)

        for t in range(K):
            state_t = past_states[:, t]
            ctx_t = contexts_past[:, t]
            state_emb = self._encode_state(state_t)
            interaction = self.neighbor_attn(state_emb, ctx_t)
            gru_in = torch.cat([state_emb, interaction, z], dim=-1).unsqueeze(1)
            gru_out, h = self.gru(gru_in, h)
            h_t = gru_out.squeeze(1)
            post_mean, post_logstd = self._posterior(h_t, state_emb)
            z = self._sample_z(post_mean, post_logstd)

        # 마지막 상태
        last_state = past_states[:, -1]  # (B, STATE_DIM)

        # Monte Carlo 미래 예측
        # h, z를 num_samples로 복제
        h_expanded = h.unsqueeze(2).expand(-1, -1, num_samples, -1).reshape(
            self.gru.num_layers, B * num_samples, self.hidden_dim).contiguous()
        z_expanded = z.unsqueeze(1).expand(-1, num_samples, -1).reshape(
            B * num_samples, self.latent_dim)
        last_pred = last_state.unsqueeze(1).expand(-1, num_samples, -1).reshape(
            B * num_samples, STATE_DIM)

        # 빈 context (미래는 context 불확실 → 0)
        empty_ctx = torch.zeros(B * num_samples, MAX_NEIGHBORS, CONTEXT_DIM, device=device)

        trajectories = []
        for t in range(future_steps):
            state_emb = self._encode_state(last_pred)
            interaction = self.neighbor_attn(state_emb, empty_ctx)
            gru_in = torch.cat([state_emb, interaction, z_expanded], dim=-1).unsqueeze(1)
            gru_out, h_expanded = self.gru(gru_in, h_expanded)
            h_t = gru_out.squeeze(1)

            # Prior에서 z 샘플링
            prior_mean, prior_logstd = self._prior(h_t)
            z_expanded = self._sample_z(prior_mean, prior_logstd)

            # Decode
            pred_mean, pred_logstd = self._decode_state(h_t, z_expanded)
            # 샘플링 (aleatoric uncertainty)
            pred_state = self._sample_z(pred_mean, pred_logstd)
            trajectories.append(pred_state)
            last_pred = pred_state

        # (future_steps, B*num_samples, STATE_DIM) → (B, num_samples, future_steps, STATE_DIM)
        traj = torch.stack(trajectories, dim=1)  # (B*S, T, D)
        traj = traj.view(B, num_samples, future_steps, STATE_DIM)

        # 비정규화
        traj = traj * self.norm_std + self.norm_mean

        return traj

    @staticmethod
    def _kl_divergence(mean1, logstd1, mean2, logstd2):
        """두 가우시안 분포 사이의 KL divergence (배치별 합산)"""
        var1 = (2 * logstd1).exp()
        var2 = (2 * logstd2).exp()
        kl = 0.5 * (var1 / var2 + (mean2 - mean1).pow(2) / var2 - 1 + 2 * (logstd2 - logstd1))
        return kl.sum(dim=-1)  # (B,) — latent_dim 차원 합산

    def compute_loss(self, past_states, future_states, contexts, future_raw,
                     kl_weight=0.1, free_nats=3.0):
        """
        학습 손실 계산

        :param future_raw: (B, N, STATE_DIM) 비정규화된 미래 상태 (NM, ft 단위)
        :param kl_weight: KL 손실 가중치
        :param free_nats: KL free nats (Dreamer-v3)
        :return: dict with total_loss, recon_loss, kl_loss
        """
        output = self.forward(past_states, future_states, contexts)

        # 미래 예측의 reconstruction loss (정규화 공간에서)
        pred_mean = output['future_pred_mean']  # (B, N, D)
        pred_logstd = output['future_pred_logstd']

        # Gaussian NLL
        target = future_states  # 정규화된 미래 상태
        var = (2 * pred_logstd).exp()
        nll = 0.5 * ((target - pred_mean).pow(2) / var + 2 * pred_logstd + math.log(2 * math.pi))

        # 위치(lat, lon)와 고도에 가중치
        dim_weights = torch.ones(STATE_DIM, device=past_states.device)
        dim_weights[0] = 3.0  # lat
        dim_weights[1] = 3.0  # lon
        dim_weights[2] = 2.0  # alt
        dim_weights[3] = 1.5  # gs
        dim_weights[4] = 1.5  # track

        recon_loss = (nll * dim_weights).sum(dim=-1).mean()

        # KL loss (free nats)
        kl = output['kl_loss'].mean(dim=0).sum()  # 시퀀스 평균 → 타임스텝 합산
        kl = torch.clamp(kl - free_nats, min=0)

        total_loss = recon_loss + kl_weight * kl

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl,
        }

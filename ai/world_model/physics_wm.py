"""
Physics-based World Model (PhysicsWM)

설계 철학:
  11차원 state matching이 아니라, 항공기 역학 자체를 역추론한다.

  관측: ADS-B (lat, lon, alt, gs, track, vrate)
  역추론 (past history → 수치 미분):
    ω_z = Δtrack/Δt       (yaw rate, rad/s)
    a_x = Δgs/Δt          (longitudinal accel, kt/s)
    a_z = Δvrate/Δt       (vertical accel, fpm/s)
    bank = atan(v·ω_z/g)  (coordinated turn, radians)
    pitch = atan(vrate/v) (radians)

  예측 대상:
    다음 시점의 (ω_z, a_x, a_z) — "조종사가 얼마나 응답할지"
    → forward dynamics로 state를 물리적으로 적분

  이점:
    - VAE 없이 직접 deterministic regression
    - 물리적 제약이 구조에 내장됨 (bank 한계, 상승률 한계 등)
    - 예측 궤적이 항상 물리적으로 유효
    - z-collapse 같은 VAE 병리 없음
"""
import math

import torch
import torch.nn as nn
import numpy as np

from .dataset import (
    STATE_DIM, NORM_MEAN, NORM_STD, CONTEXT_DIM, MAX_NEIGHBORS, WORLD_DIM,
)


# 물리 상수
G_MS2 = 9.81                  # m/s²
KT_TO_MS = 0.514444           # knots → m/s
FPM_TO_MS = 0.00508           # ft/min → m/s
DT_SEC = 10.0                 # ADS-B step


def extract_physics_rates(raw_states, dt=DT_SEC):
    """
    과거 raw state 시퀀스에서 역학량을 수치 미분.

    :param raw_states: (B, K, STATE_DIM) 비정규화
    :return: dict of rate tensors, each (B, K-1)
      omega_z: yaw rate (rad/s) — 순회 보정 적용
      a_x:     longitudinal accel (kt/s)
      a_z:     vertical accel (fpm/s)
      bank:    bank angle (rad) — coordinated turn 추정
      pitch:   pitch angle (rad) — climb angle 추정
    """
    # track (deg), wrap-around 처리
    track = raw_states[:, :, 4]  # (B, K)
    dtrack_deg = track[:, 1:] - track[:, :-1]
    dtrack_deg = (dtrack_deg + 180.0) % 360.0 - 180.0  # (-180, 180]
    omega_z = torch.deg2rad(dtrack_deg) / dt  # (B, K-1) rad/s

    # longitudinal accel (kt/s)
    gs = raw_states[:, :, 3]
    a_x = (gs[:, 1:] - gs[:, :-1]) / dt

    # vertical accel (fpm/s)
    vrate = raw_states[:, :, 5]
    a_z = (vrate[:, 1:] - vrate[:, :-1]) / dt

    # bank: coordinated turn. v(m/s) · ω(rad/s) / g = tan(bank)
    v_ms = gs[:, 1:] * KT_TO_MS
    bank = torch.atan(v_ms * omega_z / G_MS2)  # rad, can be ±

    # pitch: climb angle 근사. sin(pitch) = vrate/v
    vrate_ms = vrate[:, 1:] * FPM_TO_MS
    v_total = torch.sqrt(v_ms.pow(2) + vrate_ms.pow(2)).clamp(min=1.0)
    pitch = torch.asin((vrate_ms / v_total).clamp(-0.99, 0.99))  # rad

    return {
        'omega_z': omega_z,
        'a_x': a_x,
        'a_z': a_z,
        'bank': bank,
        'pitch': pitch,
    }


def forward_dynamics(state, rates, dt=DT_SEC):
    """
    물리 적분: (state, rates) → next_state.

    :param state: (B, STATE_DIM) raw 현재 상태
    :param rates: dict with 'omega_z' (B,) in rad/s, 'a_x' (B,) kt/s, 'a_z' (B,) fpm/s
    :return: next_state (B, STATE_DIM) raw
    """
    next_state = state.clone()
    lat = state[:, 0]
    lon = state[:, 1]
    alt = state[:, 2]
    gs = state[:, 3]
    track = state[:, 4]
    vrate = state[:, 5]

    omega_z = rates['omega_z']  # rad/s
    a_x = rates['a_x']          # kt/s
    a_z = rates['a_z']          # fpm/s

    # 물리 적분
    new_gs = (gs + a_x * dt).clamp(min=50.0, max=800.0)
    new_vrate = (vrate + a_z * dt).clamp(min=-10000.0, max=10000.0)
    new_track = (track + torch.rad2deg(omega_z) * dt) % 360.0
    new_alt = (alt + vrate * dt / 60.0).clamp(min=0.0, max=60000.0)  # vrate fpm → dt초 후 변화

    # 위치: 평균 속도/방향으로 이동
    avg_gs = (gs + new_gs) / 2.0
    avg_track_rad = torch.deg2rad((track + new_track) / 2.0)
    dist_nm = avg_gs * dt / 3600.0  # kt × s / 3600 = NM

    new_lat = lat + dist_nm * torch.cos(avg_track_rad) / 60.0
    cos_lat = torch.cos(torch.deg2rad(new_lat)).clamp(min=0.1)
    new_lon = lon + dist_nm * torch.sin(avg_track_rad) / (60.0 * cos_lat)

    next_state[:, 0] = new_lat
    next_state[:, 1] = new_lon
    next_state[:, 2] = new_alt
    next_state[:, 3] = new_gs
    next_state[:, 4] = new_track
    next_state[:, 5] = new_vrate
    # ias, mach, wind는 persistent (간단 가정)
    return next_state


class PhysicsWM(nn.Module):
    """
    물리 기반 World Model.

    구조:
      1. Encoder: past kinematic + past rates + context → hidden
      2. Rate predictor (GRU): hidden → future (ω_z, a_x, a_z) per step
      3. Forward dynamics (non-learnable): rates → state integration

    Loss: predicted rate vs ground-truth rate (target_state에서 역추출)
    """
    INNER_TASKS = ['rate_omega_z', 'rate_a_x', 'rate_a_z']

    def __init__(self, hidden_dim=256, use_world=True, past_steps=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.past_steps = past_steps
        self.world_dim = WORLD_DIM if use_world else 0

        # Kinematic encoder (past state sequence, normalized)
        self.kinematic_enc = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
        )

        # Rate encoder (ω_z, a_x, a_z, bank, pitch)
        self.rate_enc = nn.Sequential(
            nn.Linear(5, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
        )

        # Context (neighbors) encoder
        self.neighbor_enc = nn.Sequential(
            nn.Linear(CONTEXT_DIM, 32), nn.ELU(),
        )
        self.neighbor_pool = nn.Sequential(
            nn.Linear(32, 64), nn.ELU(),
        )

        # World (waypoint) encoder
        if use_world:
            self.world_enc = nn.Sequential(
                nn.Linear(WORLD_DIM, 64), nn.ELU(),
            )
            world_out = 64
        else:
            self.world_enc = None
            world_out = 0

        # GRU over past sequence
        gru_in = 128 + 64 + 64 + world_out  # kine + rate + neighbor + world
        self.gru = nn.GRU(gru_in, hidden_dim, num_layers=2,
                           batch_first=True, dropout=0.1)

        # Future rate predictor
        # Input: hidden + last state + last rates → next rates
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim + STATE_DIM + 5, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 3),  # (omega_z, a_x, a_z)
        )

        # Rate uncertainty head (optional per-dim logstd)
        self.rate_logstd_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ELU(),
            nn.Linear(64, 3),
        )

        # Normalization buffers
        self.register_buffer('norm_mean', torch.from_numpy(NORM_MEAN).float())
        self.register_buffer('norm_std', torch.from_numpy(NORM_STD).float())

        # Rate normalization (추정치, 데이터셋 통계 기반)
        # omega_z: ±3°/s = ±0.052 rad/s
        # a_x: ±2 kt/s
        # a_z: ±100 fpm/s
        self.register_buffer('rate_scale',
            torch.tensor([0.05, 2.0, 100.0]).float())

    def _encode_past(self, past_norm, past_ctx, world=None):
        """
        과거 K-step을 GRU로 인코딩.

        :param past_norm: (B, K, STATE_DIM) 정규화
        :param past_ctx: (B, T_total, MAX_N, CTX_DIM) or (B, K, MAX_N, CTX_DIM)
        :param world: (B, T_total, WORLD_DIM) or None — past 부분만 사용
        :return: (B, hidden_dim), past rates dict, last raw state
        """
        B, K = past_norm.shape[:2]
        device = past_norm.device

        # ctx/world가 total_steps(K+T_future) 길이일 수 있음 → past 부분만 슬라이스
        if past_ctx.shape[1] != K:
            past_ctx = past_ctx[:, :K]
        if world is not None and world.shape[1] != K:
            world = world[:, :K]

        # Raw state (비정규화)
        past_raw = past_norm * self.norm_std + self.norm_mean
        # Past rates from raw (K-1 steps)
        past_rates = extract_physics_rates(past_raw)  # each (B, K-1)
        # Pad to K steps (첫 step은 rate 없음, zeros)
        zero_pad = torch.zeros(B, 1, device=device)
        rates_seq = torch.stack([
            torch.cat([zero_pad, past_rates['omega_z']], dim=1),
            torch.cat([zero_pad, past_rates['a_x']], dim=1),
            torch.cat([zero_pad, past_rates['a_z']], dim=1),
            torch.cat([zero_pad, past_rates['bank']], dim=1),
            torch.cat([zero_pad, past_rates['pitch']], dim=1),
        ], dim=-1)  # (B, K, 5)

        # Encode sequences
        kine = self.kinematic_enc(past_norm)                  # (B, K, 128)
        rate_feat = self.rate_enc(rates_seq)                   # (B, K, 64)
        # Neighbors: pool over MAX_NEIGHBORS
        ctx_enc = self.neighbor_enc(past_ctx)                  # (B, K, MAX_N, 32)
        ctx_pooled = ctx_enc.mean(dim=2)                       # (B, K, 32)
        ctx_feat = self.neighbor_pool(ctx_pooled)              # (B, K, 64)

        feats = [kine, rate_feat, ctx_feat]
        if self.world_enc is not None and world is not None:
            world_feat = self.world_enc(world)                 # (B, K, 64)
            feats.append(world_feat)
        seq_in = torch.cat(feats, dim=-1)                      # (B, K, gru_in)

        out, h = self.gru(seq_in)
        final_h = out[:, -1]  # (B, hidden_dim)

        last_raw = past_raw[:, -1]  # (B, STATE_DIM)
        return final_h, past_rates, last_raw

    def forward(self, past_norm, future_states_norm, past_ctx, world=None):
        """
        학습용 forward.

        :param past_norm: (B, K, STATE_DIM)
        :param future_states_norm: (B, T, STATE_DIM) target
        :param past_ctx: (B, K, MAX_N, CTX_DIM)
        :return: dict with pred_rates, target_rates
        """
        B = past_norm.shape[0]
        T = future_states_norm.shape[1]

        final_h, past_rates, last_raw = self._encode_past(past_norm, past_ctx, world)

        # Target rates: concat last past state + future states, take diffs
        future_raw = future_states_norm * self.norm_std + self.norm_mean
        full_seq = torch.cat([last_raw.unsqueeze(1), future_raw], dim=1)  # (B, T+1, D)
        target_rates_dict = extract_physics_rates(full_seq)
        target_rates = torch.stack([
            target_rates_dict['omega_z'],
            target_rates_dict['a_x'],
            target_rates_dict['a_z'],
        ], dim=-1)  # (B, T, 3)
        # 정규화
        target_rates_norm = target_rates / self.rate_scale

        # Predicted rates: autoregressive teacher-forcing 아닌 단순 rollout
        pred_rates_norm = self._predict_rates(final_h, last_raw, past_rates, T)

        return {
            'pred_rates': pred_rates_norm,       # (B, T, 3) normalized
            'target_rates': target_rates_norm,   # (B, T, 3) normalized
        }

    def _predict_rates(self, hidden, current_state, last_rates, future_steps):
        """
        Hidden + current state + last rates → future rates (T steps).
        Autoregressive하지만 단순: hidden은 고정, state/rates만 업데이트.
        """
        B = hidden.shape[0]
        device = hidden.device
        preds = []

        # 초기 rate (past의 마지막)
        cur_rates = torch.stack([
            last_rates['omega_z'][:, -1],
            last_rates['a_x'][:, -1],
            last_rates['a_z'][:, -1],
        ], dim=-1)  # (B, 3)
        cur_state = current_state.clone()

        # logstd (per-sequence, not per-step 단순화)
        logstd = self.rate_logstd_head(hidden)  # (B, 3)

        for t in range(future_steps):
            # State는 normalized로
            state_norm = (cur_state - self.norm_mean) / self.norm_std
            cur_rates_5 = torch.cat([
                cur_rates,  # omega_z, a_x, a_z
                torch.zeros(B, 2, device=device),  # bank, pitch placeholder
            ], dim=-1) if cur_rates.shape[-1] == 3 else cur_rates
            # Use just (3) rates here
            head_in = torch.cat([hidden, state_norm, cur_rates_5[:, :5]], dim=-1)
            if head_in.shape[-1] != self.rate_head[0].in_features:
                # Pad if needed
                pad_width = self.rate_head[0].in_features - head_in.shape[-1]
                head_in = torch.cat(
                    [head_in, torch.zeros(B, pad_width, device=device)], dim=-1)

            next_rate_raw = self.rate_head(head_in)  # (B, 3) normalized
            preds.append(next_rate_raw)

            # Integrate: need unnormalized rates for physics
            raw_rates = {
                'omega_z': next_rate_raw[:, 0] * self.rate_scale[0],
                'a_x':     next_rate_raw[:, 1] * self.rate_scale[1],
                'a_z':     next_rate_raw[:, 2] * self.rate_scale[2],
            }
            cur_state = forward_dynamics(cur_state, raw_rates)
            cur_rates = next_rate_raw * 1.0  # normalized 유지

        return torch.stack(preds, dim=1)  # (B, T, 3)

    def compute_loss(self, past_norm, future_norm, past_ctx, world=None,
                     kl_weight=0.0, free_nats_per_step=None,
                     kl_min_nats=None, task_weights=None):
        """
        Rate-prediction loss.

        task_losses: per-rate MSE
        kl_weight는 사용 안 함 (VAE 없음), signature 호환용.
        """
        output = self.forward(past_norm, future_norm, past_ctx, world=world)
        pred = output['pred_rates']     # (B, T, 3)
        target = output['target_rates'] # (B, T, 3)

        tl = {
            'rate_omega_z': (pred[:, :, 0] - target[:, :, 0]).pow(2).mean(),
            'rate_a_x':     (pred[:, :, 1] - target[:, :, 1]).pow(2).mean(),
            'rate_a_z':     (pred[:, :, 2] - target[:, :, 2]).pow(2).mean(),
        }

        if task_weights is None:
            task_weights = {k: 1.0 for k in tl}

        recon_loss = sum(tl[k] * task_weights.get(k, 1.0) for k in tl)
        total_loss = recon_loss  # KL 없음

        return {
            'total_loss': total_loss,
            'task_losses': tl,
            'recon_loss': recon_loss,
            'kl_loss': torch.zeros_like(recon_loss),  # 호환
        }

    @torch.no_grad()
    def predict(self, past_norm, past_ctx, num_samples=50, future_steps=12,
                world_past=None):
        """
        MC trajectory samples via rate noise.

        1. Encode → hidden
        2. Predict mean rates + logstd
        3. Sample rate perturbations, integrate with forward_dynamics
        """
        self.eval()
        B = past_norm.shape[0]
        device = past_norm.device

        final_h, past_rates, last_raw = self._encode_past(past_norm, past_ctx, world_past)

        # Rate uncertainty
        logstd = self.rate_logstd_head(final_h)  # (B, 3)
        std = logstd.exp().clamp(0.1, 2.0)  # normalized rate std

        # Expand for MC
        final_h_exp = final_h.unsqueeze(1).expand(-1, num_samples, -1).reshape(
            B * num_samples, -1)
        last_raw_exp = last_raw.unsqueeze(1).expand(-1, num_samples, -1).reshape(
            B * num_samples, STATE_DIM)
        std_exp = std.unsqueeze(1).expand(-1, num_samples, -1).reshape(
            B * num_samples, 3)

        # Past rates에서 마지막 값 (B,) → (B*S,)
        last_rates_exp = {}
        for k in past_rates:
            v = past_rates[k][:, -1:]  # (B, 1)
            v_exp = v.unsqueeze(1).expand(-1, num_samples, -1).reshape(
                B * num_samples, 1)
            last_rates_exp[k] = v_exp

        cur_state = last_raw_exp.clone()
        cur_rates_norm = torch.stack([
            last_rates_exp['omega_z'].squeeze(-1) / self.rate_scale[0],
            last_rates_exp['a_x'].squeeze(-1) / self.rate_scale[1],
            last_rates_exp['a_z'].squeeze(-1) / self.rate_scale[2],
        ], dim=-1)  # (B*S, 3)

        trajectories = []
        for t in range(future_steps):
            state_norm = (cur_state - self.norm_mean) / self.norm_std
            cur_rates_5 = torch.cat([
                cur_rates_norm,
                torch.zeros(B * num_samples, 2, device=device),
            ], dim=-1)
            head_in = torch.cat([final_h_exp, state_norm, cur_rates_5[:, :5]], dim=-1)
            if head_in.shape[-1] != self.rate_head[0].in_features:
                pad = self.rate_head[0].in_features - head_in.shape[-1]
                head_in = torch.cat(
                    [head_in, torch.zeros(head_in.shape[0], pad, device=device)],
                    dim=-1)

            mean_rate = self.rate_head(head_in)
            # Rate-level stochastic sampling
            eps = torch.randn_like(mean_rate)
            noisy_rate = mean_rate + std_exp * eps * math.sqrt(t + 1) * 0.3

            raw_rates = {
                'omega_z': noisy_rate[:, 0] * self.rate_scale[0],
                'a_x':     noisy_rate[:, 1] * self.rate_scale[1],
                'a_z':     noisy_rate[:, 2] * self.rate_scale[2],
            }
            cur_state = forward_dynamics(cur_state, raw_rates)
            cur_rates_norm = noisy_rate
            trajectories.append(cur_state.clone())

        traj = torch.stack(trajectories, dim=1)  # (B*S, T, STATE_DIM) raw
        traj = traj.view(B, num_samples, future_steps, STATE_DIM)
        return traj

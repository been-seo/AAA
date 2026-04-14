"""
ADS-B 녹화 데이터 (SQLite DB + JSONL) → 학습용 시퀀스 데이터셋

DB는 런타임에 직접 읽음 — 레코딩이 계속 쌓여도 재시작 없이 반영.
init에서 인덱스만 구축, __getitem__에서 해당 윈도우만 DB 쿼리.
"""
import json
import math
import os
import sqlite3
import threading
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# 항공기 상태 벡터 구성 (10차원)
STATE_DIM = 10
STATE_KEYS = [
    'lat', 'lon', 'baro_altitude_ft', 'ground_speed_kt',
    'true_track_deg', 'vertical_rate_ft_min',
    'ias_kt', 'mach', 'wind_direction_deg', 'wind_speed_kt',
]

# DB 컬럼 → STATE_KEYS 매핑
DB_COLUMNS = [
    'lat', 'lon', 'alt_baro', 'gs_kt',
    'track_deg', 'vrate_fpm', 'ias_kt', 'mach',
    'wind_dir', 'wind_spd',
]

NORM_MEAN = np.array([36.0, 128.0, 25000, 400, 180, 0, 350, 0.75, 180, 20], dtype=np.float32)
NORM_STD  = np.array([3.0,  3.0,  15000, 150, 180, 2000, 150, 0.25, 180, 30], dtype=np.float32)

CONTEXT_DIM = 6
MAX_NEIGHBORS = 8
NEIGHBOR_RADIUS_NM = 50.0


def _haversine_nm(lat1, lon1, lat2, lon2):
    R = 3440.065
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _get_neighbor_context(target_icao, target_state, ac_rows, max_n=MAX_NEIGHBORS):
    """주변 항공기 상대 상태. ac_rows: [(icao, lat, lon, alt, gs, track, vrate)]"""
    t_lat, t_lon, t_alt = target_state[0], target_state[1], target_state[2]
    t_gs, t_track, t_vr = target_state[3], target_state[4], target_state[5]

    neighbors = []
    for (icao, lat, lon, alt, gs, track, vrate) in ac_rows:
        if icao == target_icao or lat is None or lon is None:
            continue
        dist = _haversine_nm(t_lat, t_lon, lat, lon)
        if dist > NEIGHBOR_RADIUS_NM:
            continue
        dx = (lon - t_lon) * 60.0 * math.cos(math.radians(t_lat))
        dy = (lat - t_lat) * 60.0
        dalt = ((alt or 0) - t_alt) / 1000.0
        dgs = ((gs or 0) - t_gs) / 100.0
        dtrack = (((track or 0) - t_track + 180) % 360 - 180) / 180.0
        dvrate = ((vrate or 0) - t_vr) / 1000.0
        neighbors.append((dist, [dx, dy, dalt, dgs, dtrack, dvrate]))

    neighbors.sort(key=lambda x: x[0])
    ctx = np.zeros((max_n, CONTEXT_DIM), dtype=np.float32)
    for i, (_, vec) in enumerate(neighbors[:max_n]):
        ctx[i] = vec
    return ctx


def _interpolate_nans(states):
    """NaN 보간: 선형 보간 → 남은 NaN은 평균으로."""
    mask = ~np.isnan(states)
    for dim in range(STATE_DIM):
        col = states[:, dim]
        nans = np.isnan(col)
        if nans.all():
            col[:] = NORM_MEAN[dim]
        elif nans.any():
            valid = ~nans
            col[nans] = np.interp(
                np.where(nans)[0], np.where(valid)[0], col[valid])
        states[:, dim] = col
    return states, mask


class TrajectoryDataset(Dataset):
    """
    DB 런타임 읽기 데이터셋.

    init: DB 스캔 → (icao24, [snapshot_ids]) 인덱스만 구축 (빠름)
    __getitem__: 해당 윈도우만 DB에서 쿼리 (I/O 최소)
    refresh(): 새로 쌓인 데이터 반영 (인덱스 재구축)
    """

    def __init__(self, data_dir, past_steps=6, future_steps=12,
                 max_files=None, stride=2):
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.total_steps = past_steps + future_steps
        self.stride = stride
        self.data_dir = Path(data_dir)

        # 스레드별 DB 커넥션 (SQLite는 스레드 간 공유 불가)
        self._local = threading.local()

        # 인덱스: [(db_path, icao24, [snapshot_id_window])]
        self.index = []
        self._db_paths = []

        self._build_index(max_files)

    def _get_conn(self, db_path):
        """스레드별 DB 커넥션 (DataLoader num_workers > 0 대응)."""
        key = f'conn_{db_path}'
        if not hasattr(self._local, key) or getattr(self._local, key) is None:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")  # 레코딩 중 읽기 허용
            setattr(self._local, key, conn)
        return getattr(self._local, key)

    def _build_index(self, max_files=None):
        """DB 파일 스캔 → 항공기별 연속 구간 인덱스 구축."""
        db_files = sorted(self.data_dir.glob("*.db"))
        if max_files:
            db_files = db_files[:max_files]
        self._db_paths = [str(f) for f in db_files]

        self.index = []

        for db_path in self._db_paths:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")

            # 항공기별 snapshot_id 목록 (시간순)
            rows = conn.execute("""
                SELECT a.icao24, s.id as sid, s.timestamp
                FROM aircraft a
                JOIN snapshots s ON a.snapshot_id = s.id
                WHERE a.lat IS NOT NULL AND a.lon IS NOT NULL
                      AND a.alt_baro IS NOT NULL
                      AND (a.on_ground IS NULL OR a.on_ground != 1)
                ORDER BY a.icao24, s.timestamp
            """).fetchall()
            conn.close()

            # 항공기별 그룹화
            icao_snaps = {}  # icao → [(sid, ts)]
            for icao, sid, ts in rows:
                if icao not in icao_snaps:
                    icao_snaps[icao] = []
                icao_snaps[icao].append((sid, ts))

            # 연속 구간 → 슬라이딩 윈도우 인덱스
            for icao, snap_list in icao_snaps.items():
                # 연속 구간 분할 (gap > 30초)
                segments = []
                current_seg = [snap_list[0]]
                for i in range(1, len(snap_list)):
                    if snap_list[i][1] - snap_list[i-1][1] <= 30:
                        current_seg.append(snap_list[i])
                    else:
                        if len(current_seg) >= self.total_steps:
                            segments.append(current_seg)
                        current_seg = [snap_list[i]]
                if len(current_seg) >= self.total_steps:
                    segments.append(current_seg)

                for seg in segments:
                    sids = [s[0] for s in seg]
                    for start in range(0, len(sids) - self.total_steps + 1, self.stride):
                        window_sids = sids[start:start + self.total_steps]
                        self.index.append((db_path, icao, window_sids))

        print(f"[Dataset] {len(self.index)} samples indexed "
              f"from {len(self._db_paths)} DB files")

    def refresh(self):
        """새로 쌓인 레코딩 반영 (인덱스 재구축)."""
        old_count = len(self.index)
        self._build_index()
        new_count = len(self.index)
        if new_count > old_count:
            print(f"[Dataset] Refreshed: {old_count} → {new_count} samples "
                  f"(+{new_count - old_count})")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        db_path, icao, window_sids = self.index[idx]
        conn = self._get_conn(db_path)

        # 해당 윈도우의 상태 벡터 쿼리
        placeholders = ','.join('?' * len(window_sids))
        rows = conn.execute(f"""
            SELECT s.id, a.{', a.'.join(DB_COLUMNS)}
            FROM aircraft a
            JOIN snapshots s ON a.snapshot_id = s.id
            WHERE a.icao24 = ? AND s.id IN ({placeholders})
            ORDER BY s.timestamp
        """, [icao] + window_sids).fetchall()

        # 상태 벡터 구축
        states = np.full((self.total_steps, STATE_DIM), np.nan, dtype=np.float32)
        sid_to_idx = {sid: i for i, sid in enumerate(window_sids)}
        for row in rows:
            sid = row[0]
            if sid in sid_to_idx:
                i = sid_to_idx[sid]
                for d in range(STATE_DIM):
                    val = row[1 + d]
                    if val is not None:
                        states[i, d] = float(val)

        states, mask = _interpolate_nans(states)

        # Context: 같은 snapshot의 다른 항공기
        contexts = np.zeros((self.total_steps, MAX_NEIGHBORS, CONTEXT_DIM),
                            dtype=np.float32)
        for i, sid in enumerate(window_sids):
            ac_rows = conn.execute("""
                SELECT icao24, lat, lon,
                       COALESCE(alt_baro, 0), COALESCE(gs_kt, 0),
                       COALESCE(track_deg, 0), COALESCE(vrate_fpm, 0)
                FROM aircraft
                WHERE snapshot_id = ? AND lat IS NOT NULL AND lon IS NOT NULL
            """, (sid,)).fetchall()
            contexts[i] = _get_neighbor_context(icao, states[i], ac_rows)

        # 정규화
        norm_states = (states - NORM_MEAN) / NORM_STD

        past = torch.from_numpy(norm_states[:self.past_steps].copy())
        future = torch.from_numpy(norm_states[self.past_steps:].copy())
        ctx = torch.from_numpy(contexts)
        msk = torch.from_numpy(mask.astype(np.float32))
        future_raw = torch.from_numpy(states[self.past_steps:].copy())

        return past, future, ctx, msk, future_raw

    def get_norm_params(self):
        return torch.from_numpy(NORM_MEAN), torch.from_numpy(NORM_STD)

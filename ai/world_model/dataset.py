"""
ADS-B 녹화 데이터(SQLite DB) 학습용 시퀀스 데이터셋

DB 파일에 직접 읽음. 레코딩이 계속 추가되어 실시간 데이터가 반영.
init에서 인덱스만 구축, __getitem__에서 해당 윈도우만 DB 쿼리.

v2: World context 추가 - 웨이포인트/항로 피처
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

# World context: 가장 가까운 K개 웨이포인트
MAX_NEAREST_WP = 3
# 웨이포인트당 피처: dx_nm, dy_nm, type_id, bearing_offset (항공기 heading 대비)
WP_FEAT_DIM = 4
WORLD_DIM = MAX_NEAREST_WP * WP_FEAT_DIM  # 12

# 웨이포인트 타입 인코딩
WP_TYPE_MAP = {
    "VORTAC": 1.0,
    "VOR": 0.8,
    "TACAN": 0.6,
    "FIX": 0.4,
    "FIR_BDRY": 0.2,
}


def _haversine_nm(lat1, lon1, lat2, lon2):
    R = 3440.065
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1, lon1, lat2, lon2):
    """lat1,lon1에서 lat2,lon2로의 방위각 (도)."""
    dlon = math.radians(lon2 - lon1)
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = (math.cos(lat1r) * math.sin(lat2r) -
         math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
    return math.degrees(math.atan2(x, y)) % 360


def _load_waypoints(data_dir):
    """ats_routes_named.json에서 웨이포인트 로드."""
    wp_path = Path(data_dir) / "ats_routes_named.json"
    if not wp_path.exists():
        print(f"[Dataset] WARNING: {wp_path} not found, world context disabled")
        return None

    with open(wp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    waypoints = data["waypoints"]
    # numpy 배열로 변환 (빠른 검색용)
    wp_array = np.array([[w["lat"], w["lon"]] for w in waypoints], dtype=np.float64)
    wp_types = np.array([WP_TYPE_MAP.get(w["type"], 0.0) for w in waypoints], dtype=np.float32)
    wp_names = [w["name"] for w in waypoints]

    print(f"[Dataset] Loaded {len(waypoints)} waypoints from {wp_path.name}")
    return wp_array, wp_types, wp_names


def _get_nearest_waypoints(lat, lon, track_deg, wp_array, wp_types, k=MAX_NEAREST_WP):
    """항공기 위치에서 가장 가까운 K개 웨이포인트 피처 반환.

    Returns: (k * WP_FEAT_DIM,) array
        각 웨이포인트: [dx_nm, dy_nm, type_id, bearing_offset_normalized]
    """
    feat = np.zeros(k * WP_FEAT_DIM, dtype=np.float32)

    # 간이 거리 계산 (정확한 haversine 대신 근사)
    cos_lat = math.cos(math.radians(lat))
    dx_all = (wp_array[:, 1] - lon) * 60.0 * cos_lat  # NM
    dy_all = (wp_array[:, 0] - lat) * 60.0  # NM
    dist_sq = dx_all**2 + dy_all**2

    # 가장 가까운 K개 인덱스
    nearest_idx = np.argpartition(dist_sq, min(k, len(dist_sq) - 1))[:k]
    nearest_idx = nearest_idx[np.argsort(dist_sq[nearest_idx])]

    for i, idx in enumerate(nearest_idx):
        dx_nm = dx_all[idx]
        dy_nm = dy_all[idx]

        # 웨이포인트 방위각
        wp_bearing = math.degrees(math.atan2(dx_nm, dy_nm)) % 360
        # 항공기 heading 대비 상대 방위 (-180 ~ 180)
        bearing_offset = ((wp_bearing - track_deg + 180) % 360 - 180) / 180.0

        # 거리 정규화 (100NM 기준)
        feat[i * WP_FEAT_DIM + 0] = dx_nm / 100.0
        feat[i * WP_FEAT_DIM + 1] = dy_nm / 100.0
        feat[i * WP_FEAT_DIM + 2] = wp_types[idx]
        feat[i * WP_FEAT_DIM + 3] = bearing_offset

    return feat


def _get_neighbor_context(target_icao, target_state, ac_rows, max_n=MAX_NEIGHBORS):
    """주변 항공기의 상태. ac_rows: [(icao, lat, lon, alt, gs, track, vrate)]"""
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
    """NaN 보간: 선형 보간, 전부 NaN이면 평균으로."""
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
    DB 기반 궤적 데이터셋.

    init: DB 스캔 후 (icao24, [snapshot_ids]) 인덱스만 구축 (빠름)
    __getitem__: 해당 윈도우만 DB에서 쿼리 (I/O 최소)
    refresh(): 새로 추가된 데이터 반영 (인덱스 재구축)
    """

    def __init__(self, data_dir, past_steps=6, future_steps=12,
                 max_files=None, stride=2):
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.total_steps = past_steps + future_steps
        self.stride = stride
        self.data_dir = Path(data_dir)

        # 스레드별 DB 커넥션 (SQLite는 스레드간 공유 불가)
        self._local = threading.local()

        # 인덱스: [(db_path, icao24, [snapshot_id_window])]
        self.index = []
        self._db_paths = []

        # World context: 웨이포인트
        wp_data = _load_waypoints(data_dir)
        if wp_data is not None:
            self._wp_array, self._wp_types, self._wp_names = wp_data
            self._has_world = True
        else:
            self._has_world = False

        self._build_index(max_files)

    def _get_conn(self, db_path):
        """스레드별 DB 커넥션 (DataLoader num_workers > 0 대응)."""
        key = f'conn_{db_path}'
        if not hasattr(self._local, key) or getattr(self._local, key) is None:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            setattr(self._local, key, conn)
        return getattr(self._local, key)

    def _build_index(self, max_files=None):
        """DB 파일 스캔 후 항공기별 연속 구간 인덱스 구축."""
        db_files = sorted(self.data_dir.glob("*.db"))
        if max_files:
            db_files = db_files[:max_files]
        self._db_paths = [str(f) for f in db_files]

        self.index = []

        for db_path in self._db_paths:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")

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

            icao_snaps = {}
            for icao, sid, ts in rows:
                if icao not in icao_snaps:
                    icao_snaps[icao] = []
                icao_snaps[icao].append((sid, ts))

            for icao, snap_list in icao_snaps.items():
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
        """새로 추가된 레코드 반영 (인덱스 재구축)."""
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

        placeholders = ','.join('?' * len(window_sids))
        rows = conn.execute(f"""
            SELECT s.id, a.{', a.'.join(DB_COLUMNS)}
            FROM aircraft a
            JOIN snapshots s ON a.snapshot_id = s.id
            WHERE a.icao24 = ? AND s.id IN ({placeholders})
            ORDER BY s.timestamp
        """, [icao] + window_sids).fetchall()

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

        # World context: 웨이포인트 피처
        world = np.zeros((self.total_steps, WORLD_DIM), dtype=np.float32)
        if self._has_world:
            for i in range(self.total_steps):
                lat, lon = states[i, 0], states[i, 1]
                track = states[i, 4]  # true_track_deg
                world[i] = _get_nearest_waypoints(
                    lat, lon, track, self._wp_array, self._wp_types)

        # 정규화
        norm_states = (states - NORM_MEAN) / NORM_STD

        past = torch.from_numpy(norm_states[:self.past_steps].copy())
        future = torch.from_numpy(norm_states[self.past_steps:].copy())
        ctx = torch.from_numpy(contexts)
        msk = torch.from_numpy(mask.astype(np.float32))
        future_raw = torch.from_numpy(states[self.past_steps:].copy())
        world_t = torch.from_numpy(world)

        return past, future, ctx, msk, future_raw, world_t

    def get_norm_params(self):
        return torch.from_numpy(NORM_MEAN), torch.from_numpy(NORM_STD)

    @property
    def world_dim(self):
        return WORLD_DIM if self._has_world else 0

    @property
    def has_world(self):
        return self._has_world

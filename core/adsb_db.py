"""
ADS-B SQLite 데이터베이스 레코더/리더

JSONL 대신 SQLite로 녹화:
- 빠른 시간 범위 쿼리 (WHERE timestamp BETWEEN)
- ICAO별 궤적 추출 (WHERE icao24 = ?)
- 공간 필터링 가능
- 파일 하나로 깔끔하게 관리
- World Model 학습 데이터 직접 쿼리

스키마:
  snapshots(id, timestamp)
  aircraft(snapshot_id, icao24, callsign, lat, lon, alt_baro, alt_geom,
           gs_kt, track_deg, vrate_fpm, ias_kt, tas_kt, mach,
           mag_hdg, true_hdg, roll_deg, squawk, on_ground,
           wind_dir, wind_spd, oat, tat, category, registration, model)
"""
import os
import sqlite3
import time
import threading
import json
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snap_ts ON snapshots(timestamp);

CREATE TABLE IF NOT EXISTS aircraft (
    snapshot_id INTEGER NOT NULL,
    icao24 TEXT NOT NULL,
    callsign TEXT,
    lat REAL,
    lon REAL,
    alt_baro INTEGER,
    alt_geom INTEGER,
    gs_kt REAL,
    track_deg REAL,
    vrate_fpm INTEGER,
    ias_kt INTEGER,
    tas_kt INTEGER,
    mach REAL,
    mag_hdg REAL,
    true_hdg REAL,
    roll_deg REAL,
    squawk TEXT,
    on_ground INTEGER,
    wind_dir INTEGER,
    wind_spd INTEGER,
    oat INTEGER,
    tat INTEGER,
    category TEXT,
    registration TEXT,
    model TEXT,
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
);
CREATE INDEX IF NOT EXISTS idx_ac_snap ON aircraft(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_ac_icao ON aircraft(icao24);
CREATE INDEX IF NOT EXISTS idx_ac_icao_snap ON aircraft(icao24, snapshot_id);
"""


class ADSBDatabase:
    """
    SQLite 기반 ADS-B 녹화 DB.

    사용법 (녹화):
        db = ADSBDatabase("data/recordings/adsb.db")
        db.write_snapshot(aircraft_list)
        db.close()

    사용법 (읽기):
        db = ADSBDatabase("data/recordings/adsb.db", readonly=True)
        trajs = db.get_trajectories(min_length=18)
        db.close()
    """

    def __init__(self, db_path, readonly=False):
        self.db_path = db_path
        self.readonly = readonly
        self._lock = threading.Lock()
        self.snapshot_count = 0

        if readonly:
            self._conn = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True,
                check_same_thread=False)
        else:
            os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

        self._conn.row_factory = sqlite3.Row
        # filepath 호환 (record_adsb.py 등에서 사용)
        self.filepath = db_path

    def write_snapshot(self, aircraft_list, timestamp=None):
        """항공기 리스트를 DB에 기록"""
        if self.readonly:
            raise RuntimeError("Database is read-only")

        ts = timestamp or time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("INSERT INTO snapshots (timestamp) VALUES (?)", (ts,))
            snap_id = cur.lastrowid

            rows = []
            for ac in aircraft_list:
                lat = ac.get('lat')
                lon = ac.get('lon')
                if lat is None or lon is None:
                    continue
                rows.append((
                    snap_id,
                    ac.get('icao24', ''),
                    ac.get('callsign', ''),
                    lat, lon,
                    ac.get('baro_altitude_ft'),
                    ac.get('alt_geom_ft'),
                    ac.get('ground_speed_kt'),
                    ac.get('true_track_deg'),
                    ac.get('vertical_rate_ft_min'),
                    ac.get('ias_kt'),
                    ac.get('tas_kt'),
                    ac.get('mach'),
                    ac.get('mag_heading_deg'),
                    ac.get('true_heading_deg'),
                    ac.get('roll_deg'),
                    ac.get('squawk'),
                    ac.get('on_ground'),
                    ac.get('wind_direction_deg'),
                    ac.get('wind_speed_kt'),
                    ac.get('oat'),
                    ac.get('tat'),
                    ac.get('category'),
                    ac.get('registration'),
                    ac.get('aircraft_model'),
                ))

            cur.executemany(
                "INSERT INTO aircraft (snapshot_id, icao24, callsign, "
                "lat, lon, alt_baro, alt_geom, gs_kt, track_deg, vrate_fpm, "
                "ias_kt, tas_kt, mach, mag_hdg, true_hdg, roll_deg, squawk, "
                "on_ground, wind_dir, wind_spd, oat, tat, category, registration, model) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows)

            self._conn.commit()
            self.snapshot_count += 1

    def get_time_range(self):
        """(min_timestamp, max_timestamp) 반환"""
        cur = self._conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM snapshots")
        row = cur.fetchone()
        return (row[0], row[1]) if row[0] else (0, 0)

    def get_snapshot_count(self):
        """총 스냅샷 수"""
        cur = self._conn.execute("SELECT COUNT(*) FROM snapshots")
        return cur.fetchone()[0]

    def get_aircraft_count(self):
        """고유 ICAO 수"""
        cur = self._conn.execute("SELECT COUNT(DISTINCT icao24) FROM aircraft")
        return cur.fetchone()[0]

    def get_snapshots(self, start_ts=None, end_ts=None, limit=None):
        """
        시간 범위로 스냅샷 조회.
        Returns: [(timestamp, [aircraft_dict, ...])]
        """
        sql = "SELECT id, timestamp FROM snapshots WHERE 1=1"
        params = []
        if start_ts:
            sql += " AND timestamp >= ?"
            params.append(start_ts)
        if end_ts:
            sql += " AND timestamp <= ?"
            params.append(end_ts)
        sql += " ORDER BY timestamp"
        if limit:
            sql += f" LIMIT {int(limit)}"

        results = []
        for snap in self._conn.execute(sql, params).fetchall():
            snap_id, ts = snap['id'], snap['timestamp']
            ac_rows = self._conn.execute(
                "SELECT * FROM aircraft WHERE snapshot_id = ?",
                (snap_id,)).fetchall()
            ac_list = [dict(row) for row in ac_rows]
            results.append((ts, ac_list))

        return results

    def get_trajectory(self, icao24, start_ts=None, end_ts=None):
        """
        특정 항공기의 궤적 추출.
        Returns: [(timestamp, lat, lon, alt, gs, track, vrate, ...)]
        """
        sql = """
        SELECT s.timestamp, a.*
        FROM aircraft a
        JOIN snapshots s ON s.id = a.snapshot_id
        WHERE a.icao24 = ?
        """
        params = [icao24]
        if start_ts:
            sql += " AND s.timestamp >= ?"
            params.append(start_ts)
        if end_ts:
            sql += " AND s.timestamp <= ?"
            params.append(end_ts)
        sql += " ORDER BY s.timestamp"

        return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    def get_trajectories(self, min_length=18, start_ts=None, end_ts=None):
        """
        모든 항공기의 궤적 추출 (World Model 학습용).
        Returns: dict[icao24] → list of dict
        """
        # 충분한 길이의 궤적만
        sql = """
        SELECT a.icao24, COUNT(*) as cnt
        FROM aircraft a
        JOIN snapshots s ON s.id = a.snapshot_id
        WHERE a.lat IS NOT NULL AND a.on_ground != 1
        """
        params = []
        if start_ts:
            sql += " AND s.timestamp >= ?"
            params.append(start_ts)
        if end_ts:
            sql += " AND s.timestamp <= ?"
            params.append(end_ts)
        sql += f" GROUP BY a.icao24 HAVING cnt >= {min_length}"

        icaos = [row['icao24'] for row in self._conn.execute(sql, params).fetchall()]

        result = {}
        for icao in icaos:
            result[icao] = self.get_trajectory(icao, start_ts, end_ts)

        return result

    def get_nearby_aircraft(self, lat, lon, radius_deg=1.0, timestamp=None):
        """특정 위치 주변 항공기 조회 (가장 최근 스냅샷)"""
        if timestamp is None:
            snap = self._conn.execute(
                "SELECT id, timestamp FROM snapshots ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        else:
            snap = self._conn.execute(
                "SELECT id, timestamp FROM snapshots WHERE timestamp <= ? "
                "ORDER BY timestamp DESC LIMIT 1", (timestamp,)
            ).fetchone()

        if not snap:
            return []

        return [dict(row) for row in self._conn.execute(
            "SELECT * FROM aircraft WHERE snapshot_id = ? "
            "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
            (snap['id'], lat - radius_deg, lat + radius_deg,
             lon - radius_deg, lon + radius_deg)
        ).fetchall()]

    def import_jsonl(self, jsonl_path, progress_interval=1000):
        """기존 JSONL 파일을 DB로 임포트"""
        if self.readonly:
            raise RuntimeError("Database is read-only")

        count = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ts = obj.get('timestamp', 0)
                    aircraft = obj.get('aircraft', [])
                    if aircraft:
                        self.write_snapshot(aircraft, timestamp=ts)
                        count += 1
                        if count % progress_interval == 0:
                            print(f"  [Import] {count} snapshots...")
                except json.JSONDecodeError:
                    continue

        print(f"  [Import] Done: {count} snapshots from {jsonl_path}")
        return count

    def close(self):
        with self._lock:
            self._conn.close()
        print(f"[ADSB-DB] Closed {self.db_path} "
              f"({self.snapshot_count} new snapshots)")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

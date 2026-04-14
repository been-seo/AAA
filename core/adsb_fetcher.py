"""
ADS-B 실시간 항공기 데이터 수신기
- ADSBexchange binCraft 바이너리 포맷 파싱 (전체 필드)
- 실시간 수신 (LiveFetcher)
- 녹화 (Recorder) — JSON Lines 형식
- 재생 (ReplayFetcher) — 녹화 파일을 배속으로 재생
"""
import math
import time
import json
import os
import struct
import threading
import glob

import requests
import zstandard as zstd
import browser_cookie3

from config import ADSB_BBOX, ADSB_INTERVAL


class ADSBFetcher(threading.Thread):
    """실시간 ADS-B 수신기 (ADSBexchange binCraft)"""

    def __init__(self, aircraft_queue, stop_event, bbox=ADSB_BBOX, interval=ADSB_INTERVAL):
        super().__init__(daemon=True)
        self.aircraft_queue = aircraft_queue
        self.stop_event = stop_event
        self.bbox = bbox
        self.interval = interval
        self.recorder = None  # ADSBRecorder 연결 시 자동 녹화

    def _parse_bincraft(self, buffer):
        """binCraft 바이너리 → 항공기 dict 리스트 (전체 필드)"""
        try:
            u32_header = struct.unpack_from('<11I', buffer, 0)
            now = u32_header[0] / 1e3 + 4294967.296 * u32_header[1]
            stride = u32_header[2]
            global_ac_count_withpos = u32_header[3]
            globeIndex = u32_header[4]
            messages = u32_header[7]

            south, west, north, east = struct.unpack_from('<4h', buffer, 20)
            receiver_lat = struct.unpack_from('<i', buffer, 32)[0] / 1e6
            receiver_lon = struct.unpack_from('<i', buffer, 36)[0] / 1e6

            aircraft_list = []

            for offset in range(stride, len(buffer), stride):
                if offset + stride > len(buffer):
                    break

                s32 = struct.unpack_from('<12i', buffer, offset)
                s16 = struct.unpack_from('<40h', buffer, offset)
                u16 = struct.unpack_from('<44H', buffer, offset)
                u8 = struct.unpack_from(f'<{stride}B', buffer, offset)

                hex_id = s32[0] & 0xFFFFFF
                t_flag = s32[0] & (1 << 24)
                hex_str = f"{hex_id:06X}"
                if t_flag:
                    hex_str = "~" + hex_str

                s_value = f"{u16[16]:04x}"
                squawk = (str(int(s_value[0], 16)) + s_value[1:]) if s_value[0] > "9" else s_value

                flight = ''.join(chr(u8[i]) for i in range(78, 86) if u8[i] != 0).strip()
                t_field = ''.join(chr(u8[i]) for i in range(88, 92) if u8[i] != 0).strip()
                r_field = ''.join(chr(u8[i]) for i in range(92, 104) if u8[i] != 0).strip()

                rssi = 10 * math.log((u8[105] * u8[105]) / 65025 + 1.125e-5) / math.log(10)

                ac = {
                    'icao24': hex_str,
                    'seen_pos': u16[2] / 10,
                    'seen': u16[3] / 10,
                    'lon': s32[2] / 1e6,
                    'lat': s32[3] / 1e6,
                    'vertical_rate_ft_min': s16[8] * 8,
                    'geom_rate': s16[9] * 8,
                    'baro_altitude_ft': s16[10] * 25,
                    'alt_geom_ft': s16[11] * 25,
                    'nav_altitude_mcp_ft': u16[12] * 4,
                    'nav_altitude_fms_ft': u16[13] * 4,
                    'nav_qnh': s16[14] / 10,
                    'nav_heading_deg': s16[15] / 90,
                    'squawk': squawk,
                    'ground_speed_kt': s16[17] / 10,
                    'mach': s16[18] / 1e3,
                    'roll_deg': s16[19] / 100,
                    'true_track_deg': s16[20] / 90,
                    'track_rate': s16[21] / 100,
                    'mag_heading_deg': s16[22] / 90,
                    'true_heading_deg': s16[23] / 90,
                    'wind_direction_deg': s16[24],
                    'wind_speed_kt': s16[25],
                    'oat': s16[26],
                    'tat': s16[27],
                    'tas_kt': u16[28],
                    'ias_kt': u16[29],
                    'rc': u16[30],
                    'messageRate': u16[31] / 10 if globeIndex and 20220916 else None,
                    'messages': None if globeIndex and 20220916 else u16[31],
                    'category': f"{u8[64]:02X}" if u8[64] else None,
                    'nic': u8[65],
                    'nav_modes': True,
                    'emergency': u8[67] & 15,
                    'aircraft_type': (u8[67] & 240) >> 4,
                    'on_ground': u8[68] & 15,
                    'nav_altitude_src': (u8[68] & 240) >> 4,
                    'sil_type': u8[69] & 15,
                    'adsb_version': (u8[69] & 240) >> 4,
                    'adsr_version': u8[70] & 15,
                    'tisb_version': (u8[70] & 240) >> 4,
                    'nac_p': u8[71] & 15,
                    'nac_v': (u8[71] & 240) >> 4,
                    'sil': u8[72] & 3,
                    'gva': (u8[72] & 12) >> 2,
                    'sda': (u8[72] & 48) >> 4,
                    'nic_a': (u8[72] & 64) >> 6,
                    'nic_c': (u8[72] & 128) >> 7,
                    'callsign': flight,
                    'dbFlags': u16[43],
                    'aircraft_model': t_field,
                    'registration': r_field,
                    'receiver_count': u8[104],
                    'rssi': rssi,
                    'extra_flags': u8[106],
                    'nogps': u8[106] & 1,
                }

                if ac['lat'] != 0 and ac['lon'] != 0:
                    aircraft_list.append(ac)

            return {
                'now': now,
                'aircraft': aircraft_list,
                'receiver_lat': receiver_lat,
                'receiver_lon': receiver_lon,
                'global_ac_count_withpos': global_ac_count_withpos,
                'globeIndex': globeIndex,
                'messages': messages,
                'limits': (south, west, north, east),
            }
        except Exception:
            return {'aircraft': []}

    def run(self):
        print(f"[ADSBFetcher] Started (interval={self.interval}s)")
        while not self.stop_event.is_set():
            t0 = time.time()
            try:
                url = (f"https://globe.adsbexchange.com/re-api/"
                       f"?binCraft&zstd&box={self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}")
                cookies = browser_cookie3.firefox(domain_name='adsbexchange.com')
                headers = {
                    "authority": "globe.adsbexchange.com",
                    "method": "GET",
                    "scheme": "https",
                    "accept": "*/*",
                    "accept-encoding": "gzip, deflate, br, zstd",
                    "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                    "dnt": "1",
                    "referer": "https://globe.adsbexchange.com/",
                    "sec-fetch-dest": "empty",
                    "sec-fetch-mode": "cors",
                    "sec-fetch-site": "same-origin",
                    "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                                   "Chrome/135.0.0.0 Safari/537.36"),
                    "x-requested-with": "XMLHttpRequest",
                }
                resp = requests.get(url, headers=headers, cookies=cookies, timeout=15, verify=True)

                if resp.status_code == 200:
                    parsed = self._parse_bincraft(
                        zstd.ZstdDecompressor().decompress(resp.content))
                    aircraft_data = parsed.get("aircraft", [])
                    if aircraft_data:
                        # 녹화 중이면 저장
                        if self.recorder:
                            self.recorder.write_snapshot(aircraft_data)
                        try:
                            self.aircraft_queue.put_nowait(aircraft_data)
                        except Exception:
                            pass
                elif resp.status_code != 429:
                    print(f"[ADSBFetcher] HTTP {resp.status_code}")
            except Exception:
                pass

            elapsed = time.time() - t0
            wait = self.interval - elapsed
            if wait > 0:
                self.stop_event.wait(wait)
        print("[ADSBFetcher] Stopped")


# ─────────────────────────────────────────────
#  녹화
# ─────────────────────────────────────────────

class ADSBRecorder:
    """
    ADS-B 스냅샷을 JSON Lines 파일로 녹화한다.
    각 줄: {"timestamp": ..., "aircraft": [...]}

    사용법 1 — ADSBFetcher에 연결:
        recorder = ADSBRecorder("data/recordings")
        fetcher.recorder = recorder   # fetcher가 수신할 때마다 자동 저장
        ...
        recorder.close()

    사용법 2 — 수동:
        recorder = ADSBRecorder("data/recordings")
        recorder.write_snapshot(aircraft_list)
        recorder.close()
    """

    def __init__(self, output_dir, prefix="adsb"):
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(output_dir, f"{prefix}_{ts}.jsonl")
        self._file = open(self.filepath, "w", encoding="utf-8")
        self._lock = threading.Lock()
        self.snapshot_count = 0
        print(f"[Recorder] Recording to {self.filepath}")

    def write_snapshot(self, aircraft_list):
        record = {
            "timestamp": time.time(),
            "aircraft": aircraft_list,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()
            self.snapshot_count += 1

    def close(self):
        with self._lock:
            self._file.close()
        print(f"[Recorder] Saved {self.snapshot_count} snapshots to {self.filepath}")


# ─────────────────────────────────────────────
#  재생
# ─────────────────────────────────────────────

class ADSBReplayFetcher(threading.Thread):
    """
    녹화된 ADS-B 데이터를 재생하는 페이크 Fetcher.
    ADSBFetcher와 동일한 인터페이스 (aircraft_queue, stop_event).

    지원 포맷:
      - .jsonl  (JSON Lines, 1줄 = 1 스냅샷)
      - .json   (기존 녹화 포맷: 1줄 = 1 스냅샷, 또는 파일 전체가 배열)

    배속:
      speed=1.0  → 실시간 (원본 timestamp 간격 유지)
      speed=10.0 → 10배속
      speed=0    → 대기 없이 가능한 빠르게

    루프:
      loop=True  → 녹화 끝나면 처음부터 반복
    """

    def __init__(self, aircraft_queue, stop_event, source, speed=1.0, loop=True):
        """
        :param source: 파일 경로 (.jsonl/.json) 또는 디렉토리 (*.json 자동 수집)
        :param speed: 재생 배속 (0=최대속도, 1=실시간, 10=10배속)
        :param loop: 끝나면 반복 여부
        """
        super().__init__(daemon=True)
        self.aircraft_queue = aircraft_queue
        self.stop_event = stop_event
        self.speed = speed
        self.loop = loop
        self.recorder = None  # 호환성: ADSBFetcher와 동일 인터페이스
        self.current_replay_ts = 0.0  # 현재 리플레이 중인 스냅샷의 timestamp

        self._snapshots = self._load(source)
        print(f"[Replay] Loaded {len(self._snapshots)} snapshots "
              f"(speed={speed}x, loop={loop})")

    def _load(self, source):
        """파일 또는 디렉토리에서 스냅샷 로드. [(timestamp, aircraft_list), ...] 반환"""
        files = []
        if os.path.isdir(source):
            files = sorted(glob.glob(os.path.join(source, "*.json")) +
                           glob.glob(os.path.join(source, "*.jsonl")))
        elif os.path.isfile(source):
            files = [source]
        else:
            raise FileNotFoundError(f"[Replay] Source not found: {source}")

        snapshots = []
        for fpath in files:
            snapshots.extend(self._load_file(fpath))

        # timestamp 순 정렬
        snapshots.sort(key=lambda s: s[0])
        return snapshots

    def _load_file(self, fpath):
        """단일 파일에서 스냅샷 읽기"""
        results = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ts = obj.get("timestamp", 0)
                    aircraft = obj.get("aircraft", [])
                    if aircraft:
                        results.append((ts, aircraft))
                except json.JSONDecodeError:
                    continue
        return results

    def run(self):
        if not self._snapshots:
            print("[Replay] No snapshots to replay")
            return

        print(f"[Replay] Starting playback...")
        while not self.stop_event.is_set():
            prev_ts = None

            for i, (ts, aircraft_data) in enumerate(self._snapshots):
                if self.stop_event.is_set():
                    break

                # 대기 시간 계산
                if prev_ts is not None and self.speed > 0:
                    gap = ts - prev_ts
                    wait = gap / self.speed
                    if wait > 0:
                        # 긴 갭은 최대 2초로 제한 (녹화 중 빈 구간 건너뛰기)
                        wait = min(wait, 2.0)
                        self.stop_event.wait(wait)
                        if self.stop_event.is_set():
                            break

                prev_ts = ts
                self.current_replay_ts = ts

                # 큐에 전달
                try:
                    self.aircraft_queue.put_nowait(aircraft_data)
                except Exception:
                    pass

            if not self.loop:
                break
            if not self.stop_event.is_set():
                print("[Replay] Looping from start...")

        print("[Replay] Playback stopped")

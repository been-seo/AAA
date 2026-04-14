"""
ADS-B 장기 녹화 스크립트 (headless, 백그라운드)

사용법:
    python record_adsb.py                       # 3일간 DB 녹화 (기본)
    python record_adsb.py --hours 72            # 시간 지정
    python record_adsb.py --format jsonl        # 레거시 JSONL 포맷
    python record_adsb.py --import-jsonl data/recordings/adsb_*.jsonl  # JSONL→DB 변환

Ctrl+C로 중단해도 데이터는 정상 저장됨.
"""
import sys
import os
import time
import queue
import signal
import threading
import argparse
import glob as glob_mod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.adsb_fetcher import ADSBFetcher, ADSBRecorder
from core.adsb_db import ADSBDatabase
from config import ADSB_INTERVAL


def record(args):
    duration_sec = args.hours * 3600
    rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.outdir)

    adsb_q = queue.Queue(maxsize=100)
    stop_event = threading.Event()

    # DB 또는 JSONL 레코더 선택
    if args.format == 'db':
        ts = time.strftime("%Y%m%d_%H%M%S")
        db_path = os.path.join(rec_dir, f"adsb_{ts}.db")
        recorder = ADSBDatabase(db_path)
    else:
        recorder = ADSBRecorder(rec_dir)

    fetcher = ADSBFetcher(adsb_q, stop_event, interval=args.interval)
    fetcher.recorder = recorder
    fetcher.start()

    def _signal_handler(sig, frame):
        print("\n[Record] Interrupted, saving...")
        stop_event.set()
    signal.signal(signal.SIGINT, _signal_handler)

    print(f"[Record] Recording for {args.hours:.0f}h ({duration_sec:.0f}s)")
    print(f"[Record] Format: {args.format} | Interval: {args.interval}s | Output: {recorder.filepath}")

    t0 = time.time()
    last_report = t0

    while not stop_event.is_set():
        elapsed = time.time() - t0
        if elapsed >= duration_sec:
            print(f"[Record] Duration reached ({args.hours:.0f}h)")
            break

        try:
            while True:
                adsb_q.get_nowait()
        except queue.Empty:
            pass

        if time.time() - last_report > 600:
            hrs = elapsed / 3600
            size_mb = os.path.getsize(recorder.filepath) / 1e6
            print(f"[Record] {hrs:.1f}h elapsed | {recorder.snapshot_count} snapshots | "
                  f"file: {size_mb:.1f}MB")
            last_report = time.time()

        stop_event.wait(timeout=5.0)

    stop_event.set()
    fetcher.join(timeout=5)
    recorder.close()
    print(f"[Record] Done. {recorder.snapshot_count} snapshots saved.")


def import_jsonl(args):
    """기존 JSONL 파일을 SQLite DB로 변환"""
    rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.outdir)
    db_path = os.path.join(rec_dir, "adsb_merged.db")

    # glob 패턴 확장
    files = []
    for pattern in args.import_jsonl:
        files.extend(glob_mod.glob(pattern))
    files = sorted(set(files))

    if not files:
        print("[Import] No JSONL files found")
        return

    print(f"[Import] Converting {len(files)} JSONL files → {db_path}")

    db = ADSBDatabase(db_path)
    for fpath in files:
        print(f"  Processing: {fpath}")
        db.import_jsonl(fpath)

    total = db.get_snapshot_count()
    unique = db.get_aircraft_count()
    t_min, t_max = db.get_time_range()
    hours = (t_max - t_min) / 3600 if t_max > t_min else 0

    print(f"\n[Import] Complete:")
    print(f"  Snapshots: {total}")
    print(f"  Unique aircraft: {unique}")
    print(f"  Time span: {hours:.1f}h")
    print(f"  DB file: {db_path} ({os.path.getsize(db_path)/1e6:.1f}MB)")
    db.close()


def main():
    parser = argparse.ArgumentParser(description="ADS-B recorder (SQLite/JSONL)")
    parser.add_argument("--hours", type=float, default=72, help="녹화 시간 (기본 72h)")
    parser.add_argument("--interval", type=int, default=ADSB_INTERVAL,
                        help=f"수신 간격 초 (기본 {ADSB_INTERVAL})")
    parser.add_argument("--outdir", default="data/recordings", help="출력 디렉토리")
    parser.add_argument("--format", choices=['db', 'jsonl'], default='db',
                        help="녹화 포맷 (기본: db)")
    parser.add_argument("--import-jsonl", nargs='+', metavar='FILE',
                        help="JSONL 파일을 SQLite DB로 변환")
    args = parser.parse_args()

    if args.import_jsonl:
        import_jsonl(args)
    else:
        record(args)


if __name__ == "__main__":
    main()

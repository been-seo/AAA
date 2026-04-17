"""
ATC-AI 통합 설정 파일
모든 상수, 경로, 하이퍼파라미터를 한 곳에서 관리
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 화면 ──
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# ── 지도 ──
MAP_CENTER_LAT = 36.0
MAP_CENTER_LON = 128.0
INITIAL_ZOOM_LEVEL = 7
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "ATC-AI-Simulator/2.0"
TILE_SIZE = 256
TILE_CACHE_DIR = os.path.join(BASE_DIR, "data", "tile_cache_osm")
MAX_MEM_CACHE_SIZE = 500
TILE_REQUEST_TIMEOUT = 15

# ── 물리 단위 ──
KNOTS_TO_MPS = 0.514444
FT_TO_M = 0.3048
NM_TO_M = 1852.0
M_TO_NM = 1.0 / 1852.0
MAGNETIC_DECLINATION = 7.0  # 한국 자기 편각 (도)

# ── 항공기 기동 ──
HDG_RATE_QUICK = 15.0        # 도/초
HDG_RATE_NORMAL = 5.0        # 도/초
PEAK_ALT_RATE_NORMAL_FPS = 12000 / 60.0  # ft/초 (12000fpm)
ALT_RATE_QUICK = 20000 / 60.0           # ft/초 (20000fpm)
SPD_RATE_QUICK = 40.0        # kts/초
SPD_RATE_NORMAL = 20.0       # kts/초
INSTRUCTION_DELAY = 0.0      # 초 (학습 시 0)

# ── ADS-B ──
ADSB_BBOX = (31.0, 40.0, 122.0, 135.0)  # (south, north, west, east) 한국 FIR
ADSB_INTERVAL = 10  # 초
STALE_AIRCRAFT_CUTOFF = 120  # 초

# ── 색상 ──
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
RED    = (255, 0, 0)
BLUE   = (0, 0, 255)
GREEN  = (0, 255, 0)
CYAN   = (0, 255, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY   = (50, 50, 50)
LIGHT_GRAY = (150, 150, 150)

# ── TCAS II 파라미터 (DO-185B / EUROCAE ED-143 기반) ──
# Sensitivity Level 테이블: (alt_ceiling_ft, TA_params, RA_params)
#   TA/RA params = (tau_s, DMOD_nm, ZTHR_ft)   RA에는 ALIM_ft 추가
# SL2: 1000-2350ft (TA only, RA 없음)
# SL3~SL7: 고도에 따라 임계값 증가
TCAS_SL_TABLE = [
    # (alt_ceiling, tau_TA, DMOD_TA, ZTHR_TA,  tau_RA, DMOD_RA, ZTHR_RA, ALIM_RA)
    ( 1000,  20, 0.30,  850,  None, None, None, None),   # SL2 (TA only)
    ( 2350,  25, 0.33,  850,    15, 0.20,  600,  300),   # SL3
    ( 5000,  30, 0.48,  850,    20, 0.35,  600,  300),   # SL4
    (10000,  40, 0.75,  850,    25, 0.55,  600,  350),   # SL5
    (20000,  45, 1.00,  850,    30, 0.80,  600,  400),   # SL6
    (42000,  45, 1.00,  850,    30, 0.80,  600,  400),   # SL6 (extended)
    (99999,  48, 1.30, 1200,    35, 1.10,  700,  600),   # SL7
]

# ── RL 환경 ──
RL_SIM_STEP_SEC = 5.0
RL_ACTION_COUNT = 7201       # 7200 조합 + 1 (유지)
RL_MAX_TRAFFIC_OBS = 5
RL_OBS_DIM = 10 + 6 * 5     # 기본 10 + 주변항공기 30 = 40

# ── KADIZ 경계 (한국방공식별구역, 2013년 확장) ──
# 출처: CIMSEC (cimsec.org/tag/adiz/)
KADIZ_POLYGON = [
    (39.00, 123.50),   # 3900N 12330E (시작/끝)
    (39.00, 133.00),   # 3900N 13300E
    (37.28, 133.00),   # 3717N 13300E
    (36.00, 130.50),   # 3600N 13030E
    (35.22, 129.80),   # 3513N 12948E
    (34.72, 129.15),   # 3443N 12909E
    (34.28, 128.87),   # 3417N 12852E
    (32.50, 127.50),   # 3230N 12730E
    (32.50, 126.83),   # 3230N 12650E
    (30.00, 125.42),   # 3000N 12525E
    (30.00, 124.00),   # 3000N 12400E
    (37.00, 124.00),   # 3700N 12400E
]

# ── 공역 다각형 데이터 (airportal.go.kr 실 좌표) ──
import os as _os, json as _json, importlib.util as _imputil
_poly_path = _os.path.join(_os.path.dirname(__file__), "data", "airspace_polygons.py")
_AIRSPACE_POLYGONS = []
if _os.path.exists(_poly_path):
    _spec = _imputil.spec_from_file_location("airspace_polygons", _poly_path)
    _mod = _imputil.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _AIRSPACE_POLYGONS = _mod.AIRSPACE_POLYGONS

# P-구역 (고정 장애물, 항상 활성)
STATIC_OBSTACLES = [a for a in _AIRSPACE_POLYGONS
                    if a["type"] == "PRD" and a["name"].startswith("P")]
# DMZ: P518 다각형이 MDL까지 확장되어 별도 원형 근사 불필요

# R-구역 (제한공역) — Hot/Cold 토글 대상
R_ZONE_LIST = [a for a in _AIRSPACE_POLYGONS
               if a["type"] == "PRD" and a["name"].startswith("R")]
R_ZONE_TOGGLE_INTERVAL_SEC = 180  # R구역은 3분 간격 토글

# MOA (군사작전구역) — Hot/Cold 토글 대상
MOA_TOGGLE_INTERVAL_SEC = 120
MOA_LIST = [a for a in _AIRSPACE_POLYGONS if a["type"] == "MOA"]

# ── 공항/비행장 ──
# mil=True: 군용(전투기 출발/도착 가능), mil=False: 민간(접근 금지 대상)
AIRPORT_EXCLUSION_RADIUS_NM = 10.0
AIRPORT_EXCLUSION_ALT_FT = 10000
AIRPORTS = [
    # 민간 공항 (접근 금지 대상)
    {"icao": "RKSI", "name": "Incheon",    "lat": 37.4602, "lon": 126.4407, "mil": False},
    {"icao": "RKSS", "name": "Gimpo",      "lat": 37.5586, "lon": 126.7906, "mil": False},
    {"icao": "RKPC", "name": "Jeju",       "lat": 33.5104, "lon": 126.4928, "mil": False},
    {"icao": "RKJY", "name": "Yeosu",      "lat": 34.8424, "lon": 127.6170, "mil": False},
    {"icao": "RKPU", "name": "Ulsan",      "lat": 35.5935, "lon": 129.3518, "mil": False},
    {"icao": "RKJB", "name": "Muan",       "lat": 34.9914, "lon": 126.3828, "mil": False},
    {"icao": "RKNY", "name": "Yangyang",   "lat": 38.0613, "lon": 128.6693, "mil": False},
    {"icao": "RKPD", "name": "Jeongseok",  "lat": 33.3996, "lon": 126.7118, "mil": False},
    {"icao": "RKTL", "name": "Uljin",      "lat": 36.7231, "lon": 129.4631, "mil": False},
    # 민군 공용 (G-M, 전투기 출발/도착 가능)
    {"icao": "RKPK", "name": "Gimhae",     "lat": 35.1795, "lon": 128.9382, "mil": True},
    {"icao": "RKTN", "name": "Daegu",      "lat": 35.8941, "lon": 128.6586, "mil": True},
    {"icao": "RKJJ", "name": "Gwangju",    "lat": 35.1264, "lon": 126.8089, "mil": True},
    {"icao": "RKTU", "name": "Cheongju",   "lat": 36.7220, "lon": 127.4987, "mil": True},
    {"icao": "RKPS", "name": "Sacheon",    "lat": 35.0886, "lon": 128.0703, "mil": True},
    {"icao": "RKNW", "name": "Wonju",      "lat": 37.4381, "lon": 127.9604, "mil": True},
    {"icao": "RKTH", "name": "Pohang",     "lat": 35.9879, "lon": 129.4204, "mil": True},
    {"icao": "RKSM", "name": "Seoul AB",   "lat": 37.4449, "lon": 127.1140, "mil": True},
    {"icao": "RKJK", "name": "Gunsan AB",  "lat": 35.9038, "lon": 126.6158, "mil": True},
    # 군 전용 비행장 (전투기 출발/도착)
    {"icao": "RKSO", "name": "Osan AB",    "lat": 37.0906, "lon": 127.0296, "mil": True},
    {"icao": "RKSW", "name": "Suwon AB",   "lat": 37.2393, "lon": 127.0078, "mil": True},
    {"icao": "RKSG", "name": "Humphreys",  "lat": 36.9622, "lon": 127.0311, "mil": True},
    {"icao": "RKTI", "name": "Jungwon AB", "lat": 36.9965, "lon": 127.8849, "mil": True},
    {"icao": "RKNN", "name": "Gangneung AB","lat": 37.7536, "lon": 128.9440, "mil": True},
    {"icao": "RKTY", "name": "Yecheon AB", "lat": 36.6319, "lon": 128.3553, "mil": True},
    {"icao": "RKTP", "name": "Seosan AB",  "lat": 36.7039, "lon": 126.4864, "mil": True},
]

# 군용 비행장만 (시나리오 출발/도착지 후보)
MILITARY_AIRBASES = [a for a in AIRPORTS if a["mil"]]

# ── Safety Advisor ──
SAFETY_SCAN_INTERVAL_SEC = 2.0
SAFETY_LOOKAHEAD_SEC = 300       # 5분 전방 예측
SAFETY_LOOKAHEAD_STEPS = 10
CONFLICT_HORIZ_NM = 5.0          # 수평 분리 기준
CONFLICT_VERT_FT = 1000.0        # 수직 분리 기준
WAKE_TURBULENCE_NM = 6.0         # 후류난기류 경계
TERRAIN_MIN_ALT_FT = 2000        # 최저 안전고도

# ── ATS 항로 (ENR 3.1) ──
# 항로 위를 따라 이동 금지, crossing만 가능
# 항로 근처의 공역 선을 따라 이동은 가능
import json as _json
_routes_path = _os.path.join(_os.path.dirname(__file__), "data", "ats_routes.json")
ATS_ROUTES = {}
if _os.path.exists(_routes_path):
    with open(_routes_path) as _f:
        ATS_ROUTES = _json.load(_f)
ATS_ROUTE_BUFFER_NM = 3.0  # 항로 중심선에서 이 거리 이내면 "항로 위" 판정
ATS_ALONG_PENALTY_NM = 5.0  # 항로를 따라 이 거리 이상 연속 비행하면 페널티

# ── 학습 하이퍼파라미터 ──
TRAIN_TOTAL_TIMESTEPS = int(1e7)
TRAIN_BUFFER_SIZE = 100_000
TRAIN_LEARNING_STARTS = 5_000
TRAIN_BATCH_SIZE = 128
TRAIN_GAMMA = 0.99
TRAIN_LEARNING_RATE = 1e-4
TRAIN_CHECKPOINT_FREQ = 100_000

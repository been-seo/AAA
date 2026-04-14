# ATC-AI: 군 항공관제 Safety Advisory AI

실시간 ADS-B 데이터 기반 전투기 관제 보조 시스템. World Model(궤적 예측) + Dreamer MBPO(위험도 학습)로 관제사에게 선제적 안전 경고를 제공한다.

## 핵심 아키텍처

```
ADS-B Feed (실시간/리플레이)
    ↓
Simulation (항공기 추적, 공역 관리)
    ↓
World Model (궤적 예측, 충돌 감지)
    ↓
Dreamer MBPO (상상 속 관제 → Value Function = 위험도 엔진)
    ↓
Safety Advisor (규칙 기반 + AI 위험도 → ACK 기반 경고)
    ↓
관제사 화면 (Pygame HUD)
```

## 실행

```bash
# 기본 시뮬레이터 (실시간 ADS-B + Safety Advisor)
python main.py

# ADS-B 리플레이
python main.py --replay data/recordings/adsb_merged.db --speed 10

# World Model 데모 (궤적 예측 + Flight Plan 시각화)
python demo_world_model.py --replay data/recordings/adsb_merged.db

# ADS-B 장기 녹화 (72시간 기본, SQLite)
python record_adsb.py --hours 72

# World Model 학습
python -m ai.world_model.trainer

# Dreamer MBPO 학습 (WM과 동시 실행 가능)
python train_dreamer.py --total-steps 100000
```

## 프로젝트 구조

```
ATC-AI/
├── main.py                  # 통합 진입점 (시뮬레이터 + Safety Advisor)
├── demo_world_model.py      # WM 데모 (궤적 예측, Flight Plan, Safety)
├── train_dreamer.py         # Dreamer MBPO 학습 스크립트
├── record_adsb.py           # ADS-B 장기 녹화 (DB/JSONL)
├── config.py                # 공역, 경로, 비행장 등 설정
├── requirements.txt
│
├── core/                    # 시뮬레이션 엔진
│   ├── simulation.py        # 메인 시뮬레이션 루프
│   ├── aircraft.py          # 항공기 상태 + 제어
│   ├── airspace.py          # MOA/R-zone, 전투기 순찰, ATS 항로
│   ├── adsb_fetcher.py      # 실시간/리플레이 ADS-B 수신
│   ├── adsb_db.py           # SQLite 기반 ADS-B 녹화
│   ├── flight_plan.py       # 레코딩 → 이륙 Flight Plan 추출
│   └── map_renderer.py      # 지도 투영 + 렌더링
│
├── ai/                      # AI 모듈
│   ├── safety_advisor.py    # ACK 기반 안전 경고 시스템
│   └── world_model/
│       ├── trajectory_predictor.py  # Transformer 궤적 예측 모델
│       ├── dataset.py               # JSONL + DB → 학습 데이터셋
│       ├── trainer.py               # World Model 학습기
│       ├── dreamer_policy.py        # Actor-Critic + EventInjector (MBPO)
│       └── conflict_detector.py     # 예측 궤적 기반 충돌 감지
│
├── gui/
│   └── control_panel.py     # PyQt5 관제 패널
│
├── utils/
│   ├── geo.py               # 지리 계산 (haversine 등)
│   ├── actions.py           # 액션 유틸
│   └── rendering.py         # 텍스트 렌더링 헬퍼
│
├── data/recordings/         # ADS-B 녹화 데이터 (JSONL + DB)
└── models/
    ├── world_model/         # WM 체크포인트
    └── dreamer/             # MBPO 체크포인트 + 학습 로그 DB
```

## Safety Advisor 경고 체계

| 구분 | 경고 유형 | ACK 가능 | 설명 |
|------|-----------|----------|------|
| SEPARATION | 분리 위반 | O | 수평/수직 분리 기준 미달 |
| COLLISION | 충돌 위험 | O | 예측 궤적 기반 CPA 위반 |
| MSA | 최저안전고도 | X | 지형 고도 미달 |
| AIRSPACE_EXIT | 공역 이탈 | X | 지정 MOA/R-zone 이탈 |
| KADIZ_EXIT | KADIZ 이탈 | X | 방공식별구역 이탈 (외교적 민감) |
| SQUAWK_EMRG | 비상 스쿼크 | X | 7500/7600/7700 감지 |
| ATS_ALONG | ATS 항로 침범 | X | 전투기 ATS 항로 along-track (crossing만 허용) |

**ACK 시스템**: 모든 경고 발행 → 관제사 ACK → 동일 이벤트 억제. 상황 악화 시 재경고. 5분 만료. Non-ackable 경고는 행동으로만 해소.

## Dreamer MBPO

World Model이 환경 역할 → Actor가 관제를 "꿈꾸며" 학습 → Critic의 Value Function이 "이 상태가 얼마나 위험한지" 학습.

- **Critic.risk_score()** → Safety Advisor의 AI 위험도 엔진으로 활용
- **EventInjector**: 상상 중 돌발 이벤트 주입 (RTB 전투기, WX deviation, 7700, pop-up traffic, 공역 HOT)
- WM과 동시 학습: 50스텝마다 WM 자동 리로드

## 의존성

```bash
pip install pygame-ce numpy torch gymnasium requests zstandard scipy
```

- Python 3.10+
- PyTorch (CUDA 권장)
- pygame-ce 2.5+

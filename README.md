# AAA — AI Avoidance Advisor

실시간 ADS-B 데이터를 기반으로 군 항공관제 환경에서의 충돌 위험을 예측하고, 관제사에게 선제적 회피 자문을 제공하는 안전 자문 시스템.

## Overview

AAA는 확률적 궤적 예측(World Model)과 강화학습 기반 위험도 평가(Dreamer MBPO)를 결합하여, 규칙 기반 경고의 한계를 넘는 지능형 안전 자문을 실현한다.

### 주요 특징

- **확률적 충돌 예측**: Transformer 기반 World Model이 Monte Carlo 샘플링으로 항공기 미래 궤적을 예측하고, 충돌 확률을 정량화
- **군 관제 특화 경고 로직**: 민간 항공기 간 충돌은 민간 관제 소관으로 제외하고, 군-민 및 군-군 충돌 위험만을 감지. 동일 공역 내 훈련 중인 전투기 간 접근은 정상 상황으로 판단하여 제외
- **ACK 기반 경고 관리**: 경고 발행 → 관제사 확인(ACK) → 동일 이벤트 억제. 상황 악화 시 재경고 발행. 비상 경고(KADIZ 이탈, 비상 스쿼크 등)는 상황 해소 시에만 소멸
- **Dreamer MBPO 위험도 엔진**: 상상 기반 강화학습(Model-Based Policy Optimization)으로 관제 상황의 위험도를 학습. Critic의 Value Function이 Safety Advisor의 AI 위험도 평가기로 기능
- **Flight Plan 자동 추출**: ADS-B 레코딩에서 이륙 이벤트를 자동 추출하여, 사전 비행계획(FPL)과 유사한 정보를 제공
- **실시간 데이터 연동**: SQLite에 축적되는 ADS-B 녹화 데이터를 학습기가 런타임에 직접 조회. 재시작 없이 신규 데이터 반영 가능

## Architecture

```
ADS-B Feed (실시간 / 리플레이)
    │
    ▼
Simulation Engine ─── Flight Plan Extractor
    │
    ▼
World Model ──── Monte Carlo Prediction ───▶ Conflict Detector
    │                                              │
    ▼                                              ▼
Dreamer MBPO ──── Value Function ──────────▶ Safety Advisor
    │                                              │
    ▼                                              ▼
Event Injector (돌발 상황 시뮬레이션)        ACK-based Alert → Controller HUD
```

## Installation

### 요구사항

- Python 3.10 이상
- PyTorch (CUDA 지원 권장)
- pygame-ce 2.5 이상

### 설치

```bash
pip install pygame-ce numpy torch gymnasium requests zstandard scipy
```

## Usage

```bash
# 시뮬레이터 실행 (실시간 ADS-B + Safety Advisor)
python main.py

# ADS-B 리플레이 (배속 지원)
python main.py --replay data/recordings/adsb_merged.db --speed 10

# 통합 데모 (궤적 예측, Flight Plan, Conflict Detection, Safety Alert)
python demo_world_model.py --replay data/recordings/adsb_merged.db

# ADS-B 장기 녹화 (기본 72시간, SQLite)
python record_adsb.py --hours 72

# World Model 학습
python -m ai.world_model.trainer

# Dreamer MBPO 학습 (World Model과 동시 실행 가능)
python train_dreamer.py --total-steps 100000
```

## Project Structure

```
AAA/
├── main.py                  # 통합 진입점
├── demo_world_model.py      # 통합 데모 (예측, FPL, Conflict, Safety)
├── train_dreamer.py         # Dreamer MBPO 학습 스크립트
├── record_adsb.py           # ADS-B 장기 녹화
├── config.py                # 공역, ATS 항로, 비행장, KADIZ 설정
│
├── core/                    # 시뮬레이션 엔진
│   ├── simulation.py        # 메인 시뮬레이션 루프
│   ├── aircraft.py          # 항공기 상태 및 제어
│   ├── airspace.py          # MOA/R-zone 관리, 전투기 순찰, ATS 항로
│   ├── adsb_fetcher.py      # 실시간/리플레이 ADS-B 수신기
│   ├── adsb_db.py           # SQLite 기반 ADS-B 녹화 관리
│   ├── flight_plan.py       # 레코딩 기반 Flight Plan 추출
│   └── map_renderer.py      # OSM 타일맵 렌더링 (디스크 캐시 지원)
│
├── ai/                      # AI 모듈
│   ├── safety_advisor.py    # ACK 기반 안전 경고 시스템
│   └── world_model/
│       ├── trajectory_predictor.py  # Transformer 궤적 예측 모델
│       ├── dataset.py               # DB 런타임 읽기 데이터셋
│       ├── trainer.py               # World Model 학습기
│       ├── dreamer_policy.py        # Actor-Critic + Event Injector (MBPO)
│       └── conflict_detector.py     # Monte Carlo 충돌 확률 예측
│
├── gui/
│   └── control_panel.py     # PyQt5 관제 패널
│
├── utils/                   # 유틸리티
│   ├── geo.py               # 지리 연산 (haversine 등)
│   ├── actions.py           # 액션 유틸리티
│   └── rendering.py         # 텍스트 렌더링
│
├── data/                    # 정적 데이터 (공역 정의, ATS 항로)
├── data/recordings/         # ADS-B 녹화 데이터 (gitignore)
└── models/                  # 학습 체크포인트 (gitignore)
```

## Alert System

### 경고 분류

| 카테고리 | 유형 | ACK 가능 | 감지 대상 | 설명 |
|----------|------|----------|-----------|------|
| CONFLICT | 충돌 예측 | O | 군-민, 군-군 | Monte Carlo 기반 확률적 충돌 예측 |
| SEPARATION | 분리 위반 | O | 군-민, 군-군 | 수평/수직 분리 기준 미달 |
| MSA | 최저안전고도 | X | 관제 대상 전투기 | 지형 고도 미달 |
| AIRSPACE_EXIT | 공역 이탈 | X | 관제 대상 전투기 | 할당 MOA/R-zone 경계 이탈 |
| KADIZ_EXIT | KADIZ 이탈 | X | 관제 대상 전투기 | 방공식별구역 이탈 |
| SQUAWK_EMRG | 비상 스쿼크 | X | 전체 | 7500(피랍)/7600(통신불능)/7700(비상) |
| ATS_ALONG | ATS 항로 침범 | X | 관제 대상 전투기 | ATS 항로 along-track 진입 (crossing만 허용) |

### 충돌 감지 규칙

- **민-민**: 감지 제외 (민간 관제 소관)
- **군-민**: 항상 감지 (단, 공항 접근/이륙 패턴의 저고도 민간 항공기는 제외)
- **군-군 (동일 공역)**: 감지 제외 (정상 훈련 기동)
- **군-군 (비할당 공역 통과 / 인접 공역 경계 접근)**: 감지

### ACK 상태 관리

모든 경고는 발행 즉시 관제사에게 표시되며, ACK 입력 시 동일 이벤트의 반복 표시를 억제한다. 아래 조건에서 ACK가 무효화되어 재경고가 발행된다:

- 상황 심각도 상승 (severity escalation)
- ACK 후 300초 경과 (시간 기반 만료)
- Non-ackable 카테고리 (AIRSPACE_EXIT, KADIZ_EXIT, SQUAWK_EMRG, MSA, ATS_ALONG)는 상황 해소 시에만 소멸

## License

Private repository. All rights reserved.

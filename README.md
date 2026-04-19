# AAA — AI Avoidance Advisor

항공관제 환경에서 충돌 위험을 예측하고, 통제사에게 선제적 회피 자문을 제공하는 AI 안전 자문 시스템. 본 시스템은 advisory tool이며, 모든 권고는 비강제적이다.

## Overview

AAA는 확률적 궤적 예측(World Model: GRU + VAE + Attention + Delta 예측)과 다목적 강화학습 위험도 평가(Dreamer MBPO + PAVING φ-flow)를 결합하여, 규칙 기반 경고의 한계를 넘는 지능형 안전 자문을 실현한다.

### 핵심 기능

- **확률적 충돌 예측**: GRU 시퀀스 모델 + VAE 확률적 잠재 변수 + Attention 기반 항공기 상호작용. Delta 예측(변화량) 방식으로 물리적 유효성 보장. MC 샘플링으로 충돌 확률 정량화 (셔플 pairing)
- **8 Inner Task 다목적 학습**: PAVING 프레임워크 기반 태스크 분해. 안전(수평분리/수직분리/고도), 효율(연료/경로), 임무(진행/고도매칭). φ-flow 단순 합산으로 학습, max|cos| 진단으로 직교성 실시간 모니터링
- **AI + 룰 이중 안전망**: World Model 기반 AI 탐지(주)와 closing rate 기반 룰 탐지(보조). 속도 기반 스캔 범위 자동 조정
- **ACK 기반 경고 관리**: 경고 발행 → 통제사 확인(ACK) → 동일 이벤트 억제. 비상 경고(KADIZ 이탈, 스쿼크 7700/7600/7500)는 상황 해소 시에만 소멸
- **Human-in-the-loop 데모**: 통제사가 직접 항공기를 생성·관제하며 AI 경고를 실시간 확인

## Architecture

```
공중상황 데이터 (ADS-B 항적)
    │
    ▼
Simulation Engine (항공기 물리 모델 + 공역 관리)
    │
    ├──▶ World Model (GRU+VAE+Attn, Delta 예측, MC 충돌 확률)
    ├──▶ Dreamer MBPO (8 inner task, PAVING φ-flow)
    │       Safety:  sep_h, sep_v, alt_floor
    │       Efficiency: fuel, direct
    │       Mission: progress, alt_match
    └──▶ Rule Engine (근거리 보조, closing rate)
            │
            ▼
      Safety Advisor (통합 경고, ACK 관리, 3축 위험도)
            │
            ▼
      통제사 화면 (CONFLICT / RISK FACTORS / SAFETY ALERTS)
```

### 학습 파이프라인

```
녹화 데이터 → World Model 학습 (Delta 예측, 칼만 LR 스케줄링)
           → Dreamer MBPO:
             에피소드 = 실제 트래픽 속에서 내 항공기 1대 관제
             8 inner task 보상 → φ = ΣL_k 합산 (PAVING CANON)
             max|cos(∇L_k, ∇L_l)| 모니터링 (τ=0.5)
             certificate h monotonic 감소 확인
             Event Injector (돌발 상황 주입, ~10.5% 확률)
           → 체크포인트 배포 (ONNX/TensorRT 변환 가능)
```

## Installation

```bash
pip install pygame-ce numpy torch gymnasium requests zstandard scipy flask
```

- Python 3.10+
- PyTorch (CUDA 권장)
- pygame-ce 2.5+

## Usage

```bash
# Human ATC 데모 (통제사 관제 + AI 경고)
python demo_human_atc.py --replay data/recordings --speed 3

# 시뮬레이터 (실시간 + Safety Advisor)
python main.py

# World Model 학습
python -u -m ai.world_model.trainer --data-dir data/recordings --epochs 200

# Dreamer MBPO 학습
python train_dreamer.py --total-steps 1000000000 --horizon 200

# 학습 대시보드
python dashboard.py  # http://localhost:5556

# ADS-B 녹화
python record_adsb.py --hours 72
```

### Human ATC 데모 조작

| 키 | 기능 |
|---|---|
| N | 항공기 생성 (PyQt5 GUI) |
| I | 선택 항공기 지시 (HDG/ALT/SPD + Quick ALT) |
| T | 시나리오 자동 생성 (departure/arrival) |
| D | Quick Descent/Climb (±5000ft) |
| C | 선택 해제 |
| F | 선택 항공기 추적 |
| A | 경고 ACK |
| 1-3 | 예측 범위 (2/5/10분) |
| SPACE | AI 일시정지 |

## Project Structure

```
AAA/
├── main.py                  # 통합 진입점
├── demo_human_atc.py        # Human ATC + AI Advisory 데모
├── demo_world_model.py      # World Model 데모
├── train_dreamer.py         # Dreamer MBPO 학습
├── dashboard.py             # Flask 학습 대시보드
├── record_adsb.py           # ADS-B 녹화
├── config.py                # 공역, ATS 항로, 비행장 설정
│
├── core/                    # 시뮬레이션 엔진
│   ├── simulation.py        # 메인 루프
│   ├── aircraft.py          # 항공기 물리 모델 (연료, 기동)
│   ├── airspace.py          # MOA/R-zone, 전투기 순찰
│   ├── adsb_fetcher.py      # 실시간/리플레이 수신
│   ├── adsb_db.py           # SQLite 녹화
│   ├── flight_plan.py       # Flight Plan 추출
│   └── map_renderer.py      # OSM 타일맵
│
├── ai/                      # AI 모듈
│   ├── safety_advisor.py    # 통합 경고 (AI+룰, ACK, 3축 위험도)
│   └── world_model/
│       ├── trajectory_predictor.py  # GRU+VAE+Attention, Delta 예측
│       ├── physics.py               # 물리 모델 (reachable envelope)
│       ├── dataset.py               # DB → 텐서 (증분 캐시)
│       ├── trainer.py               # WM 학습 (칼만 LR)
│       ├── dreamer_policy.py        # 8 inner task, PAVING φ-flow
│       └── conflict_detector.py     # MC 충돌 확률 (셔플 pairing)
│
├── gui/
│   └── control_panel.py     # PyQt5 관제 패널 (Quick ALT)
│
└── utils/
    ├── geo.py               # haversine, bearing
    ├── actions.py           # 액션 유틸리티
    └── rendering.py         # 텍스트 렌더링
```

## Training

### World Model (Delta 예측)

입력: 과거 6스텝(60초) 항공기 상태 (10D: lat, lon, alt, gs, track, vrate, ias, mach, wind_dir, wind_spd)
출력: 미래 12스텝(120초) 상태 변화량(Δ) 예측

- Delta 예측 방식: `state[t+1] = state[t] + Δ` (물리 clamp 적용)
- 각도(track, wind_dir)는 모델 내부에서 sin/cos 임베딩 처리
- 주변 항공기 8대의 상대 상태를 cross-attention으로 반영
- 칼만 필터 기반 적응형 LR 스케줄링

### Action Space (8,209 actions)

| 차원 | 범위 | 단위 |
|---|---|---|
| HDG | 0, 10, ..., 350 | 36개 목표 방위 |
| ALT | 7500, 8500, ..., 44500 | 38개 목표 고도 |
| SPD | -50, 0, +50 | 3개 속도 변경 |
| RATE | normal (12kfpm), quick (20kfpm) | 2개 |
| HOLD | 현재 유지 | 1개 |

### 8 Inner Tasks (PAVING)

| Group | Task | Feature | 보상 |
|---|---|---|---|
| Safety | sep_h | lat/lon 상대거리 | RA -100, TA -30 |
| Safety | sep_v | alt 차이 | <500ft -50, <1000ft -20 |
| Safety | alt_floor | 절대 고도 | <2000ft -60 |
| Efficiency | fuel | 속도, 진행도 | -gs/600 + progress |
| Efficiency | direct | hdg vs 목적지 방위 | cos(err) × 2 |
| Mission | progress | 거리 변화 | ±3/NM, 도달 +100 |
| Mission | alt_match | alt vs 목표 | 근접 시 -alt_err×2 |

### PAVING Framework

- **φ-flow**: `φ = ΣL_k` 단순 합산. Inner task 직교성 하에서 gradient conflict 없이 모든 task 동시 개선 (CANON, Remark 1)
- **max|cos(∇L_k, ∇L_l)|**: §8.9 진단 metric. τ=0.5 초과 시 CANON 위반 경고
- **Certificate h**: `(1/K)Σ L_k²` monotonic 감소 모니터링

## References

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014
- Janner et al., "When to Trust Your Model: Model-Based Policy Optimization", NeurIPS 2019
- Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination", ICLR 2020
- Hafner et al., "Mastering Diverse Domains through World Models", arXiv 2023 (DreamerV3)
- Seo, "Paving the Loss Landscape: Environment Design for Failure Predictability in Multi-Task Learning", 2026, doi: 10.5281/zenodo.19454888

## License

Private repository. All rights reserved.

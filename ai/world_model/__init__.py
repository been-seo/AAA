"""
World Model 기반 Safety Advisory 시스템

1. TrajectoryPredictor: RSSM 기반 항공기 궤적 예측 (확률적)
2. ConflictDetector: Monte Carlo rollout → conflict 확률 산출
3. ADS-B Dataset: 녹화 데이터 → 학습용 시퀀스 변환
4. Trainer: 학습 파이프라인
"""
from .trajectory_predictor import TrajectoryPredictor
from .conflict_detector import ConflictDetector

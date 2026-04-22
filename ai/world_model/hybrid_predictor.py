"""
Hybrid Predictor: World Context 기반 예측 전략 디스패처

항공기 카테고리와 상황에 따라 최적 예측기 선택:
- IFR 항로 → RouteFollowingPredictor (물리 기반, 결정론적)
- 군용기 MOA → MOABoundedPredictor (공역 경계 내 확률적)
- Transit → TransitPredictor (직선 물리)
- 불확실 → Neural WM fallback
"""
import numpy as np
import torch

from .world_context import (
    WorldContextBuilder, FlightMode, AircraftCategory, AircraftContext,
)
from .route_predictor import (
    RouteFollowingPredictor, TransitPredictor, MOABoundedPredictor,
    ApproachPredictor,
)


class HybridPredictor:
    """
    World Context 기반 hybrid 궤적 예측기

    ConflictDetector에서 neural WM 대신 사용.
    각 항공기의 상황을 파악하고 적절한 예측 전략을 선택.
    """

    def __init__(self, neural_model=None, device='cuda', dt_sec=10.0):
        """
        :param neural_model: TrajectoryPredictor (fallback용, optional)
        :param device: torch device
        :param dt_sec: 예측 시간 간격
        """
        self.neural_model = neural_model
        self.device = device
        self.dt = dt_sec

        # World context builder
        self.ctx_builder = WorldContextBuilder()

        # 카테고리별 예측기
        self.route_predictor = RouteFollowingPredictor(dt_sec=dt_sec)
        self.transit_predictor = TransitPredictor(dt_sec=dt_sec)
        self.moa_predictor = MOABoundedPredictor(dt_sec=dt_sec)
        self.approach_predictor = ApproachPredictor(dt_sec=dt_sec)

    def predict_single(self, icao, state_dict, history=None,
                       adsb_info=None, num_samples=50,
                       future_steps=12):
        """
        단일 항공기 궤적 예측

        :param icao: 항공기 식별자
        :param state_dict: 현재 상태 {lat, lon, alt, gs, track, vrate, ...}
        :param history: 과거 상태 리스트 (oldest first)
        :param adsb_info: ADS-B 메타 {category, aircraft_model, icao}
        :param num_samples: MC 샘플 수
        :param future_steps: 예측 스텝 수
        :return: (context, trajectories)
            context: AircraftContext
            trajectories: (num_samples, future_steps, STATE_DIM) numpy array
        """
        # ADS-B 정보가 없으면 icao에서 추론
        if adsb_info is None:
            adsb_info = {"icao": icao}

        # World context 분류
        ctx = self.ctx_builder.classify(state_dict, history, adsb_info)

        # Flight mode에 따라 예측기 선택
        if ctx.flight_mode == FlightMode.APPROACH and ctx.approach_airport:
            # 접근: Reachability-based (공항 반경 구름)
            traj = self.approach_predictor.predict(
                state_dict, ctx.approach_airport,
                future_steps=future_steps,
                num_samples=num_samples)
            return ctx, traj

        elif ctx.flight_mode == FlightMode.DEPARTURE:
            # 출발: 현재 헤딩으로 상승, 항로 매칭 시도
            if ctx.route_wps:
                traj = self.route_predictor.predict(
                    state_dict, ctx.route_wps,
                    future_steps=future_steps,
                    num_samples=num_samples,
                    confidence=0.5)  # 출발은 불확실성 높음
            else:
                dest = self._project_ahead(state_dict, 100)
                traj = self.transit_predictor.predict(
                    state_dict, dest,
                    future_steps=future_steps,
                    num_samples=num_samples)
            return ctx, traj

        elif ctx.flight_mode == FlightMode.ROUTE_FOLLOWING and ctx.route_wps:
            traj = self.route_predictor.predict(
                state_dict, ctx.route_wps,
                future_steps=future_steps,
                num_samples=num_samples,
                confidence=ctx.route_confidence)
            return ctx, traj

        elif ctx.flight_mode == FlightMode.MOA_TRANSIT:
            # Cold MOA 직항 관통 (야간/주말) — 목적지로 직선
            dest = ctx.transit_dest or self._project_ahead(state_dict, 150)
            traj = self.transit_predictor.predict(
                state_dict, dest,
                future_steps=future_steps,
                num_samples=num_samples)
            return ctx, traj

        elif ctx.flight_mode == FlightMode.MOA_PATROL and ctx.moa_vertices:
            traj = self.moa_predictor.predict(
                state_dict, ctx.moa_vertices,
                future_steps=future_steps,
                num_samples=num_samples)
            return ctx, traj

        elif ctx.flight_mode == FlightMode.TRANSIT and ctx.transit_dest:
            is_mil = ctx.category == AircraftCategory.MILITARY
            traj = self.transit_predictor.predict(
                state_dict, ctx.transit_dest,
                future_steps=future_steps,
                num_samples=num_samples,
                is_military=is_mil)
            return ctx, traj

        elif ctx.flight_mode == FlightMode.VFR_FREE:
            # VFR: 현재 헤딩으로 직진 + 넓은 분산
            dummy_dest = self._project_ahead(state_dict, 100)
            traj = self.transit_predictor.predict(
                state_dict, dummy_dest,
                future_steps=future_steps,
                num_samples=num_samples)
            return ctx, traj

        else:
            # UNKNOWN: neural WM fallback 또는 단순 직선
            traj = self._fallback_predict(
                state_dict, history,
                num_samples=num_samples,
                future_steps=future_steps)
            return ctx, traj

    def predict_batch(self, aircraft_states, num_samples=50,
                      future_steps=12):
        """
        여러 항공기 일괄 예측 (ConflictDetector 호환)

        :param aircraft_states: dict {icao: {state_dict, history, adsb_info}}
        :return: dict {icao: (context, trajectories)}
        """
        results = {}
        for icao, info in aircraft_states.items():
            state = info.get("state", info)
            history = info.get("history")
            adsb = info.get("adsb_info")
            ctx, traj = self.predict_single(
                icao, state, history, adsb,
                num_samples=num_samples,
                future_steps=future_steps)
            results[icao] = (ctx, traj)
        return results

    def _project_ahead(self, state, dist_nm):
        """현재 헤딩 방향으로 dist_nm 앞 좌표"""
        import math
        lat = state.get("lat", 0)
        lon = state.get("lon", 0)
        track = state.get("track", 0) or state.get("true_track_deg", 0)
        hdg_rad = math.radians(track)
        dest_lat = lat + dist_nm * math.cos(hdg_rad) / 60.0
        cos_lat = max(math.cos(math.radians(lat)), 0.5)
        dest_lon = lon + dist_nm * math.sin(hdg_rad) / (60.0 * cos_lat)
        return {"lat": dest_lat, "lon": dest_lon}

    def _fallback_predict(self, state_dict, history, num_samples=50,
                          future_steps=12):
        """
        Fallback: neural WM이 있으면 사용, 없으면 단순 직선 외삽
        """
        # Neural WM이 없으면 단순 직선
        dummy_dest = self._project_ahead(state_dict, 200)
        return self.transit_predictor.predict(
            state_dict, dummy_dest,
            future_steps=future_steps,
            num_samples=num_samples)

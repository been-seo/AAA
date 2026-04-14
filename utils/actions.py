"""RL 이산 행동 ↔ 관제 지시 변환"""
import numpy as np


def action_to_instruction(action_index):
    """이산 행동 인덱스 → (hdg, alt, spd) dict. 7200이면 None (유지)."""
    if action_index == 7200:
        return None
    hdg_index = action_index // (20 * 5)
    alt_index = (action_index % (20 * 5)) // 5
    spd_index = action_index % 5
    return {
        "hdg": (-180 + hdg_index * 5) % 360,
        "alt": 5000 + alt_index * 500,
        "spd": 250 + spd_index * 50,
    }


def instruction_params_to_action_index(target_hdg, target_alt, target_spd):
    """(hdg, alt, spd) → 이산 행동 인덱스"""
    alt_index = int(round((np.clip(target_alt, 5000, 14500) - 5000) / 500.0))
    alt_index = max(0, min(19, alt_index))
    spd_index = int(round((np.clip(target_spd, 250, 450) - 250) / 50.0))
    spd_index = max(0, min(4, spd_index))
    hdg_centered = (target_hdg + 180) % 360 - 180
    hdg_index = int(round((hdg_centered + 180) / 5.0))
    hdg_index = max(0, min(71, hdg_index))
    return hdg_index * (20 * 5) + alt_index * 5 + spd_index

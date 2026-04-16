"""
Calibration utilities: extract realistic parameter bounds from RBI historical data.

Called once at startup if you want to auto-calibrate; otherwise the defaults
in config.py are already based on a manual study of RBI CPI data.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

try:
    from data.data_loader import load_rbi_historical_data
    _CAN_LOAD = True
except Exception:
    _CAN_LOAD = False


def calibrate_from_rbi(filepath: str) -> dict:
    """
    Derive shock volatility and inflation range from RBI CPI data.

    Args:
        filepath: path to DBIE CPI Excel file

    Returns:
        dict of calibrated parameters (can be used to override config)
    """
    if not _CAN_LOAD:
        raise ImportError("Cannot import data_loader. Check openpyxl is installed.")

    df = load_rbi_historical_data(filepath)
    pi = df["inflation"].dropna().values

    # First-difference for shock volatility estimation
    d_pi = np.diff(pi)

    params = {
        "mean_inflation":   float(np.mean(pi)),
        "std_inflation":    float(np.std(pi)),
        "max_inflation":    float(np.max(pi)),
        "min_inflation":    float(np.min(pi)),
        "sigma_s_estimate": float(np.std(d_pi)),    # supply shock scale
        "sigma_d_estimate": float(np.std(d_pi) * 0.6),  # demand shock (smaller)
        "pi_clip_upper":    float(np.percentile(pi, 99) * 1.5),  # safety headroom
    }

    print("[Calibration] RBI-derived parameters:")
    for k, v in params.items():
        print(f"  {k:25s} = {v:.6f}")

    return params


def get_action_bounds_from_rbi(filepath: str) -> Tuple[float, float]:
    """
    Return (i_min, i_max) based on observed RBI policy rates.
    Since the file only has CPI, we infer bounds from Taylor-implied rates.
    """
    params = calibrate_from_rbi(filepath)
    # Taylor rule at mean + 2-sigma would be our upper bound estimate
    pi_high = params["mean_inflation"] + 2.0 * params["std_inflation"]
    i_max   = min(0.15, 0.02 + pi_high + 1.5 * (pi_high - 0.04))
    return 0.0, float(np.clip(i_max, 0.08, 0.20))

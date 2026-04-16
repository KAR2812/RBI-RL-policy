"""
Fixed Inflation Targeting baseline policy.

The simplest possible rule: always set the nominal rate to
    i_t = r* + pi*
regardless of current conditions.

This is a useful ultra-simple baseline to bound how badly
a "do nothing" policy performs.
"""
import numpy as np
from config.config import Config, DEFAULT_CONFIG


class FixedTargetPolicy:
    """Always sets i = r* + pi* (no state dependency)."""

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.env_cfg = config.env
        self._i_fixed = float(
            np.clip(
                config.env.r_star + config.env.pi_target,
                config.env.i_min,
                config.env.i_max,
            )
        )

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return fixed rate regardless of observation."""
        return np.array([self._i_fixed], dtype=np.float32)

    def __repr__(self) -> str:
        return f"FixedTargetPolicy(i={self._i_fixed:.4f})"

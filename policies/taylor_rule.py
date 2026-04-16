"""
Taylor Rule baseline policy.

Formula:
    i_t = r* + pi_t + alpha * (pi_t - pi*) + beta_y * y_t

This implements the standard Taylor (1993) principle with alpha > 1,
guaranteeing a stronger-than-one response to inflation gaps so that
the real interest rate rises when inflation is above target.
"""
import numpy as np
from config.config import Config, DEFAULT_CONFIG, TaylorConfig, EnvConfig


class TaylorRulePolicy:
    """Deterministic Taylor Rule central bank policy."""

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.cfg:     TaylorConfig = config.taylor
        self.env_cfg: EnvConfig    = config.env

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute interest rate from observation.

        Args:
            obs: array [pi, y, i_prev, E_pi]

        Returns:
            action: array([i_t])  clipped to [i_min, i_max]
        """
        pi, y, _i_prev, _E_pi = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])

        i_t = (
            self.env_cfg.r_star
            + pi
            + self.cfg.alpha * (pi - self.env_cfg.pi_target)
            + self.cfg.beta_y * y
        )
        i_t = float(np.clip(i_t, self.env_cfg.i_min, self.env_cfg.i_max))
        return np.array([i_t], dtype=np.float32)

    def __repr__(self) -> str:
        return (
            f"TaylorRulePolicy(r*={self.env_cfg.r_star}, "
            f"alpha={self.cfg.alpha}, beta_y={self.cfg.beta_y}, "
            f"pi*={self.env_cfg.pi_target})"
        )

"""
Central configuration for all hyperparameters.
All magic numbers live here — nowhere else.
"""
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any
import os


@dataclass
class EnvConfig:
    # ── Macroeconomic model parameters ────────────────────────────────────────
    # Calibrated from New Keynesian literature & RBI data plausibility
    a: float = 0.5            # IS curve slope (interest-rate sensitivity)
    beta: float = 0.99        # Phillips curve: discount / inflation persistence
    kappa: float = 0.15       # Phillips curve: output-gap sensitivity
    rho_exp: float = 0.75     # Expectation persistence (adaptive expectations)

    # ── Targets ───────────────────────────────────────────────────────────────
    pi_target: float = 0.04   # Inflation target (4 % — RBI mandate)
    r_star: float = 0.02      # Natural real interest rate

    # ── Action / observation bounds ───────────────────────────────────────────
    i_min: float = 0.0        # Minimum nominal interest rate (ZLB)
    i_max: float = 0.15       # Maximum nominal interest rate (RBI upper bound)

    pi_clip: Tuple[float, float] = (-0.10, 0.40)   # Inflation clipping
    y_clip:  Tuple[float, float] = (-0.30, 0.30)   # Output gap clipping

    # ── Episode length ────────────────────────────────────────────────────────
    max_steps: int = 200


@dataclass
class ShockConfig:
    # AR(1) shock parameters
    rho_d: float = 0.70        # Demand shock persistence
    rho_s: float = 0.60        # Supply shock persistence
    sigma_d: float = 0.008     # Demand shock volatility
    sigma_s: float = 0.010     # Supply shock volatility


@dataclass
class RewardConfig:
    w_pi: float = 2.0          # Weight on inflation deviation
    w_y:  float = 1.0          # Weight on output gap
    w_i:  float = 0.5          # Weight on interest-rate change (smoothing)
    shock_penalty: float = 0.0 # Additional penalty during shock windows
    clip_low: float = -20.0    # Reward lower clip
    clip_high: float = 0.0     # Reward upper clip


@dataclass
class TaylorConfig:
    alpha: float = 1.5         # Inflation gap coefficient (Taylor principle: >1)
    beta_y: float = 0.5        # Output gap coefficient


@dataclass
class PPOConfig:
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    net_arch: list = field(default_factory=lambda: [128, 128])
    total_timesteps: int = 200_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    save_freq: int = 10_000


@dataclass
class PathsConfig:
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir: str = "outputs"
    models_dir: str = "outputs/models"
    logs_dir: str = "outputs/logs"
    plots_dir: str = "outputs/plots"
    data_dir: str = "data"
    rbi_excel: str = "data/RBIB Table No. 19 _ Consumer Price Index (Base 2010=100).xlsx"

    def resolve(self, path: str) -> str:
        return os.path.join(self.base_dir, path)

    def ensure_dirs(self) -> None:
        for d in [self.models_dir, self.logs_dir, self.plots_dir]:
            os.makedirs(self.resolve(d), exist_ok=True)


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    shock: ShockConfig = field(default_factory=ShockConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    taylor: TaylorConfig = field(default_factory=TaylorConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


# Module-level default instance
DEFAULT_CONFIG = Config()

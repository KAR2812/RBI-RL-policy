"""
Standalone PPO training script.

Entry point:
    python rl/train.py  (called internally by main.py)

Or import:
    from rl.train import train_ppo_agent
"""
from __future__ import annotations

import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policies.rl_agent import RLAgentPolicy
from config.config import Config, DEFAULT_CONFIG
from env.shocks import ShockScenarioParams
from utils.seed import set_global_seed


def train_ppo_agent(
    config: Config = DEFAULT_CONFIG,
    total_timesteps: int = 200_000,
    seed: int = 42,
    save_dir: str = "outputs/models",
    log_dir: str = "outputs/logs",
) -> RLAgentPolicy:
    """
    Train a PPO agent on the MacroEconomicEnv.

    Args:
        config         : full Config dataclass
        total_timesteps: steps to train for
        seed           : global RNG seed
        save_dir       : where to save model checkpoints
        log_dir        : where to write training logs

    Returns:
        Trained RLAgentPolicy with model loaded.
    """
    set_global_seed(seed)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir  = os.path.join(base_dir, save_dir)
    log_dir   = os.path.join(base_dir, log_dir)

    shock_params = ShockScenarioParams(name="normal", seed=seed)

    print(f"[Train] Starting PPO training for {total_timesteps:,} timesteps, seed={seed}")
    agent = RLAgentPolicy(config, shock_params=shock_params)
    agent.train(
        total_timesteps=total_timesteps,
        save_dir=save_dir,
        log_dir=log_dir,
        seed=seed,
    )

    # Save final model as well
    agent.save(os.path.join(save_dir, "ppo_final"))
    print("[Train] Training complete.")
    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO agent on MacroEconomicEnv")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--save_dir",  type=str, default="outputs/models")
    parser.add_argument("--log_dir",   type=str, default="outputs/logs")
    args = parser.parse_args()

    train_ppo_agent(
        total_timesteps=args.timesteps,
        seed=args.seed,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )

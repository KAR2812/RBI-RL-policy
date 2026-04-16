"""
Policy evaluation runner.

Runs any policy (Taylor Rule, Fixed Target, or PPO) for a given number of
episodes on a specified shock scenario and returns full trajectory data
plus summary metrics.
"""
from __future__ import annotations

import os
import sys
import numpy as np
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config, DEFAULT_CONFIG
from env.macro_env import MacroEconomicEnv
from env.shocks import ShockScenarioParams
from utils.metrics import compute_metrics
from utils.logger import ExperimentLogger


def evaluate_policy(
    policy,
    config: Config = DEFAULT_CONFIG,
    shock_params: Optional[ShockScenarioParams] = None,
    n_episodes: int = 1,
    seed: int = 42,
    policy_name: str = "policy",
    log_dir: Optional[str] = None,
) -> dict[str, Any]:
    """
    Evaluate a policy for n_episodes and collect metrics.

    Args:
        policy      : object with .predict(obs) -> action
        config      : Config dataclass
        shock_params: ShockScenarioParams (default: normal, seed=seed)
        n_episodes  : number of evaluation episodes
        seed        : RNG seed for the environment
        policy_name : string label for logging
        log_dir     : if given, save per-step CSV there

    Returns:
        dict with keys:
            trajectories : list of per-episode trajectory dicts
            metrics      : aggregated metrics across episodes
    """
    if shock_params is None:
        shock_params = ShockScenarioParams(name="normal", seed=seed)

    all_trajectories = []
    all_rewards      = []

    for ep in range(n_episodes):
        ep_seed = seed + ep * 1000

        env = MacroEconomicEnv(
            config=config,
            shock_params=ShockScenarioParams(**{**shock_params.__dict__, "seed": ep_seed}),
            seed=ep_seed,
        )

        obs, _ = env.reset(seed=ep_seed)

        episode: dict[str, list] = {
            "step":          [],
            "inflation":     [],
            "output_gap":    [],
            "interest_rate": [],
            "E_pi":          [],
            "reward":        [],
            "eps_d":         [],
            "eps_s":         [],
            "shock_active":  [],
        }

        done = False
        ep_reward = 0.0

        while not done:
            action  = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode["step"].append(info["step"])
            episode["inflation"].append(info["pi"])
            episode["output_gap"].append(info["y"])
            episode["interest_rate"].append(info["i"])
            episode["E_pi"].append(info["E_pi"])
            episode["reward"].append(reward)
            episode["eps_d"].append(info["eps_d"])
            episode["eps_s"].append(info["eps_s"])
            episode["shock_active"].append(info["shock_active"])
            ep_reward += reward

        all_trajectories.append(episode)
        all_rewards.append(ep_reward)
        env.close()

    # ── Compute metrics across episodes ───────────────────────────────────────
    metrics = compute_metrics(
        trajectories=all_trajectories,
        pi_target=config.env.pi_target,
        w_y=config.reward.w_y,
        w_i=config.reward.w_i,
    )
    metrics["policy_name"]         = policy_name

    # ── Optional CSV logging ──────────────────────────────────────────────────
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        logger = ExperimentLogger(
            policy_name=policy_name,
            scenario_name=shock_params.name,
            log_dir=log_dir,
        )
        for ep in all_trajectories:
            for i in range(len(ep["step"])):
                logger.log_step(
                    step=ep["step"][i],
                    pi=ep["inflation"][i],
                    y=ep["output_gap"][i],
                    i_t=ep["interest_rate"][i],
                    E_pi=ep["E_pi"][i],
                    reward=ep["reward"][i],
                    eps_d=ep["eps_d"][i],
                    eps_s=ep["eps_s"][i],
                    shock_active=ep["shock_active"][i],
                )
        logger.save()

    return {"trajectories": all_trajectories, "metrics": metrics}

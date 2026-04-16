"""
PPO-based RL agent wrapper using Stable-Baselines3.

Provides train / load / predict interface matching the baseline policies
so experiment runners can treat all three policies identically.
"""
from __future__ import annotations

import os
import numpy as np
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)

from config.config import Config, DEFAULT_CONFIG
from env.macro_env import MacroEconomicEnv
from env.shocks import ShockScenarioParams


class RLAgentPolicy:
    """
    Wraps a Stable-Baselines3 PPO model.

    Usage:
        agent = RLAgentPolicy(config)
        agent.train(total_timesteps=200_000)
        action = agent.predict(obs)
        agent.save("outputs/models/ppo_final")
        agent.load("outputs/models/ppo_final")
    """

    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        shock_params: Optional[ShockScenarioParams] = None,
        model_path: Optional[str] = None,
    ):
        self.config = config
        self.ppo_cfg = config.ppo
        self.shock_params = shock_params or ShockScenarioParams(
            name="normal", seed=config.seed
        )
        self.model: Optional[PPO] = None

        if model_path and os.path.exists(model_path + ".zip"):
            self.load(model_path)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: Optional[int] = None,
        save_dir: str = "outputs/models",
        log_dir: str = "outputs/logs",
        seed: Optional[int] = None,
    ) -> PPO:
        """Train the PPO agent and return the trained model."""
        n_steps = total_timesteps or self.ppo_cfg.total_timesteps
        seed    = seed or self.config.seed
        cfg     = self.ppo_cfg

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir,  exist_ok=True)

        # ── Vectorised training environment (4 parallel workers) ──────────────
        def make_env():
            return MacroEconomicEnv(self.config, self.shock_params)

        train_env = make_vec_env(make_env, n_envs=4, seed=seed)

        # ── Evaluation environment (single, deterministic seed) ───────────────
        eval_env = make_vec_env(
            lambda: MacroEconomicEnv(
                self.config,
                ShockScenarioParams(name="normal", seed=seed + 1000),
            ),
            n_envs=1,
            seed=seed + 1000,
        )

        # ── Callbacks ─────────────────────────────────────────────────────────
        checkpoint_cb = CheckpointCallback(
            save_freq=max(cfg.save_freq // 4, 1),  # adjust for n_envs=4
            save_path=save_dir,
            name_prefix="ppo_checkpoint",
        )
        stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=10,
            min_evals=5,
            verbose=1,
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            log_path=log_dir,
            eval_freq=max(cfg.eval_freq // 4, 1),
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
            callback_after_eval=stop_cb,
            verbose=1,
        )

        # ── Build model ───────────────────────────────────────────────────────
        policy_kwargs = dict(net_arch=cfg.net_arch)
        self.model = PPO(
            policy=cfg.policy,
            env=train_env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            normalize_advantage=cfg.normalize_advantage,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(log_dir, "tb"),
            seed=seed,
            verbose=1,
        )

        self.model.learn(
            total_timesteps=n_steps,
            callback=[checkpoint_cb, eval_cb],
            reset_num_timesteps=True,
        )

        train_env.close()
        eval_env.close()

        # Load the best model found during eval
        best_path = os.path.join(save_dir, "best_model")
        if os.path.exists(best_path + ".zip"):
            self.model = PPO.load(best_path, env=None)
            print(f"[RLAgent] Loaded best model from {best_path}.zip")

        return self.model

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Predict action from observation.  Identical interface to baselines.

        Args:
            obs: array [pi, y, i_prev, E_pi]
        Returns:
            action: array([i_t])
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")
        action, _ = self.model.predict(obs, deterministic=True)
        return np.array(action, dtype=np.float32).reshape(1)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(path)
        print(f"[RLAgent] Model saved to {path}.zip")

    def load(self, path: str) -> None:
        self.model = PPO.load(path)
        print(f"[RLAgent] Model loaded from {path}.zip")

    def __repr__(self) -> str:
        status = "trained" if self.model else "untrained"
        return f"RLAgentPolicy(status={status}, alg=PPO)"

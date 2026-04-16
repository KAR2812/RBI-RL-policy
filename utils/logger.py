"""
Experiment logger: persists per-step data to CSV for later analysis.
"""
from __future__ import annotations

import os
import csv
from datetime import datetime
from typing import Any


class ExperimentLogger:
    """
    Logs per-step macro variables to a CSV file.

    Usage:
        logger = ExperimentLogger("taylor_rule", "demand_shock", "outputs/logs")
        for each step:
            logger.log_step(step, pi, y, i_t, E_pi, reward, ...)
        logger.save()
    """

    COLUMNS = [
        "step", "inflation", "output_gap", "interest_rate",
        "E_pi", "reward", "eps_d", "eps_s", "shock_active",
    ]

    def __init__(
        self,
        policy_name: str,
        scenario_name: str,
        log_dir: str = "outputs/logs",
    ):
        self.policy_name   = policy_name
        self.scenario_name = scenario_name
        self.log_dir       = log_dir
        self.rows: list[dict[str, Any]] = []

    def log_step(
        self,
        step: int,
        pi: float,
        y: float,
        i_t: float,
        E_pi: float,
        reward: float,
        eps_d: float = 0.0,
        eps_s: float = 0.0,
        shock_active: bool = False,
    ) -> None:
        self.rows.append({
            "step":          step,
            "inflation":     round(pi,    6),
            "output_gap":    round(y,     6),
            "interest_rate": round(i_t,   6),
            "E_pi":          round(E_pi,  6),
            "reward":        round(reward, 6),
            "eps_d":         round(eps_d, 6),
            "eps_s":         round(eps_s, 6),
            "shock_active":  int(shock_active),
        })

    def save(self) -> str:
        """Write rows to CSV and return the file path."""
        os.makedirs(self.log_dir, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.policy_name}_{self.scenario_name}_{ts}.csv"
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)

        print(f"[Logger] Saved {len(self.rows)} rows → {filepath}")
        return filepath

    def clear(self) -> None:
        self.rows.clear()

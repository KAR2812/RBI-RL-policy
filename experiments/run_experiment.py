"""
Full multi-policy, multi-scenario experiment runner.

Usage:
    from experiments.run_experiment import run_full_experiment
    results = run_full_experiment(config, model_path="outputs/models/best_model")

Or via CLI:
    python main.py --mode compare
"""
from __future__ import annotations

import os
import json
import numpy as np
from typing import Optional, Any

from config.config import Config, DEFAULT_CONFIG
from policies.taylor_rule import TaylorRulePolicy
from policies.fixed_target import FixedTargetPolicy
from policies.rl_agent import RLAgentPolicy
from rl.evaluate import evaluate_policy
from experiments.shock_scenarios import get_shock_scenario, ALL_SCENARIOS
from utils.plotting import (
    save_single_episode_plots,
    save_comparison_plot,
    save_cumulative_loss_comparison,
    save_recovery_comparison,
    save_metrics_summary_table,
)
from utils.seed import set_global_seed


def run_full_experiment(
    config: Config = DEFAULT_CONFIG,
    model_path: Optional[str] = None,
    scenarios: Optional[list[str]] = None,
    seed: int = 42,
    n_episodes: int = 1,
    output_dir: str = "outputs",
) -> dict[str, Any]:
    """
    Run all policies across all (or selected) shock scenarios.

    Args:
        config     : project Config
        model_path : path to saved PPO model (without .zip); if None, skips PPO
        scenarios  : list of scenario names (default: all 6)
        seed       : global seed
        n_episodes : evaluation episodes per policy/scenario
        output_dir : root output directory

    Returns:
        Nested dict: results[scenario][policy] = { trajectories, metrics }
    """
    set_global_seed(seed)

    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(base_dir, output_dir, "plots")
    logs_dir  = os.path.join(base_dir, output_dir, "logs")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir,  exist_ok=True)

    scenarios = scenarios or ALL_SCENARIOS

    # ── Build policy objects ───────────────────────────────────────────────────
    policies: dict = {
        "taylor_rule":  TaylorRulePolicy(config),
        "fixed_target": FixedTargetPolicy(config),
    }

    if model_path:
        full_model_path = os.path.join(base_dir, model_path)
        if os.path.exists(full_model_path + ".zip"):
            rl_agent = RLAgentPolicy(config, model_path=full_model_path)
            policies["ppo"] = rl_agent
            print(f"[Experiment] PPO model loaded from {full_model_path}.zip")
        else:
            print(f"[Experiment] WARNING: PPO model not found at {full_model_path}.zip — skipping PPO.")

    all_results: dict[str, dict] = {}

    for scenario_name in scenarios:
        print(f"\n{'='*60}")
        print(f"[Experiment] Scenario: {scenario_name.upper()}")
        print(f"{'='*60}")

        shock_params = get_shock_scenario(scenario_name, seed=seed)
        scenario_results: dict = {}

        # Per-scenario trajectories for overlay plot
        all_trajs_for_compare: dict[str, dict] = {}

        for policy_name, policy in policies.items():
            print(f"  → Evaluating {policy_name} …")

            result = evaluate_policy(
                policy=policy,
                config=config,
                shock_params=shock_params,
                n_episodes=n_episodes,
                seed=seed,
                policy_name=policy_name,
                log_dir=logs_dir,
            )

            scenario_results[policy_name] = result
            traj = result["trajectories"][0]  # first episode for plots
            all_trajs_for_compare[policy_name] = traj

            # Individual episode plots
            save_single_episode_plots(
                trajectory=traj,
                policy_name=policy_name,
                scenario_name=scenario_name,
                plot_dir=os.path.join(plots_dir, scenario_name),
                pi_target=config.env.pi_target,
            )

            m = result["metrics"]
            print(f"     mean_reward={m['mean_reward']:.4f} | "
                  f"inf_var={m['inflation_variance']:.6f} | "
                  f"cumul_loss={m['cumulative_loss']:.4f}")

        # ── Comparison plots for this scenario ─────────────────────────────────
        ref_traj     = next(iter(all_trajs_for_compare.values()))
        shock_flags  = np.array(ref_traj["shock_active"], dtype=bool)
        scenario_plot_dir = os.path.join(plots_dir, scenario_name)

        for metric_key, ylabel, title in [
            ("inflation",     "Inflation",     f"Inflation — {scenario_name}"),
            ("output_gap",    "Output Gap",    f"Output Gap — {scenario_name}"),
            ("interest_rate", "Interest Rate", f"Interest Rate — {scenario_name}"),
            ("reward",        "Reward",        f"Per-Step Reward — {scenario_name}"),
        ]:
            save_comparison_plot(
                trajectories=all_trajs_for_compare,
                metric_key=metric_key,
                ylabel=ylabel,
                title=title,
                plot_dir=scenario_plot_dir,
                filename=f"compare_{metric_key}_{scenario_name}",
                pi_target=config.env.pi_target if metric_key == "inflation" else None,
                shock_flags=shock_flags,
            )

        # ── Summary charts ─────────────────────────────────────────────────────
        metrics_map = {p: scenario_results[p]["metrics"] for p in scenario_results}

        save_cumulative_loss_comparison(
            metrics_dict=metrics_map,
            plot_dir=scenario_plot_dir,
            filename=f"cumulative_loss_{scenario_name}",
        )
        save_recovery_comparison(
            metrics_dict=metrics_map,
            plot_dir=scenario_plot_dir,
            filename=f"recovery_time_{scenario_name}",
        )
        save_metrics_summary_table(
            metrics_dict=metrics_map,
            plot_dir=scenario_plot_dir,
            filename=f"metrics_table_{scenario_name}",
        )

        all_results[scenario_name] = scenario_results

    # ── Save aggregated metrics JSON ──────────────────────────────────────────
    summary_path = os.path.join(logs_dir, "experiment_summary.json")
    _save_summary_json(all_results, summary_path)

    print(f"\n[Experiment] All done. Results saved to {output_dir}/")
    return all_results


def _save_summary_json(all_results: dict, path: str) -> None:
    """Serialise metrics (not trajectories) to JSON."""
    summary = {}
    for scenario, scenario_res in all_results.items():
        summary[scenario] = {}
        for policy, res in scenario_res.items():
            summary[scenario][policy] = res["metrics"]

    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[Experiment] Metrics summary saved → {path}")

"""
Standalone evaluation entry point.

Usage:
    python evaluate.py --policy taylor --scenario normal
    python evaluate.py --policy ppo    --scenario demand_shock --model outputs/models/best_model
    python evaluate.py --policy all    --scenario all
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import DEFAULT_CONFIG
from policies.taylor_rule import TaylorRulePolicy
from policies.fixed_target import FixedTargetPolicy
from policies.rl_agent import RLAgentPolicy
from rl.evaluate import evaluate_policy
from experiments.shock_scenarios import get_shock_scenario, ALL_SCENARIOS
from utils.plotting import save_single_episode_plots, save_comparison_plot
from utils.seed import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate a monetary policy")
    parser.add_argument("--policy",    type=str, default="taylor",
                        choices=["taylor", "fixed", "ppo", "all"])
    parser.add_argument("--scenario",  type=str, default="normal",
                        choices=ALL_SCENARIOS + ["all"])
    parser.add_argument("--model",     type=str, default="outputs/models/best_model",
                        help="Path to PPO model (without .zip)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--episodes",  type=int, default=1)
    parser.add_argument("--plot_dir",  type=str, default="outputs/plots")
    parser.add_argument("--log_dir",   type=str, default="outputs/logs")
    args = parser.parse_args()

    set_global_seed(args.seed)
    config = DEFAULT_CONFIG
    config.paths.ensure_dirs()

    # ── Select policies ────────────────────────────────────────────────────────
    sel_policies: dict = {}
    if args.policy in ("taylor", "all"):
        sel_policies["taylor_rule"] = TaylorRulePolicy(config)
    if args.policy in ("fixed", "all"):
        sel_policies["fixed_target"] = FixedTargetPolicy(config)
    if args.policy in ("ppo", "all"):
        model_path = os.path.join(os.path.dirname(__file__), args.model)
        agent = RLAgentPolicy(config, model_path=model_path)
        if agent.model is not None:
            sel_policies["ppo"] = agent
        else:
            print(f"[Evaluate] WARNING: PPO model not found at {model_path}.zip")

    # ── Select scenarios ───────────────────────────────────────────────────────
    sel_scenarios = ALL_SCENARIOS if args.scenario == "all" else [args.scenario]

    for scenario_name in sel_scenarios:
        shock_params = get_shock_scenario(scenario_name, seed=args.seed)
        trajs_for_compare = {}
        results_by_policy = {}

        for policy_name, policy in sel_policies.items():
            print(f"\n[Evaluate] Running {policy_name} | {scenario_name}...")
            result = evaluate_policy(
                policy=policy,
                config=config,
                shock_params=shock_params,
                n_episodes=args.episodes,
                seed=args.seed,
                policy_name=policy_name,
                log_dir=args.log_dir,
            )
            results_by_policy[policy_name] = result

            traj = result["trajectories"][0]
            trajs_for_compare[policy_name] = traj
            plot_dir = os.path.join(args.plot_dir, scenario_name)
            save_single_episode_plots(traj, policy_name, scenario_name, plot_dir,
                                      config.env.pi_target)

        # ── Output exact requested metrics ─────────────────────────────────────────
        best_loss = min(res["metrics"]["Total Macro Loss"] for res in results_by_policy.values())
        
        for policy_name, result in results_by_policy.items():
            m = result["metrics"]
            m["Policy Regret"] = m["Total Macro Loss"] - best_loss
            
            print(f"\n[Outputs] {policy_name} | {scenario_name}")
            print(f"  Total Macro Loss                             = {m['Total Macro Loss']:.4f}")
            print(f"  Inflation Loss                               = {m['Inflation Loss']:.4f}")
            print(f"  Unemployment Loss                            = {m['Unemployment Loss']:.4f}")
            print(f"  Interest Rate Smoothness (Policy Stability)  = {m['Interest Rate Smoothness']:.4f}")
            print(f"  Mean Return (RL Performance Score)           = {m['Mean Return']:.4f}")
            print(f"  Variance of Returns (Stability of Policy)    = {m['Variance of Returns']:.4f}")
            print(f"  Inflation Variance                           = {m['Inflation Variance']:.6f}")
            print(f"  Output Gap Variance                          = {m['Output Gap Variance']:.6f}")
            print(f"  Recovery Time After Shock                    = {m['Recovery Time After Shock']:.2f}")
            print(f"  Policy Reactivity                            = {m['Policy Reactivity']:.6f}")
            print(f"  Policy Regret                                = {m['Policy Regret']:.4f}")

            # Save exact 11 metrics into the logs directory
            import json
            metric_file = os.path.join(args.log_dir, f"{policy_name}_{scenario_name}_metrics.json")
            with open(metric_file, "w") as f:
                json.dump({
                    "Total Macro Loss": m['Total Macro Loss'],
                    "Inflation Loss": m['Inflation Loss'],
                    "Unemployment Loss": m['Unemployment Loss'],
                    "Interest Rate Smoothness (Policy Stability)": m['Interest Rate Smoothness'],
                    "Mean Return (RL Performance Score)": m['Mean Return'],
                    "Variance of Returns (Stability of Policy)": m['Variance of Returns'],
                    "Inflation Variance": m['Inflation Variance'],
                    "Output Gap Variance": m['Output Gap Variance'],
                    "Recovery Time After Shock": m['Recovery Time After Shock'],
                    "Policy Reactivity": m['Policy Reactivity'],
                    "Policy Regret": m['Policy Regret']
                }, f, indent=4)

        if len(trajs_for_compare) > 1:
            for metric, ylabel in [("inflation", "Inflation"),
                                    ("output_gap", "Output Gap"),
                                    ("interest_rate", "Interest Rate"),
                                    ("reward", "Reward")]:
                import numpy as np
                ref_traj = next(iter(trajs_for_compare.values()))
                shock_flags = np.array(ref_traj["shock_active"])
                save_comparison_plot(
                    trajectories=trajs_for_compare,
                    metric_key=metric,
                    ylabel=ylabel,
                    title=f"{ylabel} Comparison — {scenario_name}",
                    plot_dir=os.path.join(args.plot_dir, scenario_name),
                    filename=f"compare_{metric}_{scenario_name}",
                    pi_target=config.env.pi_target if metric == "inflation" else None,
                    shock_flags=shock_flags,
                )

    print("\n[Evaluate] Done.")


if __name__ == "__main__":
    main()

"""
Historical evaluation of monetary policies against real RBI data.
Evaluates the policies in an open-loop counterfactual setting to see what
rates they would have recommended given real historical economic states.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import DEFAULT_CONFIG
from policies.taylor_rule import TaylorRulePolicy
from policies.fixed_target import FixedTargetPolicy
from policies.rl_agent import RLAgentPolicy
from data.data_loader import load_us_historical_data
from utils.metrics import compute_metrics
from env.reward import compute_reward

def main():
    parser = argparse.ArgumentParser(description="Evaluate on Historical Data")
    parser.add_argument("--model", type=str, default="outputs/ppo_monetary_policy", help="Path to PPO model")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    config.paths.ensure_dirs()

    # Load policies
    policies = {
        "taylor_rule": TaylorRulePolicy(config),
        "fixed_target": FixedTargetPolicy(config)
    }

    model_path = os.path.join(os.path.dirname(__file__), args.model)
    agent = RLAgentPolicy(config, model_path=model_path)
    if agent.model is not None:
        policies["ppo"] = agent

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    print("[Historical] Loading FRED Historical Dataset...")
    df = load_us_historical_data(data_dir)

    # Compute 3-month MA for expected inflation (E_pi)
    df['E_pi'] = df['inflation'].rolling(window=3, min_periods=1).mean()

    dates = df['date'].values
    hist_pi = df['inflation'].values
    hist_y = df['output_gap'].values
    hist_E_pi = df['E_pi'].values
    hist_actual_rate = df['interest_rate'].values

    n_steps = len(df)
    print(f"[Historical] Loaded {n_steps} consecutive months of historical data.")

    trajectories = {}
    results = {}

    for policy_name, policy in policies.items():
        # Policy starts with baseline interest rate
        i_prev = config.env.r_star + config.env.pi_target

        ep = {
            "step": [],
            "inflation": [],
            "output_gap": [],
            "interest_rate": [],
            "reward": [],
            "shock_active": []
        }

        for t in range(n_steps):
            obs = np.array([hist_pi[t], hist_y[t], i_prev, hist_E_pi[t]], dtype=np.float32)
            
            # Action (Interest Rate)
            action = policy.predict(obs)
            i_t = float(np.clip(action[0], config.env.i_min, config.env.i_max))

            # Compute Reward Open-Loop Style
            reward = compute_reward(
                pi_t=hist_pi[t],
                y_t=hist_y[t],
                i_t=i_t,
                i_prev=i_prev,
                pi_target=config.env.pi_target,
                cfg=config.reward,
                in_shock_window=False
            )

            ep["step"].append(t)
            ep["inflation"].append(hist_pi[t])
            ep["output_gap"].append(hist_y[t])
            ep["interest_rate"].append(i_t)
            ep["reward"].append(reward)
            ep["shock_active"].append(False)

            i_prev = i_t

        trajectories[policy_name] = ep

        # Compute Metrics
        metrics = compute_metrics(
            trajectories=[ep],
            pi_target=config.env.pi_target,
            w_y=config.reward.w_y,
            w_i=config.reward.w_i
        )
            
        results[policy_name] = metrics

    # Plot Line Comparison (already here)
    plot_dir = os.path.join(os.path.dirname(__file__), "us_results", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, hist_actual_rate * 100, label="Historical Proxy Rate", linestyle="--", color="grey", alpha=0.7)

    colors = {"taylor_rule": "blue", "fixed_target": "black", "ppo": "red"}
    
    for pname, ep in trajectories.items():
        plt.plot(dates, np.array(ep["interest_rate"]) * 100, label=pname, color=colors.get(pname, "green"))
    
    plt.title("Historical Evaluation: Recommended Interest Rates Over Time")
    plt.ylabel("Interest Rate (%)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(plot_dir, "historical_model_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\n[Plots] Saved historical rate comparison plot -> {plot_path}")

    # Output exact requested metrics
    best_loss = min(res["Total Macro Loss"] for res in results.values())
    
    log_dir = os.path.join(os.path.dirname(__file__), "us_results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    import json
    import pandas as pd
    
    for policy_name, m in results.items():
        m["Policy Regret"] = m["Total Macro Loss"] - best_loss
        
        print(f"\n[Outputs] {policy_name} | historical_us")
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

        # Save JSON
        metric_file = os.path.join(log_dir, f"{policy_name}_historical_metrics.json")
        with open(metric_file, "w") as f:
            json.dump({
                "Total Macro Loss": m['Total Macro Loss'],
                "Inflation Loss": m['Inflation Loss'],
                "Unemployment Loss": m['Unemployment Loss'],
                "Interest Rate Smoothness (Policy Stability)": m.get('Interest Rate Smoothness', 0.0),
                "Mean Return (RL Performance Score)": m.get('Mean Return', 0.0),
                "Variance of Returns (Stability of Policy)": m.get('Variance of Returns', 0.0),
                "Inflation Variance": m['Inflation Variance'],
                "Output Gap Variance": m['Output Gap Variance'],
                "Recovery Time After Shock": m['Recovery Time After Shock'],
                "Policy Reactivity": m['Policy Reactivity'],
                "Policy Regret": m['Policy Regret']
            }, f, indent=4)
            
        # Save CSV Logs
        ep = trajectories[policy_name]
        df_log = pd.DataFrame(ep)
        csv_file = os.path.join(log_dir, f"{policy_name}_historical_logs.csv")
        df_log.to_csv(csv_file, index=False)
        print(f"  -> Saved logs to {csv_file}")

    # --- Plot the grouped bar chart for all the metrics ---
    metrics_list = [
        "Total Macro Loss", "Inflation Loss", "Unemployment Loss",
        "Interest Rate Smoothness", "Mean Return", "Policy Regret", "Policy Reactivity"
    ]
    
    n_groups = len(metrics_list)
    fig, ax = plt.subplots(figsize=(14, 7))
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8
    
    # Extract data just for taylor_rule and ppo to keep it clean
    taylor_vals = [results['taylor_rule'].get(m, results['taylor_rule'].get(m + ' (Policy Stability)', 0)) if m in results['taylor_rule'] else results['taylor_rule'].get(m + ' (RL Performance Score)', results['taylor_rule'].get(m, 0.0)) for m in metrics_list]
    ppo_vals = [results['ppo'].get(m, results['ppo'].get(m + ' (Policy Stability)', 0)) if m in results['ppo'] else results['ppo'].get(m + ' (RL Performance Score)', results['ppo'].get(m, 0.0)) for m in metrics_list]
    
    rects1 = ax.bar(index, taylor_vals, bar_width, alpha=opacity, color=colors['taylor_rule'], label='Taylor Rule')
    rects2 = ax.bar(index + bar_width, ppo_vals, bar_width, alpha=opacity, color=colors['ppo'], label='PPO Agent')
    
    ax.set_xlabel('Output Metrics')
    ax.set_ylabel('Scores / Loss')
    ax.set_title('PPO vs Taylor Rule Parameter Comparison (Historical US Dataset)')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(metrics_list, rotation=15, ha="right")
    ax.legend()
    plt.tight_layout()
    
    bar_plot_path = os.path.join(plot_dir, "historical_metrics_barchart.png")
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    
    print(f"[Plots] Saved historical metrics bar chart -> {bar_plot_path}")

if __name__ == "__main__":
    main()

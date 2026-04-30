import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3.common.results_plotter import load_results, ts2xy
from env.macro_env import MacroEnv
from policies.rl_agent import RLAgentPolicy
from config.config import DEFAULT_CONFIG
from models.taylor_rule import TaylorRule, AggressiveTaylorRule, FixedInflationTargeting

def generate_all_plots(results_dir='results', log_dir='outputs', plot_dir='plots'):
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load Data
    metrics_path = os.path.join(results_dir, 'metrics_summary.csv')
    traj_path = os.path.join(results_dir, 'episode_trajectories.pkl')
    
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
    else:
        df_metrics = None
        
    if os.path.exists(traj_path):
        with open(traj_path, 'rb') as f:
            trajectories = pickle.load(f)
    else:
        trajectories = None

    # Plot 1 - Training Curve
    try:
        plot_training_curve(log_dir, plot_dir)
    except Exception as e:
        print(f"Skipping Plot 1 (Training log not found/error): {e}")

    # Plot 2 - No Shock Comparison
    try:
        plot_no_shock_comparison(plot_dir, log_dir)
    except Exception as e:
        print(f"Skipping Plot 2 (Error): {e}")

    if trajectories and df_metrics is not None:
        # Plot 3 - Shock Response
        plot_shock_response(trajectories, plot_dir)
        # Plot 4 - Metric Bar Charts
        plot_metric_bar_charts(df_metrics, plot_dir)
        # Plot 5 - Regret Heatmap
        plot_regret_heatmap(df_metrics, plot_dir)
        # Plot 6 - Phase Plane
        plot_phase_plane(trajectories, plot_dir)
        # Plot 7 - Reward Boxplot
        plot_reward_boxplot(trajectories, plot_dir)
        print("All plots generated successfully.")
    else:
        print("Evaluation data missing, please run evaluate before plotting.")


def plot_training_curve(log_dir, plot_dir):
    try:
        # SB3 Monitor saves as monitor.csv
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            plt.figure(figsize=(10, 6))
            df = pd.DataFrame({'timesteps': x, 'reward': y})
            df['rolling'] = df['reward'].rolling(window=10).mean()
            
            plt.plot(df['timesteps'], df['reward'], alpha=0.3, label='Episode Reward')
            plt.plot(df['timesteps'], df['rolling'], color='red', label='Moving Avg (10 eps)')
            
            plt.title("PPO Training Curve")
            plt.xlabel("Timesteps")
            plt.ylabel("Episode Reward")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, 'training_curve.png'), dpi=200, bbox_inches='tight')
            plt.close()
    except Exception as e:
        raise e

def plot_no_shock_comparison(plot_dir, log_dir):
    env = MacroEnv()
    policies = {
        'Taylor Rule': TaylorRule(),
        'Taylor Aggressive': AggressiveTaylorRule(),
        'Fixed IT': FixedInflationTargeting()
    }
    try:
        ppo = RLAgentPolicy(DEFAULT_CONFIG, model_path='outputs/models/best_model')
        if ppo.model is not None:
            policies['PPO'] = ppo.model
    except Exception as e:
        print(f"Skipping PPO in plot_no_shock_comparison: {e}")
        
    env.set_shock_scenario(None)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    colors = sns.color_palette("husl", len(policies))
    
    for (name, policy), color in zip(policies.items(), colors):
        obs, _ = env.reset(seed=101)
        pi, y_g, i_r = [], [], []
        done = False
        while not done:
            pi.append(obs[0])
            y_g.append(obs[1])
            i_r.append(obs[2])
            action, _ = policy.predict(obs, deterministic=True) if name == "PPO" else policy.predict(obs)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            
        axes[0].plot(pi, label=name, color=color, alpha=0.8)
        axes[1].plot(y_g, label=name, color=color, alpha=0.8)
        axes[2].plot(i_r, label=name, color=color, alpha=0.8)

    axes[0].axhline(y=0.04, color='k', linestyle='--', alpha=0.5, label=r'Target $\pi^*$')
    axes[1].axhline(y=0.0, color='k', linestyle='--', alpha=0.5, label='Target $y^*$')
    
    axes[0].set_ylabel('Inflation Rate')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Policy Comparison (No Shocks) - Single Episode Trajectory')
    
    axes[1].set_ylabel('Output Gap')
    axes[2].set_ylabel('Interest Rate')
    axes[2].set_xlabel('Timesteps (Quarters)')
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'no_shock_comparison.png'), dpi=200)
    plt.close()

def plot_shock_response(trajectories, plot_dir):
    scenarios = list(trajectories.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        for policy in trajectories[scenario].keys():
            all_inf = trajectories[scenario][policy]['inflation']
            max_len = max(len(ep) for ep in all_inf)
            padded = np.array([ep + [np.nan]*(max_len - len(ep)) for ep in all_inf])
            mean_inf = np.nanmean(padded, axis=0)
            ax.plot(mean_inf, label=policy)
            
        ax.axhline(y=0.04, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=50, color='r', linestyle=':', alpha=0.5, label='Shock Onset')
        ax.set_title(f"Shock Response: {scenario.replace('_', ' ').title()}")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Inflation")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'shock_response.png'), dpi=200)
    plt.close()

def plot_metric_bar_charts(df, plot_dir):
    # Mean across scenarios
    df_mean = df.groupby('Policy').mean(numeric_only=True).reset_index()
    metrics = ['Inf.Var', 'Out.Var', 'Cum.Loss', 'Rec.Time']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        sns.barplot(data=df_mean, x='Policy', y=metric, ax=axes[i], palette='viridis')
        axes[i].set_title(f"Average {metric} Across All Scenarios")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'metric_comparison.png'), dpi=200)
    plt.close()

def plot_regret_heatmap(df, plot_dir):
    # Filter out PPO itself since Regret against itself is 0, or keep it to show 0
    pivot = df.pivot(index='Scenario', columns='Policy', values='Regret')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm_r", center=0)
    plt.title("Policy Regret of PPO vs Baselines\n(Positive = PPO is better)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'regret_heatmap.png'), dpi=200)
    plt.close()

def plot_phase_plane(trajectories, plot_dir):
    # Aggregate data across all scenarios
    policy_data = {}
    for sc in trajectories.keys():
        for pol, data in trajectories[sc].items():
            if pol not in policy_data:
                policy_data[pol] = {'pi': [], 'i': [], 'y': []}
            # Take first episode from each scenario just to avoid too many points
            policy_data[pol]['pi'].extend(data['inflation'][0])
            policy_data[pol]['i'].extend(data['interest'][0])
            policy_data[pol]['y'].extend(data['output'][0])
            
    policies = list(policy_data.keys())
    n = len(policies)
    
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5), sharex=True, sharey=True)
    if n == 1: axes = [axes]
    
    for ax, pol in zip(axes, policies):
        sc = ax.scatter(policy_data[pol]['pi'], policy_data[pol]['i'], 
                        c=policy_data[pol]['y'], cmap='RdYlGn', alpha=0.6, s=15)
        ax.set_title(pol)
        ax.set_xlabel('Inflation Rate')
        if ax == axes[0]:
            ax.set_ylabel('Interest Rate')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.04, color='k', linestyle='--', alpha=0.3)
        
    cbar = fig.colorbar(sc, ax=axes, label='Output Gap')
    plt.suptitle("Interest Rate Phase Plane (Colored by Output Gap)", y=1.05)
    plt.savefig(os.path.join(plot_dir, 'phase_plane.png'), dpi=200, bbox_inches='tight')
    plt.close()

def plot_reward_boxplot(trajectories, plot_dir):
    records = []
    for sc, pol_dict in trajectories.items():
        for pol, data in pol_dict.items():
            for r in data['rewards']:
                records.append({'Scenario': sc, 'Policy': pol, 'Episode Reward': r})
    df = pd.DataFrame(records)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Scenario', y='Episode Reward', hue='Policy', palette='Set2')
    plt.title("Episode Reward Distribution per Scenario")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'reward_boxplot.png'), dpi=200)
    plt.close()

import argparse
import os
import pandas as pd

from models.ppo_agent import train_ppo_agent
from env.macro_env import MacroEnv
from experiments.run_all import run_evaluation_pipeline
from plots.visualize import generate_all_plots

def train():
    print("\n--- Phase 1: Training PPO Agent ---")
    env_func = lambda: MacroEnv()
    train_ppo_agent(env_func, total_timesteps=500_000, save_dir='outputs')

def evaluate():
    print("\n--- Phase 2: Running Evaluations ---")
    df = run_evaluation_pipeline(model_path='outputs/ppo_monetary_policy.zip')
    return df

def plot():
    print("\n--- Phase 3: Generating Plots ---")
    generate_all_plots()

def print_summary_table():
    summary_path = os.path.join('results', 'metrics_summary.csv')
    if not os.path.exists(summary_path):
        print("Summary metrics not found. Run evaluate first.")
        return
        
    df = pd.read_csv(summary_path)
    
    # Group by Policy to get averages across scenarios
    df_mean = df.groupby('Policy').agg({
        'Inf.Var': 'mean',
        'Out.Var': 'mean',
        'Cum.Loss': 'mean',
        'Rec.Time': 'mean'
    }).reset_index()
    
    # Enforce order if possible: PPO, Taylor Rule, Taylor Aggressive, Fixed IT
    order = ['PPO', 'Taylor Rule', 'Taylor Aggressive', 'Fixed IT']
    df_mean['sort_key'] = df_mean['Policy'].map(lambda x: order.index(x) if x in order else 99)
    df_mean = df_mean.sort_values('sort_key').drop(columns='sort_key')
    
    print("\nFinal Averaged Results Across All Scenarios:")
    print("┌─────────────────────┬──────────┬──────────┬───────────┬───────────┐")
    print("│ Policy              │ Inf.Var  │ Out.Var  │ Cum.Loss  │ Rec.Time  │")
    print("├─────────────────────┼──────────┼──────────┼───────────┼───────────┤")
    for _, row in df_mean.iterrows():
        policy = row['Policy']
        inf_v = row['Inf.Var']
        out_v = row['Out.Var']
        cum_l = row['Cum.Loss']
        rec_t = row['Rec.Time']
        
        # Format string nicely
        print(f"│ {policy:<19} │ {inf_v:8.4f} │ {out_v:8.4f} │ {cum_l:9.2f} │ {rec_t:9.2f} │")
    print("└─────────────────────┴──────────┴──────────┴───────────┴───────────┘")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive RL-Based Monetary Policy Simulation")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'plot', 'all'], required=True, 
                        help="Mode of operation: train, evaluate, plot, or all")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        evaluate()
        print_summary_table()
    elif args.mode == 'plot':
        plot()
    elif args.mode == 'all':
        train()
        evaluate()
        plot()
        print_summary_table()

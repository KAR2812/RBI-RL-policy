import os
import pickle
import numpy as np
import pandas as pd

def compute_metrics(scenario_data, pi_star=0.04, y_star=0.0, shock_start=50):
    """
    Computes all aggregated metrics from simulation trajectories.
    Returns a DataFrame.
    """
    records = []
    
    # First pass to compute core metrics for PPO (needed for regret)
    ppo_losses = {}
    
    for scenario, policies in scenario_data.items():
        for policy_name, data in policies.items():
            
            all_inf = data['inflation']
            all_out = data['output']
            all_rew = data['rewards']
            
            # 1. & 2. Variances over time, averaged across episodes
            inf_var = np.mean([np.var(ep) for ep in all_inf]) 
            out_var = np.mean([np.var(ep) for ep in all_out])
            
            # 3. Cumulative Macro Loss: sum(|pi - pi*|^2 + |y - y*|^2) over episode
            ep_losses = [np.sum((np.array(ep_inf) - pi_star)**2 + (np.array(ep_out) - y_star)**2) for ep_inf, ep_out in zip(all_inf, all_out)]
            mean_cum_loss = np.mean(ep_losses)
            
            if policy_name == "PPO":
                ppo_losses[scenario] = mean_cum_loss
                
            # 4. Mean Episode Reward
            mean_reward = np.mean(all_rew)
            
            # 5. Recovery Time
            recovery_times = []
            for ep_inf, ep_out in zip(all_inf, all_out):
                recovered = False
                # Check from shock_start onwards
                for step in range(shock_start, len(ep_inf)):
                    if abs(ep_inf[step] - pi_star) < 0.01 and abs(ep_out[step] - y_star) < 0.01:
                        recovery_times.append(step - shock_start)
                        recovered = True
                        break
                if not recovered:
                    recovery_times.append(len(ep_inf) - shock_start) # Max possible if never recovered
                    
            mean_recovery_time = np.mean(recovery_times)
            
            records.append({
                'Scenario': scenario,
                'Policy': policy_name,
                'Inf.Var': inf_var,
                'Out.Var': out_var,
                'Cum.Loss': mean_cum_loss,
                'Reward': mean_reward,
                'Rec.Time': mean_recovery_time,
                'Regret': 0.0 # Placeholder, computed next
            })

    df = pd.DataFrame(records)
    
    # Second pass: Compute Policy Regret = CumLoss(Baseline) - CumLoss(PPO)
    for index, row in df.iterrows():
        scenario = row['Scenario']
        policy = row['Policy']
        cum_loss = row['Cum.Loss']
        
        ppo_loss = ppo_losses.get(scenario, cum_loss)
        if policy == "PPO":
            regret = 0.0
        else:
            regret = cum_loss - ppo_loss
            
        df.at[index, 'Regret'] = regret
        
    return df

def save_results(df, raw_data, results_dir='results'):
    """
    Saves the computed metrics and raw trajectories to disk.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, 'metrics_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    
    pkl_path = os.path.join(results_dir, 'episode_trajectories.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(raw_data, f)
    print(f"Trajectories saved to {pkl_path}")

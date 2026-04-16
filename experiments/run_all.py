import os
from env.macro_env import MacroEnv
from models.ppo_agent import load_ppo_agent
from models.taylor_rule import TaylorRule, AggressiveTaylorRule, FixedInflationTargeting
from utils.shock_generator import generate_all_scenario_data
from experiments.evaluate import compute_metrics, save_results

def run_evaluation_pipeline(model_path='outputs/ppo_monetary_policy.zip'):
    """
    Orchestrates the evaluation phase:
    1. Loads the environment and all agents.
    2. Runs configured shock scenarios.
    3. Computes metrics and saves results.
    """
    
    # 1. Setup Environment
    env = MacroEnv()
    
    # 2. Setup Baselines
    policies = {
        'Taylor Rule': TaylorRule(r_star=0.02, alpha=0.5, beta=0.5, pi_star=0.04),
        'Taylor Aggressive': AggressiveTaylorRule(r_star=0.02, alpha=1.5, beta=0.5, pi_star=0.04),
        'Fixed IT': FixedInflationTargeting(target_rate=0.06),
    }
    
    # 3. Setup PPO Agent
    try:
        # PPO requires DummyVecEnv context for prediction if bounds handling complex, 
        # but SB3 .predict() works on standard unwrapped obs if shape is right.
        print(f"Loading PPO model from {model_path}...")
        ppo_agent = load_ppo_agent(model_path)
        policies['PPO'] = ppo_agent
    except Exception as e:
        print(f"WARNING: Could not load PPO model correctly. Did you train it? Error: {e}")
        
    # 4. Scenarios
    scenarios = ['demand', 'supply', 'persistent_inflation', 'stagflation']
    
    # 5. Generate Data
    print("Generating scenario data. This may take a moment...")
    raw_data = generate_all_scenario_data(env, policies, scenarios)
    
    # 6. Compute Metrics
    print("Computing metrics...")
    df = compute_metrics(raw_data)
    
    # 7. Save
    save_results(df, raw_data)
    
    return df

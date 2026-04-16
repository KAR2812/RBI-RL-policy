import numpy as np
from tqdm import tqdm

def run_evaluation_episodes(env, policy_agent, policy_name, shock_type, n_episodes=50):
    """
    Run evaluation episodes for a given policy and shock scenario.
    Returns trajectories required for evaluation metrics.
    """
    print(f"Evaluating {policy_name} under '{shock_type}' shock...")
    
    all_episode_rewards = []
    all_inflation_trajectories = []
    all_output_trajectories = []
    all_interest_trajectories = []
    
    env.set_shock_scenario(shock_type, shock_start=50)

    for ep in tqdm(range(n_episodes), desc=f"{policy_name} ({shock_type})", leave=False):
        obs, _ = env.reset(seed=ep+42) # fixed seed per episode for comparability
        
        ep_rewards = 0
        ep_inflation = []
        ep_output = []
        ep_interest = []
        
        done = False
        while not done:
            ep_inflation.append(obs[0])
            ep_output.append(obs[1])
            
            # Action selection
            if policy_name == "PPO":
                action, _states = policy_agent.predict(obs, deterministic=True)
            else:
                action, _states = policy_agent.predict(obs) # Baselines
                
            obs, reward, terminated, truncated, _ = env.step(action)
            
            ep_interest.append(obs[2]) # using actual applied interest rate from obs
            ep_rewards += reward
            
            done = terminated or truncated
            
        all_episode_rewards.append(ep_rewards)
        all_inflation_trajectories.append(ep_inflation)
        all_output_trajectories.append(ep_output)
        all_interest_trajectories.append(ep_interest)
        
    return {
        'rewards': all_episode_rewards,
        'inflation': all_inflation_trajectories,
        'output': all_output_trajectories,
        'interest': all_interest_trajectories
    }

def generate_all_scenario_data(env, policies_dict, scenarios):
    """
    Generate evaluation data across all combinations of scenarios and policies.
    policies_dict: {'PPO': model, 'Taylor': model, ...}
    scenarios: list of shock types
    Returns nested dictionary: results[scenario][policy_name] = data
    """
    results = {}
    for scenario in scenarios:
        results[scenario] = {}
        for policy_name, agent in policies_dict.items():
            data = run_evaluation_episodes(env, agent, policy_name, scenario, n_episodes=50)
            results[scenario][policy_name] = data
            
    return results

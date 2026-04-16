import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

def get_ppo_model(env):
    """
    Instantiate a PPO agent with the specified configuration.
    Uses Stable-Baselines3 >= 2.2.0 specific syntax.
    """
    
    # In newer SB3 versions (>= 1.7), net_arch is a dict for shared/separated layers.
    # The user specification [dict(pi=..., vf=...)] was deprecated, 
    # so we use dict(pi=[128, 128], vf=[128, 128]) directly as policy_kwargs.
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )
    
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=42
    )
    return model

def train_ppo_agent(env_func, total_timesteps=500_000, save_dir='outputs'):
    """
    Train the PPO agent on the macroeconomic environment.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the environment inside Monitor and DummyVecEnv, saving logs to save_dir
    log_file = os.path.join(save_dir, 'training_log')
    env = DummyVecEnv([lambda: Monitor(env_func(), filename=log_file)])
    
    model = get_ppo_model(env)
    
    # Configure checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // env.num_envs, # adjust for num_envs if > 1
        save_path=save_dir,
        name_prefix='ppo_checkpoint'
    )
    
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # Save the final model
    final_model_path = os.path.join(save_dir, 'ppo_monetary_policy')
    model.save(final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}.zip")
    
    return model

def load_ppo_agent(model_path, env_func=None):
    """
    Load a trained PPO agent from disk.
    If env_func is provided, it recreates DummyVecEnv for proper testing bindings.
    """
    if env_func is not None:
        env = DummyVecEnv([lambda: Monitor(env_func())])
        return PPO.load(model_path, env=env)
    return PPO.load(model_path)

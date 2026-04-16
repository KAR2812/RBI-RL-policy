"""
Standalone training entry point.

Usage:
    python train.py
    python train.py --timesteps 300000 --seed 0
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import Config, DEFAULT_CONFIG
from rl.train import train_ppo_agent
from utils.seed import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Train PPO monetary policy agent")
    parser.add_argument("--timesteps", type=int,   default=200_000,        help="Training timesteps")
    parser.add_argument("--seed",      type=int,   default=42,             help="Global random seed")
    parser.add_argument("--save_dir",  type=str,   default="outputs/models", help="Model save directory")
    parser.add_argument("--log_dir",   type=str,   default="outputs/logs",   help="Log directory")
    args = parser.parse_args()

    set_global_seed(args.seed)
    config = DEFAULT_CONFIG
    config.paths.ensure_dirs()

    print("=" * 60)
    print("  PPO Monetary Policy Agent — Training")
    print("=" * 60)
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Seed      : {args.seed}")
    print(f"  Save dir  : {args.save_dir}")
    print("=" * 60)

    train_ppo_agent(
        config=config,
        total_timesteps=args.timesteps,
        seed=args.seed,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )
    print("\n[Train] Done. Model saved to", args.save_dir)


if __name__ == "__main__":
    main()

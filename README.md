# Adaptive Reinforcement Learning–Based Monetary Policy in a Simulated Macroeconomy

> **A research-grade implementation comparing a PPO-based RL central bank agent against classical Taylor Rule and Fixed Inflation Targeting baselines in a calibrated New Keynesian macroeconomic simulation.**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Usage Guide](#usage-guide)
5. [Macroeconomic Model](#macroeconomic-model)
6. [Shock Scenarios](#shock-scenarios)
7. [Reward Design](#reward-design)
8. [Policies Compared](#policies-compared)
9. [Configuration](#configuration)
10. [Calibration Notes](#calibration-notes)
11. [Outputs](#outputs)
12. [Assumptions](#assumptions)

---

## Project Overview

This project frames monetary policy as a sequential decision-making problem. A simulated central bank observes the current macroeconomic state (inflation, output gap, inflation expectations) and sets a nominal interest rate to minimise a quadratic loss on inflation deviation, output gap, and rate volatility.

We compare three policies:
| Policy | Type | Description |
|---|---|---|
| **Taylor Rule** | Classical | `i = r* + π + α(π − π*) + β_y·y` |
| **Fixed Inflation Target** | Rules-based | Always sets `i = r* + π*` |
| **PPO Agent** | RL | Learns optimal rate via trial-and-error |

---

## Architecture

```
btp-2/
├── config/
│   └── config.py          # ALL hyperparameters (single source of truth)
├── env/
│   ├── dynamics.py        # IS curve + Phillips curve + expectations
│   ├── shocks.py          # AR(1) shock generator (configurable)
│   ├── reward.py          # Quadratic reward function
│   └── macro_env.py       # Gymnasium environment
├── policies/
│   ├── taylor_rule.py     # Taylor Rule baseline
│   ├── fixed_target.py    # Fixed inflation targeting baseline
│   └── rl_agent.py        # PPO agent (Stable-Baselines3)
├── rl/
│   ├── train.py           # Training logic
│   └── evaluate.py        # Evaluation runner
├── data/
│   ├── data_loader.py     # RBI DBIE Excel parser
│   └── calibration.py     # Parameter extraction from RBI data
├── utils/
│   ├── seed.py            # Global seeding
│   ├── logger.py          # CSV experiment logger
│   ├── metrics.py         # Evaluation metric computations
│   └── plotting.py        # All visualisation functions
├── experiments/
│   ├── shock_scenarios.py # Pre-defined scenario configs
│   └── run_experiment.py  # Multi-policy multi-scenario runner
├── main.py                # Unified CLI
├── train.py               # Standalone training script
├── evaluate.py            # Standalone evaluation script
├── requirements.txt
└── outputs/               # Auto-created
    ├── models/            # Saved PPO checkpoints
    ├── logs/              # Per-step CSV logs + JSON summaries
    └── plots/             # All generated figures
```

---

## Quick Start

### 1. Install dependencies
```bash
cd /Users/keshavarunanumolu/btp-2
pip install -r requirements.txt
```

### 2. Run Taylor Rule baseline (no training needed)
```bash
python main.py --mode evaluate --policy taylor --scenario normal
```

### 3. Train the PPO agent
```bash
python main.py --mode train --timesteps 200000 --seed 42
```

### 4. Compare all policies across all scenarios
```bash
python main.py --mode compare
```

### 5. Historical RBI data validation
```bash
python main.py --mode rbi
```

---

## Usage Guide

### Training
```bash
python train.py --timesteps 200000 --seed 42
# or
python main.py --mode train --timesteps 200000 --seed 42
```

### Evaluating a single policy
```bash
python evaluate.py --policy taylor --scenario demand_shock
python evaluate.py --policy ppo    --scenario supply_shock
python evaluate.py --policy all    --scenario all
# or using the unified CLI:
python main.py --mode evaluate --policy ppo --scenario stagflation
```

### Full comparison run
```bash
python main.py --mode compare --seed 42
```
Generates all plots and `outputs/logs/experiment_summary.json`.

---

## Macroeconomic Model

We implement a simplified **New Keynesian (NK)** model:

### IS Curve (demand side)
```
y_t = -a * (i_t - E_t[π_{t+1}]) + ε^d_t
```
- `y_t` = output gap (% deviation from potential)
- `i_t` = nominal interest rate (policy instrument)
- `E_t[π]` = expected inflation
- `ε^d_t` = AR(1) demand shock

### Phillips Curve (supply side)
```
π_t = β * E_t[π_{t+1}] + κ * y_t + ε^s_t
```
- `π_t` = inflation rate
- `β ≈ 0.99` = intertemporal discount (near-unit in NK)
- `κ = 0.15` = slope (output gap to inflation sensitivity)
- `ε^s_t` = AR(1) supply shock

### Adaptive Expectations
```
E_t[π_{t+1}] = ρ * π_t + (1 − ρ) * E_{t-1}[π]
```
- `ρ = 0.75` = persistence of expectations

### Shock Process
```
ε_t = ρ_ε * ε_{t-1} + σ * z_t,   z_t ~ N(0, 1)
```

---

## Shock Scenarios

| Scenario | Demand Burst | Supply Burst | Window (steps) |
|---|---|---|---|
| `normal` | 0.0 | 0.0 | None |
| `demand_shock` | +0.04 | 0.0 | 50–80 |
| `supply_shock` | 0.0 | +0.05 | 50–80 |
| `stagflation` | −0.03 | +0.05 | 40–90 |
| `persistent_inflation` | +0.01 | +0.03 | 30–150 |
| `mixed` | +0.03 | +0.04 | 40–100 |

---

## Reward Design

```
r_t = -(w_π · (π_t − π*)² + w_y · y_t² + w_i · (i_t − i_{t-1})²)
```
Clipped to `[−20, 0]` to prevent gradient explosions.

| Weight | Value | Purpose |
|---|---|---|
| `w_π` | 2.0 | Stabilise inflation |
| `w_y` | 1.0 | Minimise output gap |
| `w_i` | 0.5 | Penalise large rate jumps |

---

## Policies Compared

### Taylor Rule
```
i_t = r* + π_t + α·(π_t − π*) + β_y·y_t
```
- `r* = 0.02`, `π* = 0.04`, `α = 1.5`, `β_y = 0.5`
- Taylor principle satisfied: `α > 1` ensures real rate rises with inflation

### Fixed Inflation Targeting
```
i_t = r* + π* = 0.06  (constant)
```
Simple lower-bound baseline.

### PPO Agent
- **Algorithm**: Proximal Policy Optimisation (Stable-Baselines3)
- **Architecture**: MLP [128, 128], Actor-Critic
- **Training**: 4 parallel environments, 200,000 timesteps
- **Callbacks**: EvalCallback + early stopping (no improvement after 10 evals)
- **Action**: Continuous rate in `[0, 0.15]`
- **Observation**: `[π, y, i_{t-1}, E[π]]` (4-dim)

---

## Configuration

All parameters are in `config/config.py`. Key sections:

```python
EnvConfig(a=0.5, beta=0.99, kappa=0.15, rho_exp=0.75,
          pi_target=0.04, r_star=0.02,
          i_min=0.0, i_max=0.15, max_steps=200)

ShockConfig(rho_d=0.70, rho_s=0.60, sigma_d=0.008, sigma_s=0.010)

RewardConfig(w_pi=2.0, w_y=1.0, w_i=0.5, clip_low=-20.0)

TaylorConfig(alpha=1.5, beta_y=0.5)

PPOConfig(total_timesteps=200_000, n_steps=2048, batch_size=64, ...)
```

---

## Calibration Notes

Parameters are calibrated using the following sources:

| Parameter | Source | Value |
|---|---|---|
| `π*` (target) | RBI Monetary Policy Framework | 4% |
| `r*` (neutral rate) | RBI Estimates 2020–24 | ~2% |
| `i_max` | RBI Repo Rate historical upper | 15% |
| `κ` (Phillips slope) | IMF India estimates | 0.10–0.20 |
| `β` | NK literature standard | 0.99 |
| `σ_s` | RBI CPI volatility (month-on-month σ) | ~0.01 |

The RBI CPI Excel file (`data/RBIB Table No. 19...xlsx`) is used to:
1. Verify that simulation inflation ranges match India's historical range (2–18%)
2. Calibrate `σ_s` from the month-on-month inflation change distribution
3. Provide a held-out test set for the `--mode rbi` historical validation

**Training happens entirely in simulation. Real data is only for calibration and validation.**

---

## Outputs

After running experiments, `outputs/` will contain:

```
outputs/
├── models/
│   ├── best_model.zip         # Best PPO model (by eval reward)
│   └── ppo_final.zip          # Final PPO model
├── logs/
│   ├── taylor_rule_normal_*.csv
│   ├── ppo_demand_shock_*.csv
│   └── experiment_summary.json
└── plots/
    ├── normal/
    │   ├── taylor_rule_normal_inflation.png
    │   ├── compare_inflation_normal.png
    │   ├── cumulative_loss_normal.png
    │   └── metrics_table_normal.png
    ├── demand_shock/
    └── ...
```

---

## Assumptions

1. **Simplified NK model**: We omit fiscal policy, labour market, and open-economy channels for tractability.
2. **Adaptive expectations**: A backward-looking blend is used instead of full rational expectations to avoid model-consistency requirements.
3. **AR(1) shocks**: Real-world shocks are modelled as AR(1) processes; actual structural breaks (COVID, oil embargoes) are not modelled but approximated by the shock scenarios.
4. **ZLB**: The zero lower bound is enforced by clipping `i_t ≥ 0`.
5. **Simulation-only training**: The PPO agent never sees real data during training — only the simulated environment. This is intentional (the simulation is the MDP).
6. **Output gap proxy**: The RBI dataset only provides CPI. Output gap and repo rate are synthesised from historical CPI using an AR(1) model and the Taylor formula respectively, as documented in `data/data_loader.py`.

---

## Reproducibility

Set `--seed` in all commands. The `utils/seed.py` sets NumPy, Python `random`, and PyTorch seeds deterministically. The Gymnasium environment also uses the passed seed for its RNG.

```bash
python main.py --mode compare --seed 42
```

---

*Project for BTP-2 | Adaptive RL Monetary Policy | March 2026*

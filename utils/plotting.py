"""
All visualisation functions for the macroeconomic RL project.

Each function takes trajectory data and saves a plot to disk.
Uses a consistent publication-quality style.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List

# ── Style ──────────────────────────────────────────────────────────────────────
STYLE = {
    "figure.figsize":  (12, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family":     "sans-serif",
    "axes.grid":       True,
    "grid.alpha":      0.35,
    "grid.linestyle":  "--",
}

POLICY_COLORS = {
    "taylor_rule":    "#1f77b4",
    "fixed_target":  "#ff7f0e",
    "ppo":           "#2ca02c",
    "historical":    "#9467bd",
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def _shade_shock(ax: plt.Axes, shock_flags: np.ndarray, steps: np.ndarray) -> None:
    """Shade the shock window on an axis."""
    in_shock = np.asarray(shock_flags, dtype=bool)
    if not np.any(in_shock):
        return
    starts = np.where(np.diff(np.concatenate([[False], in_shock, [False]])) == 1)[0]
    ends   = np.where(np.diff(np.concatenate([[False], in_shock, [False]])) == -1)[0]
    for s, e in zip(starts, ends):
        ax.axvspan(steps[s], steps[min(e, len(steps) - 1)], alpha=0.12, color="red",
                   label="_nolegend_")


def _add_target_line(ax: plt.Axes, target: float, label: str = "Target") -> None:
    ax.axhline(y=target, linestyle=":", color="grey", linewidth=1.2, label=label)


def save_single_episode_plots(
    trajectory: dict,
    policy_name: str,
    scenario_name: str,
    plot_dir: str,
    pi_target: float = 0.04,
) -> None:
    """
    Save individual trajectory plots (inflation, output gap, interest rate, reward).

    Args:
        trajectory  : dict of lists from evaluate_policy
        policy_name : string label
        scenario_name: shock scenario label
        plot_dir    : output directory
        pi_target   : inflation target (for horizontal guide line)
    """
    _apply_style()
    os.makedirs(plot_dir, exist_ok=True)

    steps        = np.array(trajectory["step"])
    inflation    = np.array(trajectory["inflation"])
    output_gap   = np.array(trajectory["output_gap"])
    interest_rate= np.array(trajectory["interest_rate"])
    reward       = np.array(trajectory["reward"])
    shock_flags  = np.array(trajectory["shock_active"], dtype=bool)

    slug = f"{policy_name}_{scenario_name}"
    color = POLICY_COLORS.get(policy_name, "#333333")

    panels = [
        ("inflation",     inflation,     "Inflation",     "Inflation Rate", pi_target, "Inflation Target"),
        ("output_gap",    output_gap,    "Output Gap",    "Output Gap",     0.0,       "Potential Output"),
        ("interest_rate", interest_rate, "Interest Rate", "Rate",           None,      None),
        ("reward",        reward,        "Per-Step Reward","Reward",        None,      None),
    ]

    for key, data, title, ylabel, target, target_label in panels:
        fig, ax = plt.subplots()
        ax.plot(steps, data, color=color, linewidth=1.5, label=policy_name)
        _shade_shock(ax, shock_flags, steps)
        if target is not None:
            _add_target_line(ax, target, target_label)
        ax.set_title(f"{title} — {policy_name.replace('_', ' ').title()} ({scenario_name})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = os.path.join(plot_dir, f"{slug}_{key}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)

    print(f"[Plots] Saved single-episode plots → {plot_dir}/{slug}_*.png")


def save_comparison_plot(
    trajectories: dict[str, dict],
    metric_key: str,
    ylabel: str,
    title: str,
    plot_dir: str,
    filename: str,
    pi_target: Optional[float] = None,
    shock_flags: Optional[np.ndarray] = None,
) -> None:
    """
    Overlay multiple policies on one plot for a given metric.

    Args:
        trajectories : { policy_name: trajectory_dict }
        metric_key   : key in trajectory dict to plot
        ylabel, title: axis labels
        plot_dir     : output directory
        filename     : output file name (without extension)
        pi_target    : if given, draw a target line
        shock_flags  : shared shock indicator array
    """
    _apply_style()
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 5))

    for name, traj in trajectories.items():
        steps = np.array(traj["step"])
        data  = np.array(traj[metric_key])
        color = POLICY_COLORS.get(name, None)
        ax.plot(steps, data, label=name.replace("_", " ").title(), color=color, linewidth=1.5)

    if shock_flags is not None and np.any(shock_flags):
        ref_steps = np.array(next(iter(trajectories.values()))["step"])
        _shade_shock(ax, shock_flags, ref_steps)
        shock_patch = mpatches.Patch(color="red", alpha=0.2, label="Shock Window")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [shock_patch], labels + ["Shock Window"], fontsize=9)
    else:
        ax.legend(fontsize=9)

    if pi_target is not None:
        _add_target_line(ax, pi_target)

    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    path = os.path.join(plot_dir, f"{filename}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plots] Saved comparison plot → {path}")


def save_cumulative_loss_comparison(
    metrics_dict: dict[str, dict],
    plot_dir: str,
    filename: str = "cumulative_loss_comparison",
) -> None:
    """Bar chart of cumulative loss per policy."""
    _apply_style()
    os.makedirs(plot_dir, exist_ok=True)

    names  = list(metrics_dict.keys())
    losses = [metrics_dict[n]["cumulative_loss"] for n in names]
    colors = [POLICY_COLORS.get(n, "#555555") for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [n.replace("_", " ").title() for n in names],
        losses,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, val in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("Cumulative Macro Loss per Policy")
    ax.set_ylabel("Cumulative Loss (↓ better)")
    fig.tight_layout()
    path = os.path.join(plot_dir, f"{filename}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plots] Saved cumulative loss bar chart → {path}")


def save_recovery_comparison(
    metrics_dict: dict[str, dict],
    plot_dir: str,
    filename: str = "recovery_time_comparison",
) -> None:
    """Bar chart of shock recovery time per policy."""
    _apply_style()
    os.makedirs(plot_dir, exist_ok=True)

    names  = [n for n in metrics_dict if "shock_recovery_steps" in metrics_dict[n]]
    if not names:
        return
    times  = [metrics_dict[n].get("shock_recovery_steps", -1) for n in names]
    colors = [POLICY_COLORS.get(n, "#555555") for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [n.replace("_", " ").title() for n in names],
        [max(t, 0) for t in times],
        color=colors,
        edgecolor="white",
    )
    for bar, val in zip(bars, times):
        label = f"{val}" if val >= 0 else "Never"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            label,
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("Shock Recovery Time per Policy (steps)")
    ax.set_ylabel("Steps to Recovery (↓ better)")
    fig.tight_layout()
    path = os.path.join(plot_dir, f"{filename}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plots] Saved recovery time chart → {path}")


def save_metrics_summary_table(
    metrics_dict: dict[str, dict],
    plot_dir: str,
    filename: str = "metrics_summary",
) -> None:
    """Render metrics as a matplotlib table image."""
    _apply_style()
    os.makedirs(plot_dir, exist_ok=True)

    row_keys = [
        "mean_inflation", "inflation_variance", "inflation_rmse",
        "mean_output_gap", "output_gap_variance",
        "mean_interest_rate", "interest_rate_volatility",
        "mean_reward", "cumulative_loss", "shock_recovery_steps",
    ]
    row_labels = [
        "Mean Inflation", "Inflation Variance", "Inflation RMSE",
        "Mean Output Gap", "Output Gap Variance",
        "Mean Interest Rate", "Rate Volatility",
        "Mean Reward", "Cumulative Loss", "Recovery Steps",
    ]
    col_labels = [n.replace("_", " ").title() for n in metrics_dict]

    cell_data = []
    for rk in row_keys:
        row = []
        for n in metrics_dict:
            v = metrics_dict[n].get(rk, "N/A")
            row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        cell_data.append(row)

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels)*3), len(row_keys)*0.5 + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax.set_title("Policy Evaluation Metrics Summary", pad=12, fontsize=11)
    fig.tight_layout()
    path = os.path.join(plot_dir, f"{filename}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Saved metrics summary table → {path}")


def save_historical_comparison(
    steps: np.ndarray,
    pi_actual: np.ndarray,
    pi_taylor: np.ndarray,
    pi_ppo: Optional[np.ndarray],
    i_taylor: np.ndarray,
    i_ppo: Optional[np.ndarray],
    plot_dir: str,
    filename: str = "rbi_historical_comparison",
) -> None:
    """Plot historical inflation + interest rate for Taylor Rule and PPO."""
    _apply_style()
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Inflation subplot
    ax = axes[0]
    ax.plot(steps, pi_actual * 100, color="black", linewidth=1.5, label="Actual CPI")
    ax.plot(steps, pi_taylor * 100, color=POLICY_COLORS["taylor_rule"], linewidth=1.2,
            linestyle="--", label="Taylor Rule (implied)")
    if pi_ppo is not None:
        ax.plot(steps, pi_ppo * 100, color=POLICY_COLORS["ppo"], linewidth=1.2,
                linestyle="-.", label="PPO Agent (implied)")
    ax.axhline(y=4.0, linestyle=":", color="grey", label="4% Target")
    ax.set_title("Inflation: Historical vs Policy Implied")
    ax.set_xlabel("Period")
    ax.set_ylabel("Inflation (%)")
    ax.legend(fontsize=8)

    # Interest rate subplot
    ax = axes[1]
    ax.plot(steps, i_taylor * 100, color=POLICY_COLORS["taylor_rule"], linewidth=1.2,
            linestyle="--", label="Taylor Rule Rate")
    if i_ppo is not None:
        ax.plot(steps, i_ppo * 100, color=POLICY_COLORS["ppo"], linewidth=1.2,
                linestyle="-.", label="PPO Rate")
    ax.set_title("Policy Rate: Taylor Rule vs PPO")
    ax.set_xlabel("Period")
    ax.set_ylabel("Interest Rate (%)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(plot_dir, f"{filename}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plots] Saved historical comparison → {path}")

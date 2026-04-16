"""
Predefined shock scenario configurations.

Each scenario returns a ShockScenarioParams object that can be
directly passed to MacroEconomicEnv or ShockGenerator.
"""
from __future__ import annotations

from env.shocks import ShockScenarioParams
from typing import Optional


def get_shock_scenario(
    name: str,
    seed: Optional[int] = 42,
    max_steps: int = 200,
) -> ShockScenarioParams:
    """
    Return a ShockScenarioParams for a named scenario.

    Available scenarios:
        normal              - mild AR(1) shocks only
        demand_shock        - burst demand boom (t=50–80)
        supply_shock        - burst cost-push inflation (t=50–80)
        stagflation         - simultaneous supply spike + demand drop (t=40–90)
        persistent_inflation- sustained mild supply pressure (t=30–150)
        mixed               - demand boom followed by supply spike

    Args:
        name     : scenario identifier (case-insensitive)
        seed     : RNG seed
        max_steps: episode length (used to set shock_end guard)

    Returns:
        ShockScenarioParams
    """
    name = name.lower().strip()

    if name == "normal":
        return ShockScenarioParams(
            name="normal",
            demand_burst=0.0,
            supply_burst=0.0,
            shock_start=max_steps + 1,  # never
            shock_end=max_steps + 2,
            seed=seed,
        )

    elif name == "demand_shock":
        return ShockScenarioParams(
            name="demand_shock",
            demand_burst=0.02,    # positive demand impulse (+2% of GDP)
            supply_burst=0.0,
            shock_start=50,
            shock_end=80,
            seed=seed,
        )

    elif name == "supply_shock":
        return ShockScenarioParams(
            name="supply_shock",
            demand_burst=0.0,
            supply_burst=0.025,   # positive supply shock → cost-push inflation
            shock_start=50,
            shock_end=80,
            seed=seed,
        )

    elif name == "stagflation":
        return ShockScenarioParams(
            name="stagflation",
            demand_burst=-0.015,  # demand contraction
            supply_burst=0.025,   # supply-side inflation
            shock_start=40,
            shock_end=90,
            seed=seed,
        )

    elif name == "persistent_inflation":
        return ShockScenarioParams(
            name="persistent_inflation",
            demand_burst=0.008,
            supply_burst=0.018,   # sustained mild cost-push
            shock_start=30,
            shock_end=150,
            seed=seed,
        )

    elif name == "mixed":
        # Demand boom followed by supply spike
        return ShockScenarioParams(
            name="mixed",
            demand_burst=0.015,   # demand boom
            supply_burst=0.020,   # supply spike overlapping
            shock_start=40,
            shock_end=100,
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unknown shock scenario: '{name}'. "
            "Choose from: normal, demand_shock, supply_shock, "
            "stagflation, persistent_inflation, mixed."
        )


ALL_SCENARIOS = [
    "normal",
    "demand_shock",
    "supply_shock",
    "stagflation",
    "persistent_inflation",
    "mixed",
]

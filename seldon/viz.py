"""Visualization utilities."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from seldon.analysis import percentiles


def fan_chart(
    samples: dict,
    start_age: int,
    title: str = "Net Worth Projection",
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot a fan chart showing percentile bands of net worth over time.

    Bands: 5-95 (light), 25-75 (medium), median (line).
    """
    pcts = percentiles(samples, (5, 25, 50, 75, 95))
    n_years = pcts[50].shape[0]
    ages = jnp.arange(start_age, start_age + n_years)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Fan bands
    ax.fill_between(ages, pcts[5] / 1e6, pcts[95] / 1e6, alpha=0.2, color="steelblue", label="5th-95th")
    ax.fill_between(ages, pcts[25] / 1e6, pcts[75] / 1e6, alpha=0.4, color="steelblue", label="25th-75th")
    ax.plot(ages, pcts[50] / 1e6, color="steelblue", linewidth=2, label="Median")

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Age")
    ax.set_ylabel("Net Worth ($M)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def compare_scenarios(
    scenario_results: dict[str, dict],
    start_age: int,
    title: str = "Scenario Comparison",
) -> Figure:
    """Plot median trajectories of multiple scenarios on one chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["steelblue", "coral", "seagreen", "mediumpurple", "goldenrod"]

    for i, (name, samples) in enumerate(scenario_results.items()):
        pcts = percentiles(samples, (25, 50, 75))
        n_years = pcts[50].shape[0]
        ages = jnp.arange(start_age, start_age + n_years)
        color = colors[i % len(colors)]

        ax.fill_between(ages, pcts[25] / 1e6, pcts[75] / 1e6, alpha=0.15, color=color)
        ax.plot(ages, pcts[50] / 1e6, color=color, linewidth=2, label=name)

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Age")
    ax.set_ylabel("Net Worth ($M)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig

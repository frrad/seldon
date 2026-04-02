"""Analysis utilities for simulation results."""

from __future__ import annotations

import jax.numpy as jnp


def probability_of_ruin(samples: dict, year: int | None = None) -> float:
    """Probability that net worth drops to zero.

    Args:
        samples: Output from run_forward().
        year: Check at specific year index. None = any year (ever go broke).
    """
    trajectories = samples["net_worth_trajectory"]  # (num_samples, n_years)
    if year is not None:
        broke = trajectories[:, year] <= 0
    else:
        broke = jnp.any(trajectories <= 0, axis=1)
    return float(jnp.mean(broke))


def percentiles(
    samples: dict,
    pcts: tuple[float, ...] = (5, 25, 50, 75, 95),
) -> dict[float, jnp.ndarray]:
    """Compute percentile bands over the trajectory.

    Returns dict mapping percentile -> array of shape (n_years,).
    """
    trajectories = samples["net_worth_trajectory"]
    return {p: jnp.percentile(trajectories, p, axis=0) for p in pcts}


def summary(samples: dict, start_age: int) -> str:
    """Print a text summary of simulation results."""
    trajectories = samples["net_worth_trajectory"]
    final = samples["final_net_worth"]
    n_samples, n_years = trajectories.shape

    lines = [
        f"Simulation: {n_samples} samples, {n_years} years (age {start_age} to {start_age + n_years})",
        f"",
        f"Final net worth:",
        f"  5th percentile:  ${float(jnp.percentile(final, 5)):>14,.0f}",
        f"  25th percentile: ${float(jnp.percentile(final, 25)):>14,.0f}",
        f"  Median:          ${float(jnp.percentile(final, 50)):>14,.0f}",
        f"  75th percentile: ${float(jnp.percentile(final, 75)):>14,.0f}",
        f"  95th percentile: ${float(jnp.percentile(final, 95)):>14,.0f}",
        f"",
        f"Probability of ruin (ever):     {probability_of_ruin(samples) * 100:.1f}%",
        f"Probability of ruin (final yr): {probability_of_ruin(samples, year=-1) * 100:.1f}%",
    ]
    return "\n".join(lines)

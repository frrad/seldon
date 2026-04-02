"""NumPyro generative model for financial simulation."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive

from seldon.state import (
    AccountType,
    AssetClass,
    FinancialState,
    LifeEvent,
    Scenario,
)

# --- Return distribution parameters (annualized, real) ---
# Source: historical averages, rough estimates for v1
RETURN_PARAMS: dict[AssetClass, tuple[float, float]] = {
    AssetClass.US_STOCKS: (0.07, 0.16),
    AssetClass.INTL_STOCKS: (0.06, 0.18),
    AssetClass.BONDS: (0.02, 0.06),
    AssetClass.CASH: (0.005, 0.01),
    AssetClass.REAL_ESTATE: (0.04, 0.12),
}

TAX_BRACKETS_SINGLE_2024 = [
    (11_600, 0.10),
    (47_150, 0.12),
    (100_525, 0.22),
    (191_950, 0.24),
    (243_725, 0.32),
    (609_350, 0.35),
    (float("inf"), 0.37),
]


def _effective_tax_rate(taxable_income: float) -> float:
    """Compute effective federal tax rate from brackets. Simplified for v1.

    This operates on plain Python floats (not JAX tracers) and is called
    during array pre-computation, not inside the NumPyro model.
    """
    tax = 0.0
    prev = 0.0
    for top, rate in TAX_BRACKETS_SINGLE_2024:
        bracket_income = min(taxable_income, top) - prev
        if bracket_income <= 0:
            break
        tax += bracket_income * rate
        prev = top
    return tax / max(taxable_income, 1.0)


def _build_annual_arrays(scenario: Scenario) -> dict:
    """Pre-compute year-by-year deterministic inputs from scenario."""
    state = scenario.initial_state
    n_years = state.life_expectancy - state.age

    # Build event lookup by age
    events_by_age: dict[int, LifeEvent] = {}
    for e in scenario.events:
        events_by_age[e.age] = e

    gross_income = jnp.zeros(n_years)
    base_expenses = jnp.zeros(n_years)
    tax_deferred_contrib = jnp.zeros(n_years)
    employer_match = jnp.zeros(n_years)
    debt_payments = jnp.zeros(n_years)
    tax_rates = jnp.zeros(n_years)
    retired = jnp.zeros(n_years, dtype=bool)

    # Walk through years, applying events
    current_income = list(state.income)
    current_expenses = list(state.expenses)
    current_debts = list(state.debts)
    retirement_age = state.retirement_age

    for t in range(n_years):
        age = state.age + t

        # Apply life event if any
        if age in events_by_age:
            event = events_by_age[age]
            if event.new_income is not None:
                current_income = list(event.new_income)
            if event.add_expenses is not None:
                current_expenses.extend(event.add_expenses)
            if event.remove_expenses is not None:
                names = set(event.remove_expenses)
                current_expenses = [e for e in current_expenses if e.name not in names]
            if event.add_debts is not None:
                current_debts.extend(event.add_debts)
            if event.retirement_age is not None:
                retirement_age = event.retirement_age

        is_retired = age >= retirement_age
        retired = retired.at[t].set(is_retired)

        if not is_retired:
            yr_income = sum(s.annual_gross * (1 + s.growth_rate) ** t for s in state.income)
            # Use current_income for income after events
            if current_income != list(state.income):
                years_since_change = t  # simplified
                yr_income = sum(
                    s.annual_gross * (1 + s.growth_rate) ** years_since_change
                    for s in current_income
                )
            yr_deferred = sum(s.tax_deferred_contribution for s in current_income)
            yr_match = sum(s.employer_match for s in current_income)
        else:
            yr_income = 0.0
            yr_deferred = 0.0
            yr_match = 0.0

        yr_expenses = sum(e.annual_amount for e in current_expenses)
        yr_debt_payments = sum(d.minimum_payment for d in current_debts)

        yr_taxable_income = yr_income - yr_deferred
        yr_tax_rate = _effective_tax_rate(yr_taxable_income)

        gross_income = gross_income.at[t].set(yr_income)
        base_expenses = base_expenses.at[t].set(yr_expenses)
        tax_deferred_contrib = tax_deferred_contrib.at[t].set(yr_deferred)
        employer_match = employer_match.at[t].set(yr_match)
        debt_payments = debt_payments.at[t].set(yr_debt_payments)
        tax_rates = tax_rates.at[t].set(yr_tax_rate)

    return {
        "n_years": n_years,
        "gross_income": gross_income,
        "base_expenses": base_expenses,
        "tax_deferred_contrib": tax_deferred_contrib,
        "employer_match": employer_match,
        "debt_payments": debt_payments,
        "retired": retired,
        "tax_rates": tax_rates,
        "initial_assets": state.total_assets,
        "initial_debt": state.total_debt,
    }


def financial_model(arrays: dict):
    """NumPyro generative model for a financial trajectory.

    Samples uncertain parameters and simulates year-by-year net worth.
    """
    n_years = arrays["n_years"]

    # --- Sample uncertain parameters ---
    # Per-year real investment returns
    # Mean real return ~ 5-7% with moderate uncertainty about the long-run mean
    mean_return = numpyro.sample("mean_return", dist.Normal(0.06, 0.015))
    # Annual volatility ~ 15% (typical for a stock/bond mix)
    annual_returns = numpyro.sample(
        "annual_returns",
        dist.Normal(mean_return, 0.15).expand([n_years]),
    )

    # Per-year inflation rate (centered around 3%, mild year-to-year variation)
    annual_inflation = numpyro.sample(
        "annual_inflation",
        dist.Normal(0.03, 0.015).expand([n_years]),
    )

    # --- Simulate year by year ---
    assets = jnp.float32(arrays["initial_assets"])
    debt = jnp.float32(arrays["initial_debt"])

    net_worth_trajectory = jnp.zeros(n_years)
    cumulative_inflation = jnp.float32(1.0)

    for t in range(n_years):
        # Investment returns (on assets) — varies each year
        investment_gain = assets * annual_returns[t]

        # Income (already in real terms via growth rate)
        income = arrays["gross_income"][t]

        # Expenses grow with realized cumulative inflation
        cumulative_inflation = cumulative_inflation * (1 + annual_inflation[t])
        expenses = arrays["base_expenses"][t] * cumulative_inflation

        # Tax: use pre-computed effective rate (based on deterministic income)
        taxes = income * arrays["tax_rates"][t]

        # Net cash flow
        net_flow = income - taxes - expenses - arrays["debt_payments"][t]

        # Update state
        assets = assets + investment_gain + net_flow + arrays["employer_match"][t]
        assets = jnp.maximum(assets, 0.0)

        # Debt reduction (simplified: just reduce by payments)
        debt = jnp.maximum(debt - arrays["debt_payments"][t], 0.0)

        net_worth = assets - debt
        net_worth_trajectory = net_worth_trajectory.at[t].set(net_worth)

    # Record trajectory as a deterministic site for analysis
    numpyro.deterministic("net_worth_trajectory", net_worth_trajectory)
    numpyro.deterministic("final_net_worth", net_worth_trajectory[-1])

    return net_worth_trajectory


def run_forward(
    scenario: Scenario,
    num_samples: int = 1000,
    rng_seed: int = 0,
) -> dict:
    """Run forward simulation: sample outcome distributions given fixed decisions.

    Returns dict with 'net_worth_trajectory' array of shape (num_samples, n_years)
    and 'final_net_worth' array of shape (num_samples,).
    """
    arrays = _build_annual_arrays(scenario)
    predictive = Predictive(financial_model, num_samples=num_samples)
    rng_key = random.PRNGKey(rng_seed)
    samples = predictive(rng_key, arrays)
    return samples


def run_inverse(
    scenario: Scenario,
    target_prob: float = 0.95,
    num_warmup: int = 500,
    num_samples: int = 1000,
    rng_seed: int = 0,
) -> dict:
    """Run inverse inference: given outcome constraints, infer decision variables.

    This is a placeholder for the full inverse mode. The approach:
    1. Add decision variables as latent sites in the model
    2. Condition on the desired outcome
    3. Run MCMC to infer posteriors over decision variables

    For v1, this will be built out once forward mode is validated.
    """
    # TODO: implement inverse mode with conditioned model
    raise NotImplementedError(
        "Inverse mode is planned for after forward mode is validated. "
        "Use run_forward() with different scenarios to compare decisions for now."
    )

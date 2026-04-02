# Seldon

A probabilistic financial planning library for Python. Seldon models the impact of life decisions on long-term financial outcomes using Bayesian inference.

## Motivation

Standard financial planning tools produce single-number projections ("you'll have $X at retirement") that hide the uncertainty inherent in any long-term forecast. Seldon instead produces *distributions* over outcomes, letting you ask questions like:

- "Can I afford to take a lower-paying job?"
- "What if I put my kids in private school?"
- "When can I retire with 95% probability of not running out of money?"

## How It Works

Seldon builds a generative model of your financial life — income, expenses, investments, debts, taxes — and simulates it year-by-year from the present through end of life. Uncertain quantities (market returns, inflation, lifespan) are represented as probability distributions rather than point estimates.

### Two Modes

**Forward mode**: You fix your decisions (job, savings rate, retirement age, etc.) and Seldon samples the uncertain parameters to produce a distribution over outcomes. "If I do X, what are the chances I run out of money?"

**Inverse mode**: You fix a desired outcome (e.g., P(assets >= 0 at death) > 95%) and Seldon infers what decision variables achieve that via MCMC. "What savings rate / retirement age do I need to be 95% confident I won't run out of money?"

### What the Model Captures

- **Current state**: Cash, investments (by asset class), debts (mortgage, student loans, etc.), income, recurring expenses
- **Uncertain parameters**: Market returns (correlated across asset classes), inflation, income growth, lifespan
- **Decision variables**: Job/income changes, major expenses (private school, home purchase), retirement age, savings rate, asset allocation changes
- **Taxes**: Federal + state income tax, capital gains, tax-advantaged accounts (401k, IRA, Roth)
- **Life events**: Discrete state changes — new job, kid starts school, mortgage payoff, retirement, etc.

### Scenarios

Scenarios are composable. You define a base case representing your current trajectory, then layer decisions on top to create alternatives. Seldon compares outcome distributions across scenarios side by side.

## Technology

- **NumPyro** (JAX-based probabilistic programming) for both forward simulation and inverse inference
- **Pydantic** for financial state and scenario definitions
- **Matplotlib** for visualization (fan charts, CDFs, scenario comparisons)

## v1 Scope

- Manual assumption entry (specify your financial state, returns, expenses directly)
- Yearly time steps
- Core simulation loop with forward and inverse modes
- Scenario definition and comparison
- Basic visualization

### Future

- Import and calibrate from actual financial data (bank exports, brokerage data)
- Monthly time steps
- More sophisticated tax modeling
- Social Security modeling
- Healthcare cost modeling

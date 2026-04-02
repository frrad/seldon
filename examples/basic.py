"""Basic example: compare staying at current job vs. taking a lower-paying one."""

from seldon.state import (
    Account,
    AccountType,
    AssetClass,
    Debt,
    Expense,
    FinancialState,
    IncomeSource,
    LifeEvent,
    Scenario,
)
from seldon.model import run_forward
from seldon.analysis import summary
from seldon.viz import fan_chart, compare_scenarios
import matplotlib.pyplot as plt


# --- Define current financial state ---
state = FinancialState(
    age=35,
    retirement_age=65,
    life_expectancy=90,
    income=[
        IncomeSource(
            name="salary",
            annual_gross=150_000,
            tax_deferred_contribution=22_500,
            employer_match=9_000,
            growth_rate=0.03,
        )
    ],
    expenses=[
        Expense(name="housing", annual_amount=36_000),
        Expense(name="living", annual_amount=30_000),
        Expense(name="childcare", annual_amount=24_000),
    ],
    accounts=[
        Account(
            name="401k",
            account_type=AccountType.TRADITIONAL_401K,
            balance=200_000,
            allocation={AssetClass.US_STOCKS: 0.7, AssetClass.BONDS: 0.3},
        ),
        Account(
            name="brokerage",
            account_type=AccountType.TAXABLE,
            balance=100_000,
            allocation={AssetClass.US_STOCKS: 0.8, AssetClass.INTL_STOCKS: 0.2},
        ),
    ],
    debts=[
        Debt(name="mortgage", balance=300_000, interest_rate=0.065, minimum_payment=24_000, remaining_years=25),
    ],
)

# --- Shared life events (childcare ends, housing drops in retirement) ---
common_events = [
    LifeEvent(
        age=41,
        name="childcare_ends",
        description="Kids start school, no more daycare",
        remove_expenses=["childcare"],
    ),
    LifeEvent(
        age=60,
        name="mortgage_paid_off",
        description="Mortgage is done, housing costs drop",
        remove_expenses=["housing"],
        add_expenses=[Expense(name="housing", annual_amount=12_000)],  # taxes/insurance only
    ),
]

# --- Scenario A: Stay at current job ---
scenario_a = Scenario(
    name="Current Job",
    initial_state=state,
    events=common_events,
)

# --- Scenario B: Take lower-paying job at age 37 ---
scenario_b = Scenario(
    name="Lower-Paying Job",
    initial_state=state,
    events=[
        LifeEvent(
            age=37,
            name="new_job",
            description="Take a more fulfilling but lower-paying job",
            new_income=[
                IncomeSource(
                    name="new_salary",
                    annual_gross=110_000,
                    tax_deferred_contribution=15_000,
                    employer_match=5_500,
                    growth_rate=0.02,
                )
            ],
        ),
        *common_events,
    ],
)

# --- Run simulations ---
print("Running Scenario A: Current Job...")
results_a = run_forward(scenario_a, num_samples=2000)
print(summary(results_a, start_age=state.age))
print()

print("Running Scenario B: Lower-Paying Job...")
results_b = run_forward(scenario_b, num_samples=2000)
print(summary(results_b, start_age=state.age))
print()

# --- Plot ---
fig = compare_scenarios(
    {"Current Job": results_a, "Lower-Paying Job": results_b},
    start_age=state.age,
)
fig.savefig("comparison.png", dpi=150, bbox_inches="tight")
print("Saved comparison.png")

fig2 = fan_chart(results_a, start_age=state.age, title="Current Job - Net Worth")
fig2.savefig("current_job.png", dpi=150, bbox_inches="tight")
print("Saved current_job.png")

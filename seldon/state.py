"""Financial state and scenario definitions."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel


class AssetClass(str, Enum):
    US_STOCKS = "us_stocks"
    INTL_STOCKS = "intl_stocks"
    BONDS = "bonds"
    CASH = "cash"
    REAL_ESTATE = "real_estate"


class AccountType(str, Enum):
    TAXABLE = "taxable"
    TRADITIONAL_401K = "traditional_401k"
    TRADITIONAL_IRA = "traditional_ira"
    ROTH_401K = "roth_401k"
    ROTH_IRA = "roth_ira"


class Account(BaseModel):
    """An investment or savings account."""

    name: str
    account_type: AccountType
    balance: float
    allocation: dict[AssetClass, float]  # fractions summing to 1.0


class Debt(BaseModel):
    """A liability with a repayment schedule."""

    name: str
    balance: float
    interest_rate: float  # annual, e.g. 0.065 for 6.5%
    minimum_payment: float  # annual
    remaining_years: int | None = None  # None = revolving


class IncomeSource(BaseModel):
    """A source of income."""

    name: str
    annual_gross: float
    tax_deferred_contribution: float = 0.0  # annual 401k/etc contribution
    employer_match: float = 0.0  # annual employer match amount
    growth_rate: float = 0.02  # expected annual raise


class Expense(BaseModel):
    """A recurring expense category."""

    name: str
    annual_amount: float
    inflation_adjusted: bool = True  # grows with inflation?


class FinancialState(BaseModel):
    """Complete snapshot of someone's financial situation."""

    age: int
    retirement_age: int = 65
    life_expectancy: int = 90  # used as upper bound; actual lifespan is uncertain

    income: list[IncomeSource] = []
    expenses: list[Expense] = []
    accounts: list[Account] = []
    debts: list[Debt] = []

    filing_status: str = "single"  # single, married_joint, married_separate
    state: str = "CA"  # for state tax estimation

    @property
    def total_assets(self) -> float:
        return sum(a.balance for a in self.accounts)

    @property
    def total_debt(self) -> float:
        return sum(d.balance for d in self.debts)

    @property
    def net_worth(self) -> float:
        return self.total_assets - self.total_debt


class LifeEvent(BaseModel):
    """A discrete change that happens at a specific age."""

    age: int  # when this event occurs
    name: str
    description: str = ""

    # Changes to apply (None = no change)
    new_income: list[IncomeSource] | None = None  # replaces all income if set
    add_expenses: list[Expense] | None = None
    remove_expenses: list[str] | None = None  # by name
    add_debts: list[Debt] | None = None
    retirement_age: int | None = None  # override retirement age
    add_accounts: list[Account] | None = None


class Scenario(BaseModel):
    """A complete scenario: initial state + life events."""

    name: str
    description: str = ""
    initial_state: FinancialState
    events: list[LifeEvent] = []

"""Central configuration: paths, the observation window, and tunable scales.

Every number that shapes the synthetic world lives here so the generated
population is reproducible and the assumptions are auditable in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- paths ----

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SQL_DIR = PROJECT_ROOT / "sql"
DOCS_DIR = PROJECT_ROOT / "docs"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------- time structure ----
#
# The panel runs for 15 months. Analytics never uses all 15 at once:
#
#   months  1..12  training features      -> label observed in 13..15
#   months  4..15  scoring features       -> outcome is unobserved (the future)
#
# Two snapshots, offset by three months. This is what keeps the outcome out of
# the feature window; see docs/model_governance.md.

N_MONTHS = 15
TRAIN_FEATURE_MONTHS = tuple(range(1, 13))     # 1..12
TRAIN_OUTCOME_MONTHS = tuple(range(13, 16))    # 13..15
SCORE_FEATURE_MONTHS = tuple(range(4, 16))     # 4..15
OUTCOME_HORIZON_MONTHS = len(TRAIN_OUTCOME_MONTHS)

PANEL_START = "2024-10-01"  # month index 1


# ----------------------------------------------------------- population ----

N_CUSTOMERS = 50_000
RANDOM_SEED = 20240917

REGIONS = (
    "Central", "Coastal", "Eastern", "Highlands",
    "Lakeside", "Northern", "Western",
)

EMPLOYMENT_TYPES = ("Salaried", "Self-employed", "Business owner", "Informal", "Retired")

INCOME_BANDS = ("Band 1 (lowest)", "Band 2", "Band 3", "Band 4", "Band 5 (highest)")

SEGMENTS = (
    "Emerging",
    "Mass Market",
    "Savings Focused",
    "Credit Dependent",
    "Digital First",
    "Affluent",
    "High Value",
)

ACCOUNT_TYPES = ("Current", "Savings", "Fixed deposit", "Wallet")
PRODUCT_TYPES = ("Current account", "Savings", "Investment", "Personal loan",
                 "Asset finance", "Card", "Insurance", "Digital wallet")
LOAN_TYPES = ("Personal loan", "Asset finance", "Overdraft", "Microloan")
CHANNELS = ("Mobile app", "Internet banking", "Branch", "ATM", "Agent", "Card POS")
MERCHANT_CATEGORIES = (
    "Groceries", "Fuel", "Utilities", "Airtime", "Retail", "Restaurants",
    "Travel", "Healthcare", "Education", "Transfers", "Cash withdrawal", "Other",
)
INTERACTION_CHANNELS = ("Call centre", "Branch", "In-app chat", "Email", "Social")
INTERACTION_TYPES = ("Query", "Complaint", "Service request", "Product enquiry", "Dispute")
RESOLUTION_STATUSES = ("Resolved", "Pending", "Escalated", "Unresolved")

# Currency is deliberately unnamed: these are synthetic monetary units ("MU").
CURRENCY_LABEL = "MU"


# ------------------------------------------------ data quality injection ----


@dataclass(frozen=True)
class DefectRates:
    """Share of rows deliberately corrupted in each raw table.

    Every rate here is reproduced in the data quality report, so the checks can
    be verified against a known answer rather than merely 'looking reasonable'.
    """

    customer_missing_region: float = 0.021
    customer_missing_income_band: float = 0.034
    customer_impossible_age: float = 0.004
    customer_duplicates: float = 0.006
    customer_missing_id: float = 0.002
    account_negative_balance: float = 0.005
    account_invalid_status: float = 0.007
    account_orphaned: float = 0.004
    account_invalid_open_date: float = 0.003
    transaction_duplicates: float = 0.009
    transaction_missing_amount: float = 0.006
    transaction_extreme_amount: float = 0.0015
    transaction_future_date: float = 0.002
    product_inconsistent_code: float = 0.025
    loan_missing_rate: float = 0.012
    interaction_missing_satisfaction: float = 0.045


DEFECTS = DefectRates()


@dataclass(frozen=True)
class OpportunityWeights:
    """Weights of the Relationship Opportunity Score.

    Published, not tuned: the score is a stated management heuristic, not a
    fitted model, and is presented that way everywhere in the product.
    """

    value: float = 0.30
    engagement: float = 0.20
    product_headroom: float = 0.20
    growth_propensity: float = 0.20
    stability: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return {
            "Customer value": self.value,
            "Engagement": self.engagement,
            "Product headroom": self.product_headroom,
            "Growth propensity": self.growth_propensity,
            "Relationship stability": self.stability,
        }

    def total(self) -> float:
        return sum(self.as_dict().values())


OPPORTUNITY_WEIGHTS = OpportunityWeights()

# Risk bands are fixed cut-offs on predicted probability, chosen once and
# documented, rather than quantiles that drift as the population changes.
CHURN_BANDS: tuple[tuple[str, float], ...] = (("Low", 0.10), ("Moderate", 0.25), ("Elevated", 0.45), ("High", 1.01))
PROPENSITY_BANDS: tuple[tuple[str, float], ...] = (("Low", 0.10), ("Moderate", 0.25), ("High", 1.01))
OPPORTUNITY_BANDS: tuple[tuple[str, float], ...] = (("Low", 40.0), ("Moderate", 60.0), ("High", 80.0), ("Very high", 100.01))


def band(value: float, bands: tuple[tuple[str, float], ...], default: str = "Unknown") -> str:
    """Map a score to its band label using upper-exclusive thresholds."""
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    for label, upper in bands:
        if v < upper:
            return label
    return bands[-1][0]

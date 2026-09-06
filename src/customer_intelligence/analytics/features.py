"""The modelling feature contract.

One list, used by training and by scoring. If a feature is not named here it
does not reach a model, which makes the "what went in?" question in
docs/model_governance.md answerable by reading a single file.
"""

from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES = [
    # who they are
    "age", "tenure_months", "monthly_income",
    # what they hold
    "account_count", "active_accounts", "dormant_accounts", "account_types_held",
    "product_count", "has_savings", "has_card", "has_insurance", "has_wallet",
    # what they do
    "avg_monthly_balance", "peak_balance", "transaction_count", "transaction_value",
    "avg_monthly_transactions", "avg_transaction_value", "total_inflow", "total_outflow", "balance_to_income", "outflow_to_inflow",
    "total_logins", "avg_monthly_logins", "digital_transactions", "failed_logins",
    "max_channels_used", "digital_share", "active_months", "months_since_last_activity",
    # service experience
    "service_interactions", "complaints", "unresolved_interactions", "avg_satisfaction",
    # credit
    "loan_count", "loan_exposure", "avg_interest_rate", "ever_in_arrears", "ever_defaulted",
    # direction of travel
    "balance_trend", "transaction_trend", "digital_trend", "complaints_recent",
]

CATEGORICAL_FEATURES = ["region", "employment_type", "income_band"]

# See features_retail.HEAVY_TAILED_FEATURES for the reasoning: log1p before
# standardising, so a linear model is not extrapolating on quantities that span
# orders of magnitude.
HEAVY_TAILED_FEATURES = [
    "monthly_income", "avg_monthly_balance", "peak_balance", "transaction_count",
    "transaction_value", "avg_transaction_value", "total_inflow", "total_outflow",
    "total_logins", "digital_transactions", "failed_logins", "loan_exposure",
    "balance_to_income", "outflow_to_inflow",
]

# Deliberately excluded, and why:
#
#   customer_id        an identifier, not a signal
#   declared_segment   a rule applied to the same attributes the model already
#                      sees. Including it would let the model shortcut through a
#                      label rather than learn behaviour, and would make the
#                      explanations circular ("high risk because segment X").
#   has_investment /   the target of the corresponding propensity model. Held
#   has_lending        out per-model rather than globally: see model_features().
#   window_from/to     bookkeeping
#   gender             a protected attribute. It is present in the source data,
#                      because a real extract would contain it, and it is shown
#                      in the Customer 360 profile. It is NOT a model input.
#                      "It was in the data" is not a reason to let a commercial
#                      targeting model condition on it: doing so would let the
#                      system allocate retention effort and product offers by
#                      gender, and the fact that the model found it predictive
#                      would be the problem, not the justification. It is
#                      retained for outcome *monitoring* -- checking whether
#                      flag rates differ across groups -- which is the use that
#                      requires holding the attribute at all.
EXCLUDED_FEATURES = [
    "customer_id", "declared_segment", "gender", "window_from", "window_to",
]

# Human-readable names for anything shown to a user.
FEATURE_LABELS = {
    "age": "Age",
    "tenure_months": "Tenure (months)",
    "monthly_income": "Declared monthly income",
    "account_count": "Accounts held",
    "active_accounts": "Active accounts",
    "dormant_accounts": "Dormant accounts",
    "account_types_held": "Distinct account types",
    "product_count": "Products held",
    "has_savings": "Holds savings", "has_card": "Holds card",
    "has_insurance": "Holds insurance", "has_wallet": "Holds digital wallet",
    "avg_monthly_balance": "Average monthly balance",
    "peak_balance": "Peak balance",
    "transaction_count": "Transactions in window",
    "transaction_value": "Transaction value in window",
    "avg_monthly_transactions": "Transactions per month",
    "avg_transaction_value": "Average transaction size",
    "total_inflow": "Total inflow", "total_outflow": "Total outflow",
    "balance_to_income": "Balance held per unit of income",
    "outflow_to_inflow": "Outflow against inflow",
    "total_logins": "Logins in window", "avg_monthly_logins": "Logins per month",
    "digital_transactions": "Digital transactions", "failed_logins": "Failed logins",
    "max_channels_used": "Channels used", "digital_share": "Share of activity digital",
    "active_months": "Months with activity",
    "months_since_last_activity": "Months since last activity",
    "service_interactions": "Service contacts", "complaints": "Complaints",
    "unresolved_interactions": "Unresolved contacts",
    "avg_satisfaction": "Average satisfaction",
    "loan_count": "Loans held", "loan_exposure": "Loan exposure",
    "avg_interest_rate": "Average interest rate",
    "ever_in_arrears": "Has been in arrears", "ever_defaulted": "Has defaulted",
    "balance_trend": "Balance trend (recent vs early)",
    "transaction_trend": "Transaction trend (recent vs early)",
    "digital_trend": "Digital trend (recent vs early)",
    "complaints_recent": "Complaints in last quarter",
    "region": "Region", "employment_type": "Employment", "income_band": "Income band",
    "gender": "Gender",
}


def label(feature: str) -> str:
    """Readable name for a raw or one-hot-encoded feature."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    for cat in CATEGORICAL_FEATURES:
        if feature.startswith(f"{cat}_"):
            return f"{FEATURE_LABELS.get(cat, cat)}: {feature[len(cat) + 1:]}"
    return feature.replace("_", " ").capitalize()


def model_features(target: str) -> tuple[list[str], list[str]]:
    """Numeric and categorical features for one target.

    A propensity model must not see whether the customer already holds the
    product it is predicting uptake of.
    """
    numeric = list(NUMERIC_FEATURES)
    if target == "took_investment":
        numeric = [f for f in numeric if f != "has_investment"]
    elif target == "took_lending":
        numeric = [f for f in numeric if f not in ("has_lending", "loan_count", "loan_exposure")]
    return numeric, list(CATEGORICAL_FEATURES)


def prepare(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Select the modelling columns, tolerating any that are absent."""
    cols = [c for c in numeric + categorical if c in df.columns]
    return df[cols].copy()

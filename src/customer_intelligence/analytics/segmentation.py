"""Segmentation: a business rule set, and a clustering, compared.

Both are built because they answer different questions. The rule set is what an
organisation can actually operate -- stable, explicable, and unchanged when the
data is refreshed. The clustering is a check on it: if K-means finds structure
the rules miss, the rules are wrong somewhere.

The comparison is the deliverable, not the clustering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# The dimensions a relationship is actually managed on.
CLUSTER_FEATURES = [
    "avg_monthly_balance", "transaction_count", "avg_transaction_value",
    "product_count", "avg_monthly_logins", "digital_share",
    "loan_exposure", "tenure_months", "active_months",
]

SEGMENT_ORDER = [
    "High Value", "Affluent", "Digital First", "Savings Focused",
    "Credit Dependent", "Mass Market", "Emerging", "Dormant / Lapsing",
]

SEGMENT_DESCRIPTIONS = {
    "High Value": "The top of the book by balance and product depth. Small in number, large in value, and the most expensive to lose.",
    "Affluent": "Substantial balances and several products, without the concentration of the top tier. The natural growth pool for investment products.",
    "Digital First": "Ordinary balances, heavy app use, low branch contact. Cheap to serve and responsive to in-channel offers.",
    "Savings Focused": "Accumulating balances, little borrowing, low transaction frequency. Stable and under-served rather than inactive.",
    "Credit Dependent": "Relationship centred on borrowing. Value is real but so is credit risk; arrears drive both.",
    "Mass Market": "The centre of the distribution. Modest on every dimension, and the largest group by count.",
    "Emerging": "Early or low-balance relationships. Individually small, collectively the pipeline.",
    "Dormant / Lapsing": "Little or no activity in the window. Attrition has largely already happened here.",
}


def rule_segments(c360: pd.DataFrame) -> pd.Series:
    """Assign segments by published rules.

    Ordered so the most specific condition wins. Thresholds are population
    quantiles computed once, so the rule set adapts to the book without being
    silently redefined by every refresh.
    """
    balance = c360["avg_monthly_balance"]
    b_hi = balance.quantile(0.95)
    b_mid = balance.quantile(0.80)
    logins_hi = c360["avg_monthly_logins"].quantile(0.75)
    digital_hi = c360["digital_share"].fillna(0).quantile(0.70)

    seg = pd.Series("Mass Market", index=c360.index, dtype=object)

    seg[(balance < balance.quantile(0.35)) & (c360["product_count"] <= 2)] = "Emerging"

    savings_focused = (
        (c360["has_savings"] == 1) & (c360["loan_exposure"] <= 0)
        & (c360["avg_monthly_transactions"] < c360["avg_monthly_transactions"].median())
        & (balance >= balance.quantile(0.35))
    )
    seg[savings_focused] = "Savings Focused"

    credit_dependent = (c360["loan_exposure"] > 0) & (
        c360["loan_exposure"] > balance * 1.5
    )
    seg[credit_dependent] = "Credit Dependent"

    digital_first = (
        (c360["avg_monthly_logins"] >= logins_hi)
        & (c360["digital_share"].fillna(0) >= digital_hi)
        & (balance < b_mid)
    )
    seg[digital_first] = "Digital First"

    seg[(balance >= b_mid) & (balance < b_hi)] = "Affluent"
    seg[balance >= b_hi] = "High Value"

    # Inactivity overrides everything: a lapsed relationship is not an affluent
    # one, whatever the balance says. The cut-off is a third of the window --
    # at "active in one month of twelve" the rule caught five customers in
    # fifty thousand, which is a segment nobody can operate.
    seg[c360["active_months"] <= 4] = "Dormant / Lapsing"
    return seg


def kmeans_segments(c360: pd.DataFrame, k: int = 6) -> tuple[pd.Series, dict]:
    """Unsupervised comparison. Log-scaled first, because balances and
    transaction values span orders of magnitude and K-means is a distance
    method: without it, the clustering is a partition of the richest 1%.
    """
    X = c360[[c for c in CLUSTER_FEATURES if c in c360.columns]].copy()
    for col in X.columns:
        if X[col].min() >= 0 and X[col].max() > 1000:
            X[col] = np.log1p(X[col])

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    Z = pipe.fit_transform(X)

    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(Z)

    sample = np.random.default_rng(RANDOM_STATE).choice(
        len(Z), size=min(5000, len(Z)), replace=False
    )
    diagnostics = {
        "k": k,
        "silhouette": float(silhouette_score(Z[sample], labels[sample])),
        "inertia": float(km.inertia_),
    }
    return pd.Series([f"Cluster {i + 1}" for i in labels], index=c360.index), diagnostics


def choose_k(c360: pd.DataFrame, candidates: range = range(3, 9)) -> pd.DataFrame:
    """Silhouette and inertia across k, so the choice of k is shown, not asserted."""
    rows = []
    for k in candidates:
        _, diag = kmeans_segments(c360, k=k)
        rows.append(diag)
    return pd.DataFrame(rows)


def segment_profiles(c360: pd.DataFrame, segment_col: str = "segment") -> pd.DataFrame:
    """The table that turns a segment label into a business description."""
    agg = c360.groupby(segment_col, observed=True).agg(
        customers=("customer_id", "size"),
        avg_balance=("avg_monthly_balance", "mean"),
        median_balance=("avg_monthly_balance", "median"),
        avg_products=("product_count", "mean"),
        avg_monthly_transactions=("avg_monthly_transactions", "mean"),
        avg_transaction_value=("avg_transaction_value", "mean"),
        avg_logins=("avg_monthly_logins", "mean"),
        digital_share=("digital_share", "mean"),
        avg_loan_exposure=("loan_exposure", "mean"),
        arrears_rate=("ever_in_arrears", "mean"),
        avg_tenure=("tenure_months", "mean"),
        complaint_rate=("complaints", "mean"),
        active_months=("active_months", "mean"),
    ).reset_index()

    agg["share_of_customers"] = agg["customers"] / agg["customers"].sum() * 100
    total_balance = (c360["avg_monthly_balance"].sum())
    balance_by_seg = c360.groupby(segment_col, observed=True)["avg_monthly_balance"].sum()
    agg["share_of_balances"] = agg[segment_col].map(balance_by_seg) / total_balance * 100
    agg["description"] = agg[segment_col].map(SEGMENT_DESCRIPTIONS).fillna("")

    order = {name: i for i, name in enumerate(SEGMENT_ORDER)}
    agg["_o"] = agg[segment_col].map(order).fillna(99)
    return agg.sort_values("_o", ignore_index=True).drop(columns="_o")


def crosstab(c360: pd.DataFrame, a: str = "segment", b: str = "cluster") -> pd.DataFrame:
    """Rules against clusters. Where they disagree is where the interesting
    question is -- not which one is 'right'."""
    return pd.crosstab(c360[a], c360[b])

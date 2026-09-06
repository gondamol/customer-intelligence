"""The Relationship Opportunity Score.

A stated management heuristic, not a fitted model. It answers "where is the
headroom in this relationship?" -- deliberately separate from churn risk, which
answers "is this relationship ending?". Keeping them apart is what makes the
risk/opportunity matrix meaningful: a customer can be high on both, and that
combination is the one worth acting on first.

Every component is a percentile rank within the book, so the score is a relative
statement ("top decile of opportunity"), never an absolute currency amount. It
is not a revenue forecast and is never presented as one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import OPPORTUNITY_WEIGHTS

COMPONENT_DEFINITIONS = {
    "Customer value": "Balances and transaction value today. What the relationship is currently worth.",
    "Engagement": "Activity and digital use. An engaged customer can be reached; a silent one cannot.",
    "Product headroom": "Products *not* held. Depth of relationship left to build, so a customer holding everything scores low here by design.",
    "Growth propensity": "Modelled likelihood of taking an investment or lending product next period.",
    "Relationship stability": "Consistency of activity, net of service friction. Discounts opportunity that rests on a deteriorating relationship.",
}


def _pct_rank(s: pd.Series) -> pd.Series:
    """Percentile rank, 0–100, missing treated as the population median."""
    return s.rank(pct=True, na_option="keep").fillna(0.5) * 100


def opportunity_components(
    c360: pd.DataFrame,
    investment_propensity: np.ndarray | pd.Series,
    lending_propensity: np.ndarray | pd.Series,
    churn_risk: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """The five components, each on a 0–100 scale."""
    df = pd.DataFrame(index=c360.index)

    # 1. What the relationship is worth now.
    df["Customer value"] = (
        0.6 * _pct_rank(c360["avg_monthly_balance"])
        + 0.4 * _pct_rank(c360["transaction_value"])
    )

    # 2. Whether the customer is reachable.
    df["Engagement"] = (
        0.4 * _pct_rank(c360["avg_monthly_logins"])
        + 0.3 * _pct_rank(c360["avg_monthly_transactions"])
        + 0.3 * _pct_rank(c360["active_months"])
    )

    # 3. Room left to grow. Inverted: holding everything means no headroom.
    max_products = 8
    df["Product headroom"] = (
        (max_products - c360["product_count"].clip(0, max_products)) / max_products * 100
    )

    # 4. Likelihood of taking something next period.
    df["Growth propensity"] = (
        0.5 * _pct_rank(pd.Series(np.asarray(investment_propensity), index=c360.index))
        + 0.5 * _pct_rank(pd.Series(np.asarray(lending_propensity), index=c360.index))
    )

    # 5. Is the relationship holding together? Opportunity resting on a customer
    #    who is leaving is not opportunity.
    stability = (
        0.45 * _pct_rank(c360["balance_trend"].clip(0, 3))
        + 0.35 * _pct_rank(c360["transaction_trend"].clip(0, 3))
        + 0.20 * (100 - _pct_rank(c360["complaints"]))
    )
    df["Relationship stability"] = stability * (1 - 0.5 * np.asarray(churn_risk))

    return df.clip(0, 100)


def opportunity_score(components: pd.DataFrame) -> pd.Series:
    """Weighted sum of the components, 0–100.

    Weights are applied by position, not by name. Every domain contributes the
    same five roles in the same order -- value, engagement, headroom, growth,
    reliability -- but calls them what makes sense in that business, and the
    published weights should not have to be restated for each one.
    """
    weights = list(OPPORTUNITY_WEIGHTS.as_dict().values())
    if len(components.columns) != len(weights):
        raise ValueError(
            f"opportunity score expects {len(weights)} components, "
            f"got {len(components.columns)}: {list(components.columns)}"
        )
    total = sum(weights)
    score = sum(components[col] * w for col, w in zip(components.columns, weights)) / total
    return score.clip(0, 100).round(1)


def weights_for(components: pd.DataFrame) -> dict[str, float]:
    """The published weights, labelled with this domain's component names."""
    return dict(zip(components.columns, OPPORTUNITY_WEIGHTS.as_dict().values()))


def risk_opportunity_quadrant(
    opportunity: pd.Series, churn_probability: pd.Series,
    opportunity_cut: float = 60.0, risk_cut: float = 0.25,
) -> pd.Series:
    """Place each customer in the risk/opportunity matrix.

    The names are instructions, not descriptions -- the point of the matrix is
    that it tells a team what to do with each group.
    """
    scored = churn_probability.notna()
    high_opp = opportunity >= opportunity_cut
    high_risk = churn_probability >= risk_cut
    return pd.Series(
        np.select(
            [~scored,
             high_opp & high_risk, high_opp & ~high_risk,
             ~high_opp & high_risk, ~high_opp & ~high_risk],
            ["Not scored", "Retain", "Grow", "Monitor", "Develop"],
            default="Not scored",
        ),
        index=opportunity.index,
    )


QUADRANT_MEANING = {
    "Not scored": "Too little trading history for the models to say anything. Served through low-cost channels, and the recommendation says why rather than implying a judgement that was never made.",
    "Retain": "Worth keeping and at risk of leaving. The most time-critical group: act before the behaviour completes.",
    "Grow": "Worth keeping and stable. Where proactive relationship effort earns the most.",
    "Monitor": "At risk, but limited headroom. Handle through low-cost automated channels rather than scarce human attention.",
    "Develop": "Stable with limited headroom today. Serve well, revisit as circumstances change.",
}


# ---------------------------------------------------------------- retail ----

RETAIL_COMPONENT_DEFINITIONS = {
    "Account value": "Revenue and order value over the window. What the relationship is worth today.",
    "Engagement": "Order frequency and how much of the year the account was active. An engaged account can be reached; a silent one cannot.",
    "Range headroom": "Categories *not* bought. Room left to widen the relationship, so an account already buying everything scores low here by design.",
    "Growth propensity": "Modelled likelihood of spending above its own run-rate, and of taking a category it has never bought.",
    "Reliability": "Ordering consistency net of returns. Discounts opportunity that rests on an erratic or return-heavy account.",
}


def opportunity_components_retail(c360: pd.DataFrame, n_categories: int = 12) -> pd.DataFrame:
    """The five components for a wholesale account book.

    Same structure and the same published weights as every other domain: what
    the account is worth, whether it can be reached, how much room is left,
    whether it is likely to grow, and whether any of that is dependable.
    """
    df = pd.DataFrame(index=c360.index)

    df["Account value"] = (
        0.6 * _pct_rank(c360["revenue"])
        + 0.4 * _pct_rank(c360["avg_order_value"].fillna(0))
    )

    df["Engagement"] = (
        0.5 * _pct_rank(c360["invoices"])
        + 0.5 * _pct_rank(c360["active_months"])
    )

    df["Range headroom"] = (
        (n_categories - c360["distinct_categories"].clip(0, n_categories))
        / n_categories * 100
    )

    growth = c360.get("growth_propensity")
    expansion = c360.get("expansion_propensity")
    df["Growth propensity"] = (
        0.5 * _pct_rank(pd.Series(np.asarray(growth), index=c360.index).fillna(0.0))
        + 0.5 * _pct_rank(pd.Series(np.asarray(expansion), index=c360.index).fillna(0.0))
    )

    # Reliability discounts an account whose ordering is erratic or whose value
    # keeps coming back as returns, then discounts it again by lapse risk --
    # opportunity resting on an account that is leaving is not opportunity.
    reliability = (
        0.45 * _pct_rank(c360["revenue_trend"].clip(0, 3))
        + 0.35 * (100 - _pct_rank(c360["gap_variability"].fillna(0)))
        + 0.20 * (100 - _pct_rank(c360["return_rate"].fillna(0)))
    )
    lapse = pd.Series(np.asarray(c360.get("lapse_risk")), index=c360.index).fillna(0.5)
    df["Reliability"] = reliability * (1 - 0.5 * lapse)

    return df.clip(0, 100)

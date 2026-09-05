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
    """Weighted sum of the components, 0–100."""
    weights = OPPORTUNITY_WEIGHTS.as_dict()
    total = sum(weights.values())
    score = sum(components[name] * w for name, w in weights.items()) / total
    return score.clip(0, 100).round(1)


def risk_opportunity_quadrant(
    opportunity: pd.Series, churn_probability: pd.Series,
    opportunity_cut: float = 60.0, risk_cut: float = 0.25,
) -> pd.Series:
    """Place each customer in the risk/opportunity matrix.

    The names are instructions, not descriptions -- the point of the matrix is
    that it tells a team what to do with each group.
    """
    high_opp = opportunity >= opportunity_cut
    high_risk = churn_probability >= risk_cut
    return pd.Series(
        np.select(
            [high_opp & high_risk, high_opp & ~high_risk,
             ~high_opp & high_risk, ~high_opp & ~high_risk],
            ["Retain", "Grow", "Monitor", "Develop"],
            default="Develop",
        ),
        index=opportunity.index,
    )


QUADRANT_MEANING = {
    "Retain": "Worth keeping and at risk of leaving. The most time-critical group: act before the behaviour completes.",
    "Grow": "Worth keeping and stable. Where proactive relationship effort earns the most.",
    "Monitor": "At risk, but limited headroom. Handle through low-cost automated channels rather than scarce human attention.",
    "Develop": "Stable with limited headroom today. Serve well, revisit as circumstances change.",
}

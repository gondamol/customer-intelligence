"""The next-best-action engine.

This is where prediction becomes a recommendation, and it is deliberately a
transparent rule set rather than a model. Three reasons:

1. A recommendation has to be *arguable*. A relationship manager who disagrees
   needs to see which condition fired, and be able to say it is wrong.
2. The rules encode business policy -- who gets human contact, who gets an
   automated nudge -- and policy should be written down and changed
   deliberately, not learned from whatever happened last quarter.
3. Learning actions from historical outcomes learns the historical policy,
   including its mistakes.

The models supply the evidence. The rules decide what to do with it. Every
recommendation carries the conditions that produced it and the model reasons
underneath, and every one is phrased as a suggestion for a person to weigh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CHURN_BANDS, OPPORTUNITY_BANDS, PROPENSITY_BANDS, band


@dataclass
class Action:
    key: str
    title: str
    channel: str
    priority: str            # Urgent | High | Standard | Low
    rationale: str
    cost: str                # Relationship manager | Contact centre | Automated
    guardrail: str = ""


ACTION_CATALOGUE: dict[str, Action] = {
    "retention_rm": Action(
        key="retention_rm",
        title="Relationship-manager retention contact",
        channel="Named relationship manager, by phone",
        priority="Urgent",
        cost="Relationship manager",
        rationale="High-value relationship showing sustained disengagement. Human contact is justified by the value at stake.",
        guardrail="Review the account before contact. Do not lead with an offer where the driver is unresolved service failure.",
    ),
    "retention_service": Action(
        key="retention_service",
        title="Service recovery call",
        channel="Contact centre, outbound",
        priority="Urgent",
        cost="Contact centre",
        rationale="Disengagement is accompanied by unresolved service contacts. The service issue is the more likely cause than price or product.",
        guardrail="Resolve the open case first. A retention offer before resolution reads as an attempt to buy silence.",
    ),
    "retention_digital": Action(
        key="retention_digital",
        title="Automated re-engagement journey",
        channel="In-app and email",
        priority="High",
        cost="Automated",
        rationale="Declining activity in a relationship where the cost of human contact would exceed the value at risk.",
        guardrail="Cap contact frequency. Suppress if the customer has opted out of marketing.",
    ),
    "grow_investment": Action(
        key="grow_investment",
        title="Investment conversation",
        channel="Relationship manager or advised channel",
        priority="High",
        cost="Relationship manager",
        rationale="Stable relationship with accumulating balances and a high modelled likelihood of taking an investment product.",
        guardrail="Suitability assessment required before any product is discussed. The score indicates interest, never suitability.",
    ),
    "grow_lending": Action(
        key="grow_lending",
        title="Lending eligibility review",
        channel="Relationship manager or branch",
        priority="Standard",
        cost="Relationship manager",
        rationale="Behaviour consistent with credit demand, with no current arrears.",
        guardrail="Affordability and credit assessment govern the outcome. This score is not a credit decision and carries no weight in one.",
    ),
    "deepen_digital": Action(
        key="deepen_digital",
        title="In-channel product prompt",
        channel="Mobile app",
        priority="Standard",
        cost="Automated",
        rationale="Digitally active customer with unused product headroom. Reachable at near-zero marginal cost in the channel already used.",
        guardrail="Single prompt per period. Do not repeat after dismissal.",
    ),
    "arrears_support": Action(
        key="arrears_support",
        title="Collections and support review",
        channel="Collections team",
        priority="Urgent",
        cost="Contact centre",
        rationale="Facility in arrears. Growth actions are suspended while the credit position is unresolved.",
        guardrail="Follow the arrears and forbearance policy. No cross-sell of any kind while a facility is in arrears.",
    ),
    "onboard_activate": Action(
        key="onboard_activate",
        title="Activation journey",
        channel="In-app and SMS",
        priority="Standard",
        cost="Automated",
        rationale="Early-tenure relationship that has not yet established a regular pattern of use. Activation now determines the relationship's shape.",
        guardrail="Do not target customers who have opted out of onboarding communications.",
    ),
    "service_recovery": Action(
        key="service_recovery",
        title="Resolve open service case",
        channel="Service team",
        priority="High",
        cost="Contact centre",
        rationale="Unresolved or escalated contacts outstanding. Resolution precedes any commercial conversation.",
        guardrail="Close the loop with the customer once resolved.",
    ),
    "maintain": Action(
        key="maintain",
        title="No action this period",
        channel="—",
        priority="Low",
        cost="Automated",
        rationale="Stable relationship with no risk signal and no clear headroom. Contacting without a reason spends goodwill.",
        guardrail="Deliberately choosing not to contact is a decision, and is recorded as one.",
    ),
}


@dataclass
class Recommendation:
    customer_id: str
    action: Action
    conditions: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    confidence: str = "Moderate"

    def as_dict(self) -> dict:
        return {
            "customer_id": self.customer_id,
            "action": self.action.title,
            "channel": self.action.channel,
            "priority": self.action.priority,
            "cost": self.action.cost,
            "rationale": self.action.rationale,
            "guardrail": self.action.guardrail,
            "conditions": self.conditions,
            "reasons": self.reasons,
            "confidence": self.confidence,
        }


def _confidence(row: pd.Series) -> str:
    """How much weight the recommendation deserves.

    Driven by how much evidence there is about this customer, not by how extreme
    the score is. A confident-looking score on three months of thin activity is
    the case to be most careful about.
    """
    months = row.get("active_months", 0)
    products = row.get("product_count", 0)
    if months >= 9 and products >= 2:
        return "High"
    if months >= 5:
        return "Moderate"
    return "Low — limited activity history"


def recommend(
    row: pd.Series,
    model_reasons: list[dict] | None = None,
    thresholds: dict[str, float] | None = None,
) -> Recommendation:
    """Decide the suggested action for one customer.

    Rules are evaluated in priority order and the first match wins, so the
    ordering *is* the policy: credit position before growth, service failure
    before commercial contact, retention before cross-sell.

    ``thresholds`` carries each model's operating threshold -- the probability
    above which it flags a customer, set to the share of the book a team can
    actually contact. The engine acts on those rather than on the descriptive
    probability bands, so the capacity assumption is made once, in one place,
    and the engine cannot recommend outreach to more customers than the models
    were thresholded to flag. Without them the bands are used, which is looser.
    """
    thresholds = thresholds or {}
    churn_p = float(row.get("churn_probability", 0.0))
    churn_band = band(churn_p, CHURN_BANDS)
    opp = float(row.get("opportunity_score", 0.0))
    opp_band = band(opp, OPPORTUNITY_BANDS)
    inv_p = float(row.get("investment_propensity", 0.0) or 0.0)
    lend_p = float(row.get("lending_propensity", 0.0) or 0.0)
    inv_band = band(inv_p, PROPENSITY_BANDS)
    lend_band = band(lend_p, PROPENSITY_BANDS)

    # Flagged = above the model's own operating threshold, where one is supplied.
    inv_flagged = inv_p >= thresholds.get("investment", float("inf")) if thresholds \
        else inv_band == "High"
    lend_flagged = lend_p >= thresholds.get("lending", float("inf")) if thresholds \
        else lend_band == "High"
    churn_flagged = churn_p >= thresholds.get("churn", float("inf")) if thresholds \
        else churn_band in ("Elevated", "High")

    unresolved = float(row.get("unresolved_interactions", 0) or 0)
    in_arrears = int(row.get("ever_in_arrears", 0) or 0)
    tenure = float(row.get("tenure_months", 0) or 0)
    active_months = float(row.get("active_months", 0) or 0)
    high_value = opp >= 60 or float(row.get("avg_monthly_balance", 0) or 0) > 0

    conditions: list[str] = []

    def fire(key: str, *why: str) -> Recommendation:
        return Recommendation(
            customer_id=str(row.get("customer_id", "")),
            action=ACTION_CATALOGUE[key],
            conditions=list(why),
            reasons=[
                f"{r['label']}: {r['value']:,.1f} against a book average of {r['population_mean']:,.1f}"
                for r in (model_reasons or [])
            ],
            confidence=_confidence(row),
        )

    # 1. Credit position first. Nothing is sold to a customer in arrears.
    if in_arrears == 1:
        return fire("arrears_support", "Facility currently or previously in arrears")

    # 2. Unresolved service failure outranks any commercial conversation.
    if unresolved >= 2 and churn_flagged:
        return fire("retention_service",
                    f"{int(unresolved)} unresolved or escalated service contacts",
                    f"Attrition risk {churn_band.lower()} ({churn_p:.0%})")
    if unresolved >= 2:
        return fire("service_recovery", f"{int(unresolved)} unresolved or escalated service contacts")

    # 3. Retention, graded by what the relationship is worth.
    if churn_flagged:
        if opp_band in ("High", "Very high"):
            return fire("retention_rm",
                        f"Attrition risk {churn_band.lower()} ({churn_p:.0%})",
                        f"Relationship opportunity {opp_band.lower()} ({opp:.0f}/100)")
        return fire("retention_digital",
                    f"Attrition risk {churn_band.lower()} ({churn_p:.0%})",
                    f"Relationship opportunity {opp_band.lower()} ({opp:.0f}/100)")

    # 4. Early-tenure activation.
    if tenure <= 18 and active_months <= 7:
        return fire("onboard_activate",
                    f"Tenure {tenure:.0f} months",
                    f"Active in only {active_months:.0f} of the last 12 months")

    # 5. Growth, on a stable relationship.
    if inv_flagged and opp_band in ("High", "Very high"):
        return fire("grow_investment",
                    f"Investment propensity {inv_band.lower()}",
                    f"Attrition risk {churn_band.lower()}",
                    f"Relationship opportunity {opp:.0f}/100")
    if lend_flagged and churn_band == "Low":
        return fire("grow_lending",
                    f"Lending propensity {lend_band.lower()}",
                    "No arrears on record")
    if float(row.get("digital_share", 0) or 0) >= 0.5 and float(row.get("product_count", 0)) <= 3:
        return fire("deepen_digital",
                    f"{float(row.get('digital_share', 0)):.0%} of activity in digital channels",
                    f"{int(row.get('product_count', 0))} products held")

    # 6. Otherwise, deliberately do nothing.
    return fire("maintain",
                f"Attrition risk {churn_band.lower()}",
                f"Relationship opportunity {opp:.0f}/100")


def recommend_batch(
    scored: pd.DataFrame, thresholds: dict[str, float] | None = None
) -> pd.DataFrame:
    """Apply the rules across the book and return one row per customer."""
    records = [recommend(row, thresholds=thresholds).as_dict()
               for _, row in scored.iterrows()]
    out = pd.DataFrame(records)
    out["conditions"] = out["conditions"].apply(lambda c: " · ".join(c))
    return out.drop(columns=["reasons"])

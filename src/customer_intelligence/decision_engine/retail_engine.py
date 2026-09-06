"""The next-best-action rule set for a wholesale account book.

Same principle as every other part of this project: a transparent, ordered rule
set, not a model. A recommendation has to be arguable by the account manager who
receives it; the ordering encodes commercial policy and should be changed
deliberately; and learning actions from what happened last year learns last
year's policy, mistakes included.

The ordering is the policy. Here it is: fix a service problem before selling
anything, save a valuable account before growing a small one, and let doing
nothing be a reachable outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .engine import Action, Recommendation

RETAIL_ACTIONS: dict[str, Action] = {
    "winback_call": Action(
        key="winback_call",
        title="Account-manager win-back call",
        channel="Named account manager, by phone",
        priority="Urgent",
        cost="Account manager",
        rationale="A high-value account has gone quiet well past its own usual ordering gap. The value at stake justifies a person picking up the phone.",
        guardrail="Read the return and discount history before calling. Where an account went quiet after a run of returns, the conversation is about the service failure, not about a new order.",
    ),
    "winback_offer": Action(
        key="winback_offer",
        title="Structured win-back offer",
        channel="Email, with a dated incentive",
        priority="High",
        cost="Marketing",
        rationale="Long silent, with real history behind them. Worth one structured attempt rather than an indefinite campaign.",
        guardrail="One attempt per period. If it does not land, move the account to self-service rather than repeating.",
    ),
    "retention_check": Action(
        key="retention_check",
        title="Proactive check-in",
        channel="Account manager, email or short call",
        priority="High",
        cost="Account manager",
        rationale="An established account has slipped past its normal rhythm but is not yet lost. This is the cheapest point at which to intervene.",
        guardrail="Lead with a question about their trading, not a catalogue.",
    ),
    "service_recovery": Action(
        key="service_recovery",
        title="Resolve the returns problem",
        channel="Customer service",
        priority="Urgent",
        cost="Customer service",
        rationale="An unusually high share of this account's value has come back as returns. Whatever is wrong will not be fixed by another order.",
        guardrail="Establish the cause — product, packing, or expectation — before any commercial conversation.",
    ),
    "grow_category": Action(
        key="grow_category",
        title="Category expansion offer",
        channel="Account manager, with a targeted range",
        priority="Standard",
        cost="Account manager",
        rationale="A stable, growing account buying from a narrow range. The modelled likelihood of taking a new category is high, and there is room to widen the relationship.",
        guardrail="Lead with the specific category the model indicates, not the full catalogue. A general offer to a specific opportunity wastes both.",
    ),
    "grow_volume": Action(
        key="grow_volume",
        title="Volume and terms conversation",
        channel="Account manager",
        priority="Standard",
        cost="Account manager",
        rationale="Ordering is trending up against this account's own run-rate. Worth a conversation about volume terms while the direction is favourable.",
        guardrail="Any change to terms is a commercial decision made by a person, with margin in front of them. This is a prompt to have the conversation, nothing more.",
    ),
    "seasonal_prompt": Action(
        key="seasonal_prompt",
        title="Seasonal range prompt",
        channel="Email, in-catalogue",
        priority="Standard",
        cost="Automated",
        rationale="An active account that has never bought the seasonal range, in a business where seasonal stock is a large share of the year.",
        guardrail="Timed to the buying season, not to the month the model happened to run.",
    ),
    "nurture": Action(
        key="nurture",
        title="Onboarding sequence",
        channel="Email",
        priority="Standard",
        cost="Automated",
        rationale="Too new to have an ordering rhythm to judge. What happens in the next two quarters sets the shape of the relationship.",
        guardrail="Do not score a new account as low value. It has no history, which is not the same as a poor one.",
    ),
    "self_serve": Action(
        key="self_serve",
        title="Self-service and catalogue only",
        channel="Automated",
        priority="Low",
        cost="Automated",
        rationale="Limited value and limited engagement. Serving this account well through automated channels is the correct outcome, not a failure.",
        guardrail="Review on the next cycle. An account's circumstances change faster than its segment label.",
    ),
    "maintain": Action(
        key="maintain",
        title="No action this period",
        channel="—",
        priority="Low",
        cost="Automated",
        rationale="Ordering on rhythm, no risk signal, no clear headroom. Contacting without a reason spends goodwill that is needed later.",
        guardrail="Choosing not to contact is a decision, and is recorded as one.",
    ),
}


def _confidence(row: pd.Series) -> str:
    """How much weight the recommendation deserves.

    Driven by how much trading history exists, not by how extreme the score is.
    A decisive-looking score on two orders is the case to treat most carefully.
    """
    active = float(row.get("active_months", 0) or 0)
    orders = float(row.get("invoices", 0) or 0)
    if active >= 6 and orders >= 8:
        return "High"
    if active >= 3 and orders >= 3:
        return "Moderate"
    return "Low — limited trading history"


def recommend_retail(
    row: pd.Series,
    model_reasons: list[dict] | None = None,
    thresholds: dict[str, float] | None = None,
) -> Recommendation:
    """Decide the suggested action for one account."""
    thresholds = thresholds or {}

    lapse_p = float(row.get("lapse_risk", 0.0) or 0.0)
    growth_p = float(row.get("growth_propensity", 0.0) or 0.0)
    expand_p = float(row.get("expansion_propensity", 0.0) or 0.0)
    opportunity = float(row.get("opportunity_score", 0.0) or 0.0)

    lapse_flagged = lapse_p >= thresholds.get("lapse", 0.5)
    growth_flagged = growth_p >= thresholds.get("growth", 0.5)
    expand_flagged = expand_p >= thresholds.get("expansion", 0.5)

    revenue = float(row.get("revenue", 0) or 0)
    return_rate = float(row.get("return_rate", 0) or 0)
    overdue = float(row.get("cadence_overdue", 0) or 0)
    months_silent = float(row.get("months_since_last_order", 0) or 0)
    months_on_book = float(row.get("months_on_book", 0) or 0)
    categories = float(row.get("distinct_categories", 0) or 0)
    share_christmas = float(row.get("share_christmas", 0) or 0)
    high_value = opportunity >= 60

    def fire(key: str, *why: str) -> Recommendation:
        return Recommendation(
            customer_id=str(row.get("customer_id", "")),
            action=RETAIL_ACTIONS[key],
            conditions=list(why),
            reasons=[
                f"{r['label']}: {r['value']:,.1f} against a book average of {r['population_mean']:,.1f}"
                for r in (model_reasons or [])
            ],
            confidence=_confidence(row),
        )

    # 1. A service problem is not a sales opportunity.
    if return_rate >= 0.20 and revenue > 0:
        return fire("service_recovery", f"{return_rate:.0%} of order value returned")

    # 2. Too new to judge.
    if months_on_book <= 3:
        return fire("nurture", f"On book {months_on_book:.0f} months")

    # 3. Not scored, and the reason is said out loud.
    #
    # These accounts ordered in fewer than three months of the window, so the
    # models -- fitted only on accounts with an established rhythm -- have
    # nothing to say about them. They still get an action, because "no score"
    # is not the same as "no customer", but it is a low-cost one and the
    # recommendation states why rather than implying a judgement that was
    # never made.
    if row.get("scoreable") is False or pd.isna(row.get("lapse_risk")):
        detail = f"Active in only {float(row.get('active_months', 0) or 0):.0f} months of the window"
        if share_christmas <= 0.001 and revenue > 0:
            return fire("seasonal_prompt", "Not scored — insufficient trading history", detail)
        return fire("self_serve", "Not scored — insufficient trading history", detail)

    # 4. Retention, graded by what the account is worth and how far gone it is.
    if lapse_flagged or overdue >= 2.0:
        reason_overdue = (f"{overdue:.1f}× their usual gap between orders"
                          if overdue else f"{months_silent:.0f} months since last order")
        if months_silent >= 6:
            if high_value:
                return fire("winback_call", reason_overdue,
                            f"Relationship opportunity {opportunity:.0f}/100")
            return fire("winback_offer", reason_overdue,
                        f"Relationship opportunity {opportunity:.0f}/100")
        if high_value:
            return fire("winback_call", f"Lapse risk {lapse_p:.0%}", reason_overdue,
                        f"Relationship opportunity {opportunity:.0f}/100")
        return fire("retention_check", f"Lapse risk {lapse_p:.0%}", reason_overdue)

    # 5. Growth, on an account that is not going anywhere.
    if expand_flagged and categories <= 6:
        return fire("grow_category", f"Category expansion propensity {expand_p:.0%}",
                    f"Buys from {categories:.0f} of 12 categories")
    if growth_flagged:
        return fire("grow_volume", f"Growth propensity {growth_p:.0%}",
                    f"Ordering on rhythm, lapse risk {lapse_p:.0%}")
    if share_christmas <= 0.001 and revenue > 0:
        return fire("seasonal_prompt", "Has never bought the seasonal range")

    # 6. Serve well, cheaply.
    if opportunity < 30:
        return fire("self_serve", f"Relationship opportunity {opportunity:.0f}/100")

    return fire("maintain", f"Lapse risk {lapse_p:.0%}",
                f"Relationship opportunity {opportunity:.0f}/100")


def recommend_batch_retail(
    scored: pd.DataFrame, thresholds: dict[str, float] | None = None
) -> pd.DataFrame:
    records = [recommend_retail(row, thresholds=thresholds).as_dict()
               for _, row in scored.iterrows()]
    out = pd.DataFrame(records)
    out["conditions"] = out["conditions"].apply(lambda c: " · ".join(c))
    return out.drop(columns=["reasons"])

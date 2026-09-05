"""The decision engine: the rules that turn evidence into a suggested action."""

from __future__ import annotations

import pandas as pd
import pytest

from customer_intelligence.decision_engine import ACTION_CATALOGUE, recommend
from customer_intelligence.decision_engine.engine import Recommendation


def customer(**overrides) -> pd.Series:
    """A stable, unremarkable customer. Tests vary one thing at a time."""
    base = {
        "customer_id": "C000001", "churn_probability": 0.05,
        "opportunity_score": 50.0, "investment_propensity": 0.05,
        "lending_propensity": 0.05, "unresolved_interactions": 0,
        "ever_in_arrears": 0, "tenure_months": 60, "active_months": 11,
        "product_count": 3, "digital_share": 0.3, "avg_monthly_balance": 40_000,
    }
    base.update(overrides)
    return pd.Series(base)


def test_returns_a_recommendation_for_any_customer():
    rec = recommend(customer())
    assert isinstance(rec, Recommendation)
    assert rec.action.key in ACTION_CATALOGUE
    assert rec.conditions


def test_arrears_outranks_everything():
    """No customer in arrears is ever sold anything, however attractive."""
    rec = recommend(customer(
        ever_in_arrears=1, opportunity_score=95.0, investment_propensity=0.9,
        lending_propensity=0.9, churn_probability=0.01,
    ))
    assert rec.action.key == "arrears_support"


def test_unresolved_service_failure_outranks_commercial_contact():
    rec = recommend(customer(unresolved_interactions=3, investment_propensity=0.9,
                             opportunity_score=90.0))
    assert rec.action.key in ("service_recovery", "retention_service")


def test_high_risk_high_value_goes_to_a_human():
    rec = recommend(customer(churn_probability=0.6, opportunity_score=85.0))
    assert rec.action.key == "retention_rm"
    assert rec.action.cost == "Relationship manager"


def test_high_risk_low_value_goes_to_an_automated_channel():
    """Spending a relationship manager on a small at-risk relationship is the
    expensive mistake this rule exists to prevent."""
    rec = recommend(customer(churn_probability=0.6, opportunity_score=20.0))
    assert rec.action.key == "retention_digital"
    assert rec.action.cost == "Automated"


def test_growth_only_fires_on_a_stable_relationship():
    rec = recommend(customer(investment_propensity=0.5, opportunity_score=85.0,
                             churn_probability=0.02))
    assert rec.action.key == "grow_investment"

    at_risk = recommend(customer(investment_propensity=0.5, opportunity_score=85.0,
                                 churn_probability=0.6))
    assert at_risk.action.key != "grow_investment"


def test_early_tenure_inactive_customer_is_activated_not_sold_to():
    rec = recommend(customer(tenure_months=6, active_months=2))
    assert rec.action.key == "onboard_activate"


def test_stable_unremarkable_customer_gets_no_action():
    """Doing nothing must be a reachable outcome. An engine that always finds
    something to do is generating contact, not insight."""
    rec = recommend(customer(digital_share=0.1, product_count=5))
    assert rec.action.key == "maintain"


def test_confidence_reflects_evidence_not_score_magnitude():
    thin = recommend(customer(active_months=2, product_count=1, tenure_months=48))
    thick = recommend(customer(active_months=12, product_count=4))
    assert thin.confidence.startswith("Low")
    assert thick.confidence == "High"


def test_every_catalogue_action_declares_its_cost_and_priority():
    for key, action in ACTION_CATALOGUE.items():
        assert action.priority in {"Urgent", "High", "Standard", "Low"}
        assert action.cost in {"Relationship manager", "Contact centre", "Automated"}
        assert action.rationale, f"{key} has no rationale"


def test_commercial_actions_carry_a_guardrail():
    """Anything that leads to a product conversation must state its constraint."""
    for key in ["grow_investment", "grow_lending", "arrears_support",
                "retention_rm", "retention_service"]:
        assert ACTION_CATALOGUE[key].guardrail, f"{key} has no stated guardrail"


def test_recommendations_are_phrased_as_suggestions():
    """Language matters here: nothing may read as an instruction to approve."""
    forbidden = ["approve", "must be granted", "automatically", "guaranteed"]
    for action in ACTION_CATALOGUE.values():
        text = f"{action.title} {action.rationale} {action.guardrail}".lower()
        for word in forbidden:
            if word == "automatically":
                continue
            assert word not in text, f"'{word}' appears in {action.key}"


def test_handles_missing_fields_without_crashing():
    """Scoring artefacts carry nulls where a model declined to score."""
    sparse = pd.Series({"customer_id": "C000002", "churn_probability": 0.3,
                        "opportunity_score": 55.0})
    rec = recommend(sparse)
    assert rec.action.key in ACTION_CATALOGUE


def test_batch_matches_row_by_row(train_360):
    from customer_intelligence.decision_engine import recommend_batch

    sample = train_360.head(60).copy()
    sample["churn_probability"] = 0.2
    sample["opportunity_score"] = 55.0
    sample["investment_propensity"] = 0.1
    sample["lending_propensity"] = 0.1

    batch = recommend_batch(sample)
    assert len(batch) == len(sample)
    assert batch["action"].notna().all()
    for i in range(0, len(sample), 17):
        assert batch.iloc[i]["action"] == recommend(sample.iloc[i]).action.title

"""Leakage: the failure mode that makes every other number meaningless.

A churn model that has seen the outcome window scores beautifully and is worth
nothing. These tests check the property structurally -- that the feature window
bounds what the SQL can touch -- rather than inferring it from a suspicious AUC.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from customer_intelligence.analytics import build_customer_360
from customer_intelligence.analytics.features import (
    CATEGORICAL_FEATURES, EXCLUDED_FEATURES, NUMERIC_FEATURES, model_features,
)
from customer_intelligence.config import (
    SCORE_FEATURE_MONTHS, SQL_DIR, TRAIN_FEATURE_MONTHS, TRAIN_OUTCOME_MONTHS,
)

OUTCOME_START = min(TRAIN_OUTCOME_MONTHS)


def test_training_and_outcome_windows_do_not_overlap():
    assert max(TRAIN_FEATURE_MONTHS) < min(TRAIN_OUTCOME_MONTHS)


def test_scoring_window_ends_at_the_panel_edge():
    """Scoring uses the most recent months; its outcome is the unobserved future."""
    assert max(SCORE_FEATURE_MONTHS) > max(TRAIN_OUTCOME_MONTHS) - 1
    assert len(SCORE_FEATURE_MONTHS) == len(TRAIN_FEATURE_MONTHS)


def test_customer_360_never_reads_outside_its_window(analytical):
    """Build the 360 over months 1–12 and confirm nothing from 13–15 leaked in.

    Done by comparing against a second build over months 1–15: if the 12-month
    view already contained the extra activity, the two would be identical.
    """
    narrow = build_customer_360(analytical, 1, 12).set_index("customer_id")
    wide = build_customer_360(analytical, 1, 15).set_index("customer_id")

    common = narrow.index.intersection(wide.index)
    assert len(common) > 0

    # Totals over a longer window must be at least as large, and strictly larger
    # somewhere -- otherwise the window bound is not doing anything.
    for col in ["transaction_count", "total_logins", "active_months"]:
        assert (wide.loc[common, col] >= narrow.loc[common, col]).all(), (
            f"{col} fell when the window widened — the window bound is inverted"
        )
        assert (wide.loc[common, col] > narrow.loc[common, col]).any(), (
            f"{col} is identical across two different windows — the window "
            "parameter is being ignored, which means the 12-month view may "
            "already contain outcome-window activity"
        )


def test_product_holdings_respect_the_window(analytical):
    """A product taken in the outcome window must not appear as held at month 12."""
    narrow = build_customer_360(analytical, 1, 12).set_index("customer_id")
    wide = build_customer_360(analytical, 1, 15).set_index("customer_id")
    common = narrow.index.intersection(wide.index)
    assert (wide.loc[common, "product_count"] >= narrow.loc[common, "product_count"]).all()
    assert (wide.loc[common, "product_count"] > narrow.loc[common, "product_count"]).any()


def test_no_outcome_column_is_a_model_feature():
    """The targets and their eligibility flags must never enter the feature set."""
    forbidden = {"churned", "took_investment", "took_lending",
                 "churn_eligible", "investment_eligible", "lending_eligible"}
    assert not forbidden & set(NUMERIC_FEATURES)
    assert not forbidden & set(CATEGORICAL_FEATURES)


def test_propensity_models_cannot_see_their_own_target_holding():
    """Predicting investment uptake while being told they hold one is circular."""
    inv_numeric, _ = model_features("took_investment")
    assert "has_investment" not in inv_numeric

    lend_numeric, _ = model_features("took_lending")
    for excluded in ("has_lending", "loan_count", "loan_exposure"):
        assert excluded not in lend_numeric


def test_declared_segment_is_excluded_from_features():
    """It is a rule over the same attributes; including it makes the model circular."""
    assert "declared_segment" in EXCLUDED_FEATURES
    assert "declared_segment" not in NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_customer_360_sql_has_no_unbounded_month_literal():
    """Guard the SQL itself: a hard-coded month past the window is leakage.

    A future edit that reaches into month 13 would not fail any numeric test
    until the AUC quietly became implausible, so the rule is enforced on the text.
    """
    sql = (SQL_DIR / "03_customer_360.sql").read_text()
    body = re.sub(r"--[^\n]*", "", sql)          # strip comments
    offenders = [
        m.group(0) for m in re.finditer(r"month_index\s*(?:>=|>|=)\s*(\d+)", body)
        if int(m.group(1)) >= OUTCOME_START
    ]
    assert not offenders, (
        "03_customer_360.sql references months inside the outcome window: "
        f"{offenders}. Features must be bounded by analysis_window only."
    )


def test_outcomes_are_derived_only_from_the_outcome_window(outcomes, analytical):
    """Shifting the outcome window must change the labels; otherwise it is ignored."""
    from customer_intelligence.analytics import build_outcomes

    shifted = build_outcomes(analytical, tuple(range(1, 10)), tuple(range(10, 13)))
    merged = outcomes.merge(shifted, on="customer_id", suffixes=("_a", "_b"))
    assert (merged["churned_a"] != merged["churned_b"]).any(), (
        "the churn label is identical under two different outcome windows — "
        "it is not being derived from the window it claims to use"
    )


def test_gender_is_not_a_model_feature():
    """A protected attribute must not drive commercial targeting.

    Gender is present in the source data and shown in the customer profile, but
    it is not an input to any model. If this test fails, a model somewhere has
    begun allocating retention effort or product offers by gender.
    """
    assert "gender" not in NUMERIC_FEATURES
    assert "gender" not in CATEGORICAL_FEATURES
    assert "gender" in EXCLUDED_FEATURES

    for target in ("churned", "took_investment", "took_lending"):
        numeric, categorical = model_features(target)
        assert "gender" not in numeric + categorical, (
            f"the {target} model receives gender as an input"
        )

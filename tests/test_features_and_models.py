"""Feature generation, model inputs, and scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_intelligence.analytics import models as M
from customer_intelligence.analytics.features import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, label, model_features, prepare,
)
from customer_intelligence.analytics.opportunity import (
    opportunity_components, opportunity_score, risk_opportunity_quadrant,
)


def test_customer_360_has_one_row_per_customer(train_360):
    assert train_360["customer_id"].is_unique
    assert train_360["customer_id"].notna().all()


def test_all_declared_features_exist_in_the_360(train_360):
    missing = [f for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES
               if f not in train_360.columns]
    assert not missing, f"features declared but not produced by the SQL: {missing}"


def test_no_feature_is_entirely_null(train_360):
    for f in NUMERIC_FEATURES:
        assert train_360[f].notna().any(), f"{f} is null for every customer"


def test_counts_and_values_are_non_negative(train_360):
    for f in ["transaction_count", "transaction_value", "total_logins",
              "active_months", "product_count", "account_count", "complaints",
              "loan_exposure", "avg_monthly_balance"]:
        assert (train_360[f].fillna(0) >= 0).all(), f"{f} contains negative values"


def test_active_months_cannot_exceed_the_window(train_360):
    assert train_360["active_months"].max() <= 12


def test_digital_share_is_a_proportion(train_360):
    share = train_360["digital_share"].dropna()
    assert share.between(0, 1).all()


def test_trends_default_to_one_when_the_base_is_zero(train_360):
    """A customer with no early activity gets a neutral 1.0, never a null or an
    infinity that would silently become the median at imputation."""
    for col in ["balance_trend", "transaction_trend", "digital_trend"]:
        assert train_360[col].notna().all()
        assert np.isfinite(train_360[col]).all()


def test_prepare_selects_only_declared_columns(train_360):
    numeric, categorical = model_features("churned")
    X = prepare(train_360, [c for c in numeric if c in train_360.columns],
                [c for c in categorical if c in train_360.columns])
    assert "customer_id" not in X.columns
    assert "declared_segment" not in X.columns


def test_labels_are_human_readable():
    assert label("avg_monthly_balance") == "Average monthly balance"
    assert label("region_Coastal").startswith("Region:")


@pytest.fixture(scope="module")
def trained(train_360, outcomes):
    merged = train_360.merge(outcomes, on="customer_id")
    subset = merged[merged["churn_eligible"] == 1]
    return M.train_model(subset, subset["churned"], "churned", "Attrition risk"), subset


def test_model_beats_random(trained):
    result, _ = trained
    assert result.metrics["roc_auc"] > 0.6, "the model is no better than guessing"
    assert result.metrics["lift"] > 1.0, "the model does not beat the base rate"


def test_threshold_flags_the_intended_share(trained):
    result, _ = trained
    assert 0.05 <= result.metrics["flagged_share"] <= 0.16


def test_scores_are_probabilities(trained, train_360):
    result, _ = trained
    p = M.score(result, train_360)
    assert len(p) == len(train_360)
    assert ((p >= 0) & (p <= 1)).all()


def test_scoring_is_deterministic(trained, train_360):
    result, _ = trained
    assert np.allclose(M.score(result, train_360), M.score(result, train_360))


def test_scoring_tolerates_an_unseen_category(trained, train_360):
    """A region that did not exist at training time must not crash scoring."""
    result, _ = trained
    altered = train_360.copy()
    altered.loc[altered.index[:5], "region"] = "Newly opened region"
    p = M.score(result, altered)
    assert np.isfinite(p).all()


def test_top_reasons_are_ordered_by_magnitude(trained):
    result, subset = trained
    reasons = M.top_reasons(result, subset.iloc[0], subset, n=5)
    magnitudes = [abs(r["contribution"]) for r in reasons]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_calibration_table_is_well_formed(trained):
    result, _ = trained
    cal = result.calibration
    assert set(["predicted", "observed", "n"]).issubset(cal.columns)
    assert cal["predicted"].between(0, 1).all()
    assert cal["observed"].between(0, 1).all()
    assert cal["n"].sum() == result.n_test


def test_opportunity_score_is_bounded(train_360):
    n = len(train_360)
    rng = np.random.default_rng(0)
    components = opportunity_components(
        train_360, rng.random(n), rng.random(n), rng.random(n))
    score = opportunity_score(components)
    assert score.between(0, 100).all()
    assert components.apply(lambda c: c.between(0, 100).all()).all()


def test_quadrant_assignment_is_exhaustive(train_360):
    n = len(train_360)
    rng = np.random.default_rng(1)
    score = pd.Series(rng.uniform(0, 100, n), index=train_360.index)
    risk = pd.Series(rng.uniform(0, 1, n), index=train_360.index)
    q = risk_opportunity_quadrant(score, risk)
    assert q.notna().all()
    assert set(q.unique()) <= {"Retain", "Grow", "Monitor", "Develop"}


def test_predicted_probabilities_match_the_observed_rate(trained):
    """Calibration in the large: mean predicted must track the base rate.

    This is the test that catches class-weighting applied by reflex. Balancing
    an imbalanced target inflates probabilities toward 0.5, which leaves every
    ranking metric untouched and quietly invalidates every probability band,
    risk cut-off and calibration curve downstream.
    """
    result, _ = trained
    predicted = result.metrics["mean_predicted"]
    observed = result.metrics["observed_rate"]
    assert abs(predicted - observed) < 0.35 * observed + 0.02, (
        f"mean predicted probability {predicted:.3f} is far from the observed "
        f"rate {observed:.3f} — the model is not calibrated, so the probability "
        "bands and risk cut-offs that consume it are meaningless"
    )


def test_calibration_curve_is_monotone_enough_to_be_useful(trained):
    """Higher predicted deciles must actually carry higher observed rates."""
    cal = trained[0].calibration
    assert cal["observed"].iloc[-1] > cal["observed"].iloc[0], (
        "the top predicted decile does not have a higher observed rate than the "
        "bottom — the model has no usable signal"
    )

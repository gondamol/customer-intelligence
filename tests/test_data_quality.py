"""The quality framework, graded against the defects that were injected."""

from __future__ import annotations

import pytest

from customer_intelligence.data_quality import CHECKS, run_checks
from customer_intelligence.data_quality.report import (
    build_quality_report, dimension_scores, quality_score,
)


def test_every_check_executes(con):
    results = run_checks(con)
    assert len(results) == len(CHECKS)
    assert results["failing"].notna().all()
    assert (results["failing"] >= 0).all()


def test_pass_rate_is_a_proportion(con):
    results = run_checks(con)
    assert results["pass_rate"].between(0, 1).all()


def test_every_injected_defect_is_detected(con, manifest):
    """The central test: each defect type must be caught by its own check.

    Detected may exceed injected -- nulling a customer identifier orphans the
    accounts beneath it, so one defect manufactures others. Under-detection is
    the failure; over-detection is a cascade and is expected.
    """
    report = build_quality_report(con, manifest)
    missed = report["reconciliation"][~report["reconciliation"]["caught"]]
    assert missed.empty, (
        "checks failed to detect injected defects:\n"
        + missed[["table", "defect", "injected", "detected", "check"]].to_string(index=False)
    )


def test_each_check_key_is_unique():
    keys = [c.key for c in CHECKS]
    assert len(keys) == len(set(keys))


def test_manifest_keys_reference_real_checks(manifest):
    """A defect claiming detection by a non-existent check would pass silently."""
    known = {c.key for c in CHECKS}
    for entry in manifest:
        assert entry["detected_by"] in known, (
            f"{entry['defect']} names an unknown check: {entry['detected_by']}"
        )


def test_dimension_scores_are_bounded(con):
    dims = dimension_scores(run_checks(con))
    assert dims["score"].between(0, 100).all()
    assert 0 <= quality_score(run_checks(con)) <= 100


def test_customer_impact_exceeds_zero_and_is_bounded(con, manifest):
    report = build_quality_report(con, manifest)
    total = report["impact"].attrs["customers_total"]
    assert 0 < report["customers_affected"] <= total


def test_clean_data_passes_the_checks_it_should(clean_tables, tmp_path):
    """Run the suite over undamaged data: the injected-defect rules must pass.

    Without this, a check that always fires would look like a working detector.
    """
    import duckdb

    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    for name, df in clean_tables.items():
        path = tmp_path / f"{name}.parquet"
        df.to_parquet(path, index=False)
        con.execute(f"CREATE OR REPLACE VIEW {name} AS "
                    f"SELECT * FROM read_parquet('{path.as_posix()}')")

    results = run_checks(con).set_index("key")
    for key in ["customers.customer_id.missing", "customers.customer_id.duplicate",
                "customers.age.invalid", "accounts.customer_id.orphan",
                "accounts.status.invalid", "transactions.transaction_id.duplicate",
                "transactions.amount.missing", "transactions.amount.anomaly",
                "transactions.date.future", "products.product_type.inconsistent"]:
        assert results.loc[key, "failing"] == 0, (
            f"{key} fires on clean data — it is not detecting a defect, "
            "it is detecting the data"
        )

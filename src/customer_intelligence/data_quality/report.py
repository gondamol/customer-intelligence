"""Scoring and presentation of the quality results.

A single quality score is a blunt instrument, so it is built from parts that can
be inspected: each check contributes its pass rate, weighted by severity, into a
dimension score; the dimensions are then averaged with published weights. The
number is only meaningful alongside the table it comes from, which is why the
application always shows both.
"""

from __future__ import annotations

import pandas as pd

# High-severity rules dominate the dimension score; low-severity rules move it
# only slightly. Published here rather than buried in the calculation.
SEVERITY_WEIGHTS = {"high": 3.0, "medium": 2.0, "low": 1.0}

DIMENSION_WEIGHTS = {
    "Completeness": 0.20,
    "Uniqueness": 0.15,
    "Validity": 0.20,
    "Consistency": 0.10,
    "Integrity": 0.20,
    "Timeliness": 0.10,
    "Accuracy": 0.05,
}

DIMENSION_DEFINITIONS = {
    "Completeness": "Are the values that should be present actually present?",
    "Uniqueness": "Does each real-world thing appear exactly once?",
    "Validity": "Do values fall inside their permitted range or type?",
    "Consistency": "Do values agree with the controlled vocabulary and with each other?",
    "Integrity": "Do foreign keys resolve to a real parent record?",
    "Timeliness": "Do dates fall inside the reporting window?",
    "Accuracy": "Are values plausible against the distribution they belong to?",
}


def dimension_scores(results: pd.DataFrame) -> pd.DataFrame:
    """Severity-weighted pass rate per dimension."""
    df = results.copy()
    df["weight"] = df["severity"].map(SEVERITY_WEIGHTS)
    grouped = df.groupby("dimension", observed=True).apply(
        lambda g: pd.Series({
            "score": 100.0 * (g["pass_rate"] * g["weight"]).sum() / g["weight"].sum(),
            "checks": len(g),
            "failed_checks": int((g["failing"] > 0).sum()),
            "failing_rows": int(g["failing"].sum()),
        }),
        include_groups=False,
    ).reset_index()
    grouped["definition"] = grouped["dimension"].map(DIMENSION_DEFINITIONS)
    grouped["dimension_weight"] = grouped["dimension"].map(DIMENSION_WEIGHTS)
    return grouped.sort_values("score", ignore_index=True)


def quality_score(results: pd.DataFrame) -> float:
    """Overall score, 0–100, as the weighted mean of the dimension scores."""
    dims = dimension_scores(results)
    weights = dims["dimension"].map(DIMENSION_WEIGHTS).fillna(0.0)
    if weights.sum() == 0:
        return float(dims["score"].mean())
    return float((dims["score"] * weights).sum() / weights.sum())


def customer_impact(con) -> pd.DataFrame:
    """How many distinct customers each defect actually touches.

    A row-level pass rate of 99% sounds like nothing is wrong. But defects are
    not spread evenly across customers, and the question an owner asks is "how
    many of my customers are affected?" -- which is a different, larger number.
    """
    from .checks import CHECKS

    rows, affected_sets = [], []
    for check in CHECKS:
        if not check.impact_sql:
            continue
        ids = con.execute(check.impact_sql).df()["customer_id"].dropna()
        affected_sets.append(set(ids))
        rows.append({
            "key": check.key, "dimension": check.dimension,
            "rule": check.rule, "customers_affected": len(ids),
        })

    union = set().union(*affected_sets) if affected_sets else set()
    total = int(con.execute("SELECT count(DISTINCT customer_id) FROM customers").fetchone()[0])
    df = pd.DataFrame(rows).sort_values("customers_affected", ascending=False, ignore_index=True)
    df.attrs["customers_any_defect"] = len(union)
    df.attrs["customers_total"] = total
    df.attrs["share_any_defect"] = (len(union) / total * 100) if total else 0.0
    return df


def build_quality_report(con, manifest: list[dict] | None = None) -> dict:
    """Run the checks and assemble everything the application and tests need."""
    from .checks import run_checks

    results = run_checks(con)
    impact = customer_impact(con)
    report = {
        "results": results,
        "dimensions": dimension_scores(results),
        "score": quality_score(results),
        "checks_run": len(results),
        "checks_passed": int((results["failing"] == 0).sum()),
        "checks_failed": int((results["failing"] > 0).sum()),
        "rows_affected": int(results["failing"].sum()),
        "impact": impact,
        "customers_affected": impact.attrs.get("customers_any_defect", 0),
        "customers_affected_pct": impact.attrs.get("share_any_defect", 0.0),
    }
    if manifest is not None:
        report["reconciliation"] = reconcile(results, manifest)
    return report


def reconcile(results: pd.DataFrame, manifest: list[dict]) -> pd.DataFrame:
    """Compare what was injected against what the checks caught.

    This is the part that makes the quality framework testable. Where detected
    exceeds injected, the rule is also catching defects that arose naturally in
    the generated population -- which is noted rather than treated as an error.
    """
    detected = results.set_index("key")["failing"]
    rows = []
    for entry in manifest:
        key = entry["detected_by"]
        found = int(detected.get(key, 0))
        rows.append({
            "table": entry["table"],
            "defect": entry["defect"],
            "dimension": entry["dimension"].title(),
            "injected": entry["injected_rows"],
            "detected": found,
            "caught": found >= entry["injected_rows"],
            "check": key,
        })
    return pd.DataFrame(rows)

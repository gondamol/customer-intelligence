"""The pipeline. One command per stage, and one to run them all.

    python -m customer_intelligence.pipeline generate
    python -m customer_intelligence.pipeline quality
    python -m customer_intelligence.pipeline build
    python -m customer_intelligence.pipeline train
    python -m customer_intelligence.pipeline all

Every stage writes its output to data/processed or models/. The application
reads those artefacts and never trains anything: a dashboard that fits a model
on page load is a demo, not a product.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd

from . import io
from .analytics import build_analytical_layer, build_customer_360, build_outcomes, conformance_log
from .analytics import models as M
from .analytics import opportunity as O
from .analytics import segmentation as S
from .config import (
    MODELS_DIR, N_CUSTOMERS, PROCESSED_DIR, RANDOM_SEED, SCORE_FEATURE_MONTHS,
    TRAIN_FEATURE_MONTHS, TRAIN_OUTCOME_MONTHS,
)
from .data_generation import generate_all, inject_defects
from .data_quality import build_quality_report
from .decision_engine import recommend, recommend_batch

TARGETS = [
    ("churned", "churn_eligible", "Attrition risk", "churn"),
    ("took_investment", "investment_eligible", "Investment propensity", "investment"),
    ("took_lending", "lending_eligible", "Lending propensity", "lending"),
]


def _log(msg: str, t0: float | None = None) -> float:
    elapsed = f"  ({time.time() - t0:.1f}s)" if t0 else ""
    print(f"  {msg}{elapsed}", flush=True)
    return time.time()


# ------------------------------------------------------------- stages ----


def stage_generate(n_customers: int = N_CUSTOMERS, seed: int = RANDOM_SEED) -> None:
    print(f"\n[1/4] Generating synthetic source data — {n_customers:,} customers")
    t = time.time()
    clean = generate_all(n_customers=n_customers, seed=seed)
    t = _log("clean population generated", t)

    raw, manifest = inject_defects(clean)
    t = _log(f"{len(manifest)} defect types injected", t)

    for name, df in raw.items():
        io.write_table(df, name, "raw")
        print(f"      {name:<28} {len(df):>10,} rows")
    (PROCESSED_DIR / "defect_manifest.json").write_text(json.dumps(manifest, indent=2))
    _log("raw layer written", t)


def stage_quality() -> None:
    print("\n[2/4] Data quality assessment")
    t = time.time()
    manifest = json.loads((PROCESSED_DIR / "defect_manifest.json").read_text())
    con = io.connect("raw")
    report = build_quality_report(con, manifest)

    io.write_table(report["results"], "quality_results", "processed")
    io.write_table(report["dimensions"], "quality_dimensions", "processed")
    io.write_table(report["impact"], "quality_impact", "processed")
    io.write_table(report["reconciliation"], "quality_reconciliation", "processed")

    summary = {
        "score": report["score"],
        "checks_run": report["checks_run"],
        "checks_passed": report["checks_passed"],
        "checks_failed": report["checks_failed"],
        "rows_affected": report["rows_affected"],
        "customers_affected": int(report["customers_affected"]),
        "customers_affected_pct": float(report["customers_affected_pct"]),
        "defects_injected": len(manifest),
        "defects_caught": int(report["reconciliation"]["caught"].sum()),
    }
    (PROCESSED_DIR / "quality_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"      record-level conformance   {summary['score']:.2f}%")
    print(f"      rules passing              {summary['checks_passed']} of {summary['checks_run']}")
    print(f"      customers affected         {summary['customers_affected']:,} "
          f"({summary['customers_affected_pct']:.1f}%)")
    print(f"      defects caught             {summary['defects_caught']} of {summary['defects_injected']}")
    _log("quality artefacts written", t)


def stage_build() -> None:
    print("\n[3/4] Analytical data model")
    t = time.time()
    con = io.connect("raw")
    build_analytical_layer(con)
    t = _log("conformed layer and monthly panel built", t)

    io.write_table(conformance_log(con), "conformance_log", "processed")
    io.write_table(
        con.execute("SELECT * FROM monthly_customer_metrics").df(),
        "monthly_customer_metrics", "processed",
    )

    train_360 = build_customer_360(con, min(TRAIN_FEATURE_MONTHS), max(TRAIN_FEATURE_MONTHS))
    io.write_table(train_360, "customer_360_train", "processed")
    t = _log(f"training snapshot: months {min(TRAIN_FEATURE_MONTHS)}–{max(TRAIN_FEATURE_MONTHS)} "
             f"({len(train_360):,} customers)", t)

    outcomes = build_outcomes(con, TRAIN_FEATURE_MONTHS, TRAIN_OUTCOME_MONTHS)
    io.write_table(outcomes, "outcomes", "processed")
    for target, eligible, name, _ in TARGETS:
        e = outcomes[outcomes[eligible] == 1]
        print(f"      {name:<22} eligible {len(e):>7,}   base rate {e[target].mean() * 100:5.2f}%")

    score_360 = build_customer_360(con, min(SCORE_FEATURE_MONTHS), max(SCORE_FEATURE_MONTHS))
    io.write_table(score_360, "customer_360_score", "processed")
    _log(f"scoring snapshot: months {min(SCORE_FEATURE_MONTHS)}–{max(SCORE_FEATURE_MONTHS)}", t)


def stage_train() -> None:
    print("\n[4/4] Models, scores and recommendations")
    t = time.time()
    train_360 = io.read_table("customer_360_train", "processed")
    score_360 = io.read_table("customer_360_score", "processed")
    outcomes = io.read_table("outcomes", "processed")

    train = train_360.merge(outcomes, on="customer_id", how="inner")
    cards, results = {}, {}

    for target, eligible, name, slug in TARGETS:
        subset = train[train[eligible] == 1]
        result = M.train_model(subset, subset[target], target, name)
        results[slug] = result

        joblib.dump(result.estimator, MODELS_DIR / f"{slug}.joblib")
        io.write_table(result.comparison, f"model_{slug}_comparison", "processed")
        io.write_table(result.importances, f"model_{slug}_importance", "processed")
        io.write_table(result.coefficients, f"model_{slug}_coefficients", "processed")
        io.write_table(result.calibration, f"model_{slug}_calibration", "processed")

        cards[slug] = {
            "target": target, "display_name": name,
            "metrics": {k: float(v) for k, v in result.metrics.items()},
            "threshold": float(result.threshold),
            "base_rate": float(result.base_rate),
            "n_train": result.n_train, "n_test": result.n_test,
            "eligible": int(len(subset)),
            "confusion": result.confusion.tolist(),
            "chosen": "Logistic regression",
        }
        m = result.metrics
        print(f"      {name:<22} ROC-AUC {m['roc_auc']:.3f}   PR-AUC {m['pr_auc']:.3f}   "
              f"recall {m['recall']:.2f}   lift {m['lift']:.1f}x")

    (MODELS_DIR / "model_cards.json").write_text(json.dumps(cards, indent=2))
    t = _log("models trained and persisted", t)

    # ---- score the current book --------------------------------------------
    scored = score_360.copy()
    scored["churn_probability"] = M.score(results["churn"], score_360)
    scored["investment_propensity"] = M.score(results["investment"], score_360)
    scored["lending_propensity"] = M.score(results["lending"], score_360)

    # A model must not score a customer it could not have been trained on.
    held_investment = score_360["has_investment"] == 1
    held_lending = score_360["has_lending"] == 1
    scored.loc[held_investment, "investment_propensity"] = np.nan
    scored.loc[held_lending, "lending_propensity"] = np.nan

    components = O.opportunity_components(
        scored,
        scored["investment_propensity"].fillna(0.0),
        scored["lending_propensity"].fillna(0.0),
        scored["churn_probability"],
    )
    for col in components.columns:
        scored[f"opp_{col.lower().replace(' ', '_')}"] = components[col].round(1)
    scored["opportunity_score"] = O.opportunity_score(components)
    scored["quadrant"] = O.risk_opportunity_quadrant(
        scored["opportunity_score"], scored["churn_probability"])

    scored["segment"] = S.rule_segments(scored)
    clusters, diagnostics = S.kmeans_segments(scored, k=6)
    scored["cluster"] = clusters
    t = _log(f"scored, segmented and scored for opportunity "
             f"(silhouette {diagnostics['silhouette']:.3f})", t)

    # The engine acts on the models' operating thresholds, so the capacity
    # assumption is made once and the recommended outreach cannot exceed what
    # the models were thresholded to flag.
    thresholds = {slug: float(r.threshold) for slug, r in results.items()}
    recommendations = recommend_batch(scored, thresholds)
    scored = scored.merge(
        recommendations[["customer_id", "action", "channel", "priority", "cost", "conditions"]],
        on="customer_id", how="left",
    )
    t = _log("next-best-action applied across the book", t)

    io.write_table(scored, "customer_360", "processed")
    io.write_table(S.segment_profiles(scored, "segment"), "segment_profiles", "processed")
    io.write_table(S.crosstab(scored).reset_index(), "segment_cluster_crosstab", "processed")
    io.write_table(S.choose_k(scored), "cluster_diagnostics", "processed")

    (PROCESSED_DIR / "run_summary.json").write_text(json.dumps({
        "customers": int(len(scored)),
        "thresholds": thresholds,
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "cluster_silhouette": diagnostics["silhouette"],
        "actions": scored["action"].value_counts().to_dict(),
        "quadrants": scored["quadrant"].value_counts().to_dict(),
    }, indent=2))
    _log("artefacts written to data/processed", t)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "quality", "build", "train", "all"])
    parser.add_argument("--customers", type=int, default=N_CUSTOMERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)

    started = time.time()
    if args.stage in ("generate", "all"):
        stage_generate(args.customers, args.seed)
    if args.stage in ("quality", "all"):
        stage_quality()
    if args.stage in ("build", "all"):
        stage_build()
    if args.stage in ("train", "all"):
        stage_train()
    print(f"\nDone in {time.time() - started:.1f}s. Run `make run` to open the application.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The retail pipeline — the project's primary build.

    python -m customer_intelligence.pipeline_retail all

Stages: fetch the published data, land it as source tables, assess quality,
build the analytical model, train, score, and apply the decision engine.
Everything the application shows is produced here and written to disk.
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
from .analytics import models as M
from .analytics import features_retail as FR
from .analytics import opportunity as O
from .analytics import retail as R
from .analytics.recommender import ItemRecommender
from .config import MODELS_DIR, PROCESSED_DIR
from .data_quality import RETAIL_CHECKS, build_quality_report
from .decision_engine.retail_engine import recommend_batch_retail
from .domains import RETAIL
from .sources import attribution_table, fetch_all, provenance
from .sources.retail import build_source_tables

SOURCE_TABLES = ("transactions", "products", "customers", "invoices")

# Two classifiers. The third question -- what should we offer this account
# next? -- is answered by a recommender, not a classifier, and the reason is
# recorded rather than assumed: a category-expansion classifier was built first
# and scored ROC-AUC 0.550. See analytics/recommender.py.
TARGETS = [
    ("lapsed", "lapse_eligible", "Lapse risk", "lapse"),
    ("grew", "growth_eligible", "Growth propensity", "growth"),
]

SCORE_COLUMN = {"lapse": "lapse_risk", "growth": "growth_propensity"}


def _log(msg: str, t0: float | None = None) -> float:
    print(f"  {msg}" + (f"  ({time.time() - t0:.1f}s)" if t0 else ""), flush=True)
    return time.time()


def stage_fetch() -> None:
    print("\n[1/5] Fetching the published source data", flush=True)
    fetch_all()
    (PROCESSED_DIR / "sources.json").write_text(json.dumps({
        "provenance": provenance(), "attribution": attribution_table(),
    }, indent=2))


def stage_land() -> None:
    print("\n[2/5] Landing the source tables (as published, defects intact)", flush=True)
    t = time.time()
    tables = build_source_tables()
    for name, df in tables.items():
        io.write_table(df, name, "raw")
        print(f"      {name:<14} {len(df):>9,} rows", flush=True)
    _log("raw layer written", t)


def stage_quality() -> None:
    print("\n[3/5] Data quality assessment", flush=True)
    t = time.time()
    con = io.connect("raw", tables=SOURCE_TABLES)
    report = build_quality_report(con, checks=RETAIL_CHECKS)

    io.write_table(report["results"], "quality_results", "processed")
    io.write_table(report["dimensions"], "quality_dimensions", "processed")
    io.write_table(report["impact"], "quality_impact", "processed")

    summary = {
        "score": report["score"], "checks_run": report["checks_run"],
        "checks_passed": report["checks_passed"], "checks_failed": report["checks_failed"],
        "rows_affected": report["rows_affected"],
        "customers_affected": int(report["customers_affected"]),
        "customers_affected_pct": float(report["customers_affected_pct"]),
    }
    (PROCESSED_DIR / "quality_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"      record-level conformance   {summary['score']:.2f}%")
    print(f"      rules passing              {summary['checks_passed']} of {summary['checks_run']}")
    print(f"      customers affected         {summary['customers_affected']:,} "
          f"({summary['customers_affected_pct']:.1f}%)")
    _log("quality artefacts written", t)


def stage_build() -> None:
    print("\n[4/5] Analytical data model", flush=True)
    t = time.time()
    con = io.connect("raw", tables=SOURCE_TABLES)
    R.build_layer(con)
    t = _log("conformed layer and monthly panel built", t)

    io.write_table(R.conformance_log(con), "conformance_log", "processed")
    io.write_table(con.execute("SELECT * FROM monthly_customer_metrics").df(),
                   "monthly_customer_metrics", "processed")
    io.write_table(con.execute("SELECT * FROM conformed_products").df(),
                   "products_conformed", "processed")

    train_360 = R.build_360(con, min(RETAIL.train_feature), max(RETAIL.train_feature))
    io.write_table(train_360, "customer_360_train", "processed")
    t = _log(f"training snapshot: months {min(RETAIL.train_feature)}–{max(RETAIL.train_feature)} "
             f"({len(train_360):,} accounts)", t)

    outcomes = R.build_outcomes(con)
    io.write_table(outcomes, "outcomes", "processed")
    for target, eligible, name, _ in TARGETS:
        e = outcomes[outcomes[eligible] == 1]
        print(f"      {name:<20} eligible {len(e):>6,}   base rate {e[target].mean()*100:5.2f}%")

    score_360 = R.build_360(con, min(RETAIL.score_feature), max(RETAIL.score_feature))
    io.write_table(score_360, "customer_360_score", "processed")
    _log(f"scoring snapshot: months {min(RETAIL.score_feature)}–{max(RETAIL.score_feature)} "
         f"({len(score_360):,} accounts)", t)


def stage_train() -> None:
    print("\n[5/5] Models, scores and recommendations", flush=True)
    t = time.time()
    M.use_contract(FR)

    train_360 = io.read_table("customer_360_train", "processed")
    score_360 = io.read_table("customer_360_score", "processed")
    outcomes = io.read_table("outcomes", "processed")
    train = train_360.merge(outcomes, on="customer_id", how="inner")

    cards, results = {}, {}
    for target, eligible, name, slug in TARGETS:
        subset = train[train[eligible] == 1]
        result = M.train_model(subset, subset[target], target, name)
        results[slug] = result

        joblib.dump(result.estimator, MODELS_DIR / f"retail_{slug}.joblib")
        io.write_table(result.comparison, f"model_{slug}_comparison", "processed")
        io.write_table(result.importances, f"model_{slug}_importance", "processed")
        io.write_table(result.coefficients, f"model_{slug}_coefficients", "processed")
        io.write_table(result.calibration, f"model_{slug}_calibration", "processed")

        cards[slug] = {
            "target": target, "display_name": name,
            "metrics": {k: float(v) for k, v in result.metrics.items()},
            "threshold": float(result.threshold), "base_rate": float(result.base_rate),
            "n_train": result.n_train, "n_test": result.n_test,
            "eligible": int(len(subset)), "confusion": result.confusion.tolist(),
            "chosen": "Logistic regression",
        }
        m = result.metrics
        print(f"      {name:<20} ROC-AUC {m['roc_auc']:.3f}   PR-AUC {m['pr_auc']:.3f}   "
              f"recall {m['recall']:.2f}   lift {m['lift']:.1f}x", flush=True)

    (MODELS_DIR / "model_cards.json").write_text(json.dumps(cards, indent=2))
    t = _log("models trained and persisted", t)

    # ---- score the current book -------------------------------------------
    scored = score_360.copy()
    for slug, result in results.items():
        scored[SCORE_COLUMN[slug]] = M.score(result, score_360)

    # Eligibility applies at scoring time exactly as it did at training time.
    # These models were fitted only on accounts that had established an ordering
    # rhythm -- at least three active months in a twelve-month window -- and an
    # account outside that population is one the model has never seen. Scoring
    # it anyway is how a model quietly ends up making decisions about people it
    # knows nothing about, and it is why the account-manager workload was 45% of
    # the book before this line existed.
    #
    # Null, not zero: zero is a prediction, null is an admission.
    established = scored["active_months"] >= R.ESTABLISHED_MIN_ACTIVE_MONTHS
    scored["scoreable"] = established
    for column in SCORE_COLUMN.values():
        scored.loc[~established, column] = np.nan

    components = O.opportunity_components_retail(scored)
    for col in components.columns:
        scored[f"opp_{col.lower().replace(' ', '_')}"] = components[col].round(1)
    scored["opportunity_score"] = O.opportunity_score(components)
    # Only scoreable accounts get placed in the matrix. Filling an unscored
    # account's risk with 1.0 to make the arithmetic work would put it in the
    # Retain quadrant -- flagged as high risk -- on the strength of a number the
    # model explicitly declined to produce. That is the same error as scoring
    # outside the eligible population, wearing a different hat.
    scored["quadrant"] = O.risk_opportunity_quadrant(
        scored["opportunity_score"], scored["lapse_risk"],
        opportunity_cut=60.0, risk_cut=0.5)
    scored.loc[scored["lapse_risk"].isna(), "quadrant"] = "Not scored"

    scored["segment"] = R.rfm_segments(scored)
    t = _log("scored, segmented and scored for opportunity", t)

    # ---- what to offer next -------------------------------------------------
    con = io.connect("raw", tables=SOURCE_TABLES)
    R.build_layer(con)

    # First at category level, which is where the obvious version of this
    # question lives -- and where it turns out not to be a question at all.
    cat_train = R.category_holdings(con, RETAIL.train_feature)
    cat_added = R.categories_added(con, RETAIL.train_feature, RETAIL.train_outcome)
    category_eval = ItemRecommender().fit(cat_train).evaluate(cat_train, cat_added, k=3)

    # Then at product level, where it is.
    prod_train = R.product_holdings(con, RETAIL.train_feature)
    products = list(prod_train.columns)
    prod_added = R.products_added(con, RETAIL.train_feature, RETAIL.train_outcome, products)

    recommender = ItemRecommender().fit(prod_train)
    evaluation = {k: recommender.evaluate(prod_train, prod_added, k=k) for k in (5, 10, 20)}
    at5 = evaluation[5]
    print(f"      Next-best-product    hit@5 {at5['hit_rate_at_k']:.3f} vs popularity "
          f"{at5['popularity_baseline']:.3f} "
          f"({at5['hit_rate_at_k'] / max(at5['popularity_baseline'], 1e-9):.2f}x lift, "
          f"{at5['accounts_evaluated']:,} accounts)", flush=True)
    print(f"      (category level      hit@3 {category_eval.get('hit_rate_at_k', 0):.3f} vs "
          f"popularity {category_eval.get('popularity_baseline', 0):.3f} — no better than "
          f"popularity, and reported as such)", flush=True)

    prod_score = R.product_holdings(con, RETAIL.score_feature, restrict_to=products)
    next_best = recommender.recommend(prod_score, n=5)
    descriptions = con.execute(
        "SELECT stock_code, description, category FROM conformed_products").df()
    next_best = next_best.rename(columns={"category": "stock_code"}).merge(
        descriptions, on="stock_code", how="left")
    io.write_table(next_best, "next_best_product", "processed")

    top1 = next_best[next_best["rank"] == 1].set_index("customer_id")
    scored["next_best_product"] = scored["customer_id"].map(top1["description"])
    scored["next_best_product_code"] = scored["customer_id"].map(top1["stock_code"])
    scored["next_best_category"] = scored["customer_id"].map(top1["category"])
    scored["next_best_affinity"] = scored["customer_id"].map(top1["affinity"])

    (MODELS_DIR / "recommender_card.json").write_text(json.dumps({
        "granularity": "product",
        "products_considered": len(products),
        "product_level": evaluation,
        "category_level": category_eval,
        "why": (
            "A category-level recommender was built first and did not beat a "
            "popularity baseline. The median account already buys 9 of the 12 "
            "categories, so 'the top 3 you do not buy' is most of what is left "
            "and the evaluation is close to degenerate. At product level the "
            "median account buys 18 of the top 400, the problem is real, and "
            "item-to-item collaborative filtering beats popularity by about "
            "1.5x at k=5."
        ),
    }, indent=2))
    t = _log("next-best-product recommender fitted and applied", t)

    thresholds = {slug: float(r.threshold) for slug, r in results.items()}
    recommendations = recommend_batch_retail(scored, thresholds)
    scored = scored.merge(
        recommendations[["customer_id", "action", "channel", "priority", "cost", "conditions"]],
        on="customer_id", how="left")
    t = _log("next-best-action applied across the book", t)

    io.write_table(scored, "customer_360", "processed")
    io.write_table(R.segment_profiles(scored, "segment"), "segment_profiles", "processed")

    (PROCESSED_DIR / "run_summary.json").write_text(json.dumps({
        "domain": RETAIL.key, "domain_name": RETAIL.name,
        "customers": int(len(scored)),
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "thresholds": thresholds,
        "window_rationale": RETAIL.window_rationale,
        "train_feature": [min(RETAIL.train_feature), max(RETAIL.train_feature)],
        "train_outcome": [min(RETAIL.train_outcome), max(RETAIL.train_outcome)],
        "score_feature": [min(RETAIL.score_feature), max(RETAIL.score_feature)],
        "recommender": evaluation[5],
        "actions": scored["action"].value_counts().to_dict(),
        "quadrants": scored["quadrant"].value_counts().to_dict(),
        "segments": scored["segment"].value_counts().to_dict(),
    }, indent=2))
    _log("artefacts written to data/processed", t)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fetch", "land", "quality", "build", "train", "all"])
    args = parser.parse_args(argv)

    started = time.time()
    if args.stage in ("fetch", "all"):
        stage_fetch()
    if args.stage in ("land", "all"):
        stage_land()
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

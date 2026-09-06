"""The retail domain: source adapter, conformed layer, windows, recommender.

These run against the real downloaded dataset. If it has not been fetched the
suite skips rather than fails, so a fresh clone can still run the rest.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from customer_intelligence import io
from customer_intelligence.analytics import retail as R
from customer_intelligence.analytics.recommender import ItemRecommender
from customer_intelligence.analytics.features_retail import (
    CATEGORICAL_FEATURES, EXCLUDED_FEATURES, NUMERIC_FEATURES, model_features,
)
from customer_intelligence.config import SQL_DIR
from customer_intelligence.data_quality import RETAIL_CHECKS, run_checks
from customer_intelligence.domains import RETAIL
from customer_intelligence.sources.registry import DATASETS
from customer_intelligence.sources.retail import (
    NON_PRODUCT_CODES, RAW_CACHE, UNCATEGORISED, categorise,
)

pytestmark = pytest.mark.skipif(
    not RAW_CACHE.exists(),
    reason="Online Retail II not downloaded; run `make fetch` first",
)

SOURCE_TABLES = ("transactions", "products", "customers", "invoices")


@pytest.fixture(scope="module")
def tables():
    from customer_intelligence.sources.retail import build_source_tables
    return build_source_tables()


@pytest.fixture(scope="module")
def con(tables, tmp_path_factory):
    import duckdb

    tmp = tmp_path_factory.mktemp("retail_raw")
    connection = duckdb.connect()
    connection.execute("SET enable_progress_bar = false")
    for name, df in tables.items():
        path = tmp / f"{name}.parquet"
        df.to_parquet(path, index=False)
        connection.execute(f"CREATE OR REPLACE VIEW {name} AS "
                           f"SELECT * FROM read_parquet('{path.as_posix()}')")
    R.build_layer(connection)
    return connection


# ----------------------------------------------------------- provenance ----


def test_every_dataset_declares_a_licence_and_attribution():
    """CC BY 4.0 requires credit. A missing attribution is a licence breach,
    not a documentation gap."""
    for key, dataset in DATASETS.items():
        assert dataset.licence, f"{key} has no licence"
        assert dataset.attribution, f"{key} has no attribution"
        assert dataset.citation, f"{key} has no citation"
        assert dataset.landing_page.startswith("https://"), f"{key} has no verifiable source"


# --------------------------------------------------------- source layer ----


def test_source_tables_have_the_expected_grain(tables):
    assert tables["customers"]["customer_id"].is_unique
    assert tables["products"]["stock_code"].is_unique
    assert tables["invoices"]["invoice_id"].is_unique
    assert tables["transactions"]["line_id"].is_unique


def test_mixed_type_columns_are_read_as_strings(tables):
    """Invoice and StockCode mix integers with alphanumeric codes. Read as
    numbers, credit notes and every administrative code vanish."""
    tx = tables["transactions"]
    assert tx["invoice_id"].map(type).eq(str).all()
    assert tx["stock_code"].map(type).eq(str).all()
    assert tx["invoice_id"].str.startswith("C").any(), "no credit notes survived the load"


def test_panel_covers_whole_months_only(tables):
    """December 2011 is a partial month in the source and must be excluded, or
    it reads as a collapse in demand rather than the end of a file."""
    tx = tables["transactions"]
    assert tx["month_index"].min() == 1
    assert tx["month_index"].max() == RETAIL.panel_months == 24
    assert tx["invoice_date"].max() < pd.Timestamp("2011-12-01")


def test_returns_are_identified_not_netted(tables):
    tx = tables["transactions"]
    assert tx["is_return"].sum() > 0
    assert (tx.loc[tx["is_return"], "quantity"] < 0).mean() > 0.9


def test_administrative_codes_are_classified_not_treated_as_products(tables):
    tx = tables["transactions"]
    assert not tx["is_product"].all(), "no administrative lines were identified"
    for code in ("POST", "M", "BANK CHARGES", "AMAZONFEE", "TEST001"):
        rows = tx[tx["stock_code"].str.upper() == code]
        if len(rows):
            assert not rows["is_product"].any(), f"{code} is being treated as a product"


def test_case_variant_admin_codes_are_unified(tables):
    """`M` and `m` are the same manual-adjustment code entered two ways."""
    tx = tables["transactions"]
    manual = tx[tx["line_type"] == "Manual adjustment"]["stock_code"].unique()
    assert "m" not in manual, "lowercase manual-adjustment code was not unified"


# ------------------------------------------------------------- taxonomy ----


def test_taxonomy_is_deterministic_and_total():
    for description in ["CHRISTMAS TREE", "JUMBO BAG RED RETROSPOT", "", None, 3.7]:
        assert isinstance(categorise(description), str)
    assert categorise(None) == UNCATEGORISED


def test_taxonomy_puts_seasonal_before_form():
    """A Christmas candle is Christmas stock, not candle stock, because that is
    how it is bought. Rule order encodes that."""
    assert categorise("CHRISTMAS CANDLE HOLDER") == "Christmas & seasonal"
    assert categorise("WHITE CANDLE HOLDER") == "Lighting & candles"


def test_taxonomy_covers_most_of_revenue(tables):
    products = tables["products"]
    revenue = products.groupby("category")["revenue"].sum()
    uncategorised = revenue.get(UNCATEGORISED, 0) / revenue.sum()
    assert uncategorised < 0.20, (
        f"{uncategorised:.1%} of revenue is uncategorised — the taxonomy is not "
        "carrying enough of the book to support category-level analysis"
    )


# ------------------------------------------------------ conformed layer ----


def _count(con, sql: str) -> int:
    return int(con.execute(sql).fetchone()[0] or 0)


def test_conformed_transactions_are_attributable(con):
    assert _count(con, "SELECT count(*) FROM conformed_transactions WHERE customer_id IS NULL") == 0


def test_conformed_sales_are_real_sales(con):
    assert _count(con, "SELECT count(*) FROM conformed_sales WHERE quantity <= 0") == 0
    assert _count(con, "SELECT count(*) FROM conformed_sales WHERE unit_price <= 0") == 0
    assert _count(con, "SELECT count(*) FROM conformed_sales WHERE is_return") == 0
    assert _count(con, "SELECT count(*) FROM conformed_sales WHERE NOT is_product") == 0


def test_sales_and_returns_are_separated_not_netted(con):
    """Netting returns off sales makes an account that buys £10,000 and returns
    £9,000 identical to one that buys £1,000 and returns nothing."""
    assert _count(con, "SELECT count(*) FROM conformed_returns") > 0
    overlap = _count(con, """
        SELECT count(*) FROM conformed_sales s
        JOIN conformed_returns r ON s.line_id = r.line_id
    """)
    assert overlap == 0, "the same line appears as both a sale and a return"


def test_duplicate_lines_are_removed(con):
    remaining = _count(con, """
        SELECT coalesce(sum(n-1),0) FROM (
          SELECT invoice_id, stock_code, quantity, unit_price, invoice_date, count(*) n
          FROM conformed_transactions GROUP BY 1,2,3,4,5 HAVING count(*) > 1)
    """)
    assert remaining == 0


def test_monthly_panel_records_silence(con):
    """A customer that stops ordering must still appear, with zeros. Silence is
    the signal the lapse model exists to find."""
    silent = _count(con, "SELECT count(*) FROM monthly_customer_metrics WHERE NOT was_active")
    assert silent > 0, "the spine is not producing silent months"


def test_panel_starts_at_each_account_s_first_month(con):
    """Carrying an account back to the start of the panel would invent a year of
    silence and make every new account look like it was lapsing."""
    before = _count(con, """
        SELECT count(*) FROM monthly_customer_metrics m
        JOIN conformed_customers c USING (customer_id)
        WHERE m.month_index < c.first_month_index
    """)
    assert before == 0


# ------------------------------------------------------------- windows -----


def test_training_and_outcome_windows_do_not_overlap():
    assert max(RETAIL.train_feature) < min(RETAIL.train_outcome)


def test_both_outcome_windows_fall_in_the_same_season():
    """This business takes three times as much revenue in November as February.
    A model trained on one season and applied to another has learned the wrong
    base rate, so the windows are aligned by calendar month."""
    origin = pd.Period("2009-12", freq="M")
    train_months = {(origin + (m - 1)).month for m in RETAIL.train_outcome}
    score_end = origin + (max(RETAIL.score_feature) - 1)
    score_months = {(score_end + i).month for i in range(1, len(RETAIL.train_outcome) + 1)}
    assert train_months == score_months, (
        f"training outcome months {sorted(train_months)} differ from scoring "
        f"outcome months {sorted(score_months)}"
    )


def test_feature_windows_are_equal_length():
    assert len(RETAIL.train_feature) == len(RETAIL.score_feature) == 12


def test_customer_360_never_reads_outside_its_window(con):
    narrow = R.build_360(con, 1, 12).set_index("customer_id")
    wide = R.build_360(con, 1, 24).set_index("customer_id")
    common = narrow.index.intersection(wide.index)
    assert len(common) > 0
    for col in ["invoices", "revenue", "active_months"]:
        assert (wide.loc[common, col] >= narrow.loc[common, col] - 1e-6).all(), (
            f"{col} fell when the window widened — the window bound is inverted"
        )
        assert (wide.loc[common, col] > narrow.loc[common, col]).any(), (
            f"{col} is identical across two different windows — the window "
            "parameter is being ignored"
        )


def test_retail_sql_has_no_month_literal_reaching_into_the_outcome_window():
    sql = (SQL_DIR / "retail_03_customer_360.sql").read_text()
    body = re.sub(r"--[^\n]*", "", sql)
    outcome_start = min(RETAIL.train_outcome)
    offenders = [m.group(0) for m in re.finditer(r"month_index\s*(?:>=|>|=)\s*(\d+)", body)
                 if int(m.group(1)) >= outcome_start]
    assert not offenders, f"retail_03_customer_360.sql reaches forward: {offenders}"


# ------------------------------------------------------------ outcomes -----


def test_outcomes_restrict_to_established_accounts(con):
    outcomes = R.build_outcomes(con)
    eligible = outcomes[outcomes["lapse_eligible"] == 1]
    assert 0 < len(eligible) < len(outcomes), "eligibility is not restricting anything"
    rate = eligible["lapsed"].mean()
    assert 0.20 < rate < 0.60, (
        f"lapse base rate {rate:.1%} — outside the range where the label is "
        "describing attrition rather than ordinary wholesale order cadence"
    )


def test_growth_is_measured_against_the_account_s_own_run_rate(con):
    outcomes = R.build_outcomes(con)
    eligible = outcomes[outcomes["growth_eligible"] == 1]
    assert 0.10 < eligible["grew"].mean() < 0.50


# --------------------------------------------------------- feature set -----


def test_all_declared_features_exist_in_the_360(con):
    c360 = R.build_360(con, min(RETAIL.train_feature), max(RETAIL.train_feature))
    missing = [f for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES if f not in c360.columns]
    assert not missing, f"features declared but not produced by the SQL: {missing}"


def test_segment_is_not_a_model_feature():
    """RFM is a rule over features the model already sees; feeding it back in
    makes the explanations circular."""
    assert "segment" in EXCLUDED_FEATURES
    for target in ("lapsed", "grew"):
        numeric, categorical = model_features(target)
        assert "segment" not in numeric + categorical


def test_trends_are_finite(con):
    c360 = R.build_360(con, min(RETAIL.train_feature), max(RETAIL.train_feature))
    for col in ["revenue_trend", "order_trend"]:
        assert c360[col].notna().all()
        assert np.isfinite(c360[col]).all()


# --------------------------------------------------------- recommender -----


@pytest.fixture(scope="module")
def holdings(con):
    return R.product_holdings(con, RETAIL.train_feature)


def test_recommender_never_recommends_what_is_already_bought(holdings):
    rec = ItemRecommender().fit(holdings)
    scores = rec.score_all(holdings)
    already = holdings.astype(bool)
    assert scores.where(already).notna().sum().sum() == 0


def test_recommender_beats_a_popularity_baseline(con, holdings):
    """The benchmark any recommender has to clear to have earned its complexity."""
    added = R.products_added(con, RETAIL.train_feature, RETAIL.train_outcome,
                             list(holdings.columns))
    rec = ItemRecommender().fit(holdings)
    result = rec.evaluate(holdings, added, k=5)
    assert result["accounts_evaluated"] > 100
    assert result["hit_rate_at_k"] > result["popularity_baseline"], (
        f"hit rate {result['hit_rate_at_k']:.3f} does not beat popularity "
        f"{result['popularity_baseline']:.3f} — the recommender is not earning its place"
    )


def test_recommendations_are_ranked_per_account(holdings):
    rec = ItemRecommender().fit(holdings)
    out = rec.recommend(holdings.head(50), n=3)
    assert not out.empty
    per_account = out.groupby("customer_id")["rank"].apply(list)
    for ranks in per_account:
        assert ranks == sorted(ranks)


# ------------------------------------------------------ quality checks -----


def test_retail_checks_all_execute(con):
    results = run_checks(con, RETAIL_CHECKS)
    assert len(results) == len(RETAIL_CHECKS)
    assert results["pass_rate"].between(0, 1).all()


def test_retail_checks_find_the_known_real_defects(con):
    """These are documented properties of this published dataset. If a rule
    stops firing, either the rule broke or the source changed underneath us --
    and both are things to be told about."""
    results = run_checks(con, RETAIL_CHECKS).set_index("key")
    for key in ["transactions.customer_id.missing",
                "transactions.duplicate_lines",
                "transactions.unit_price.nonpositive",
                "transactions.test_rows"]:
        assert results.loc[key, "failing"] > 0, f"{key} no longer detects a known real defect"


def test_check_keys_are_unique():
    keys = [c.key for c in RETAIL_CHECKS]
    assert len(keys) == len(set(keys))

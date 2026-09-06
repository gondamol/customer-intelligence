"""The retail analytical layer: build, features, outcomes, segmentation.

Everything specific to Online Retail II lives here. The generic machinery --
model training and evaluation, the opportunity score, the decision engine -- is
shared with every other domain, which is the point of keeping this file
separate rather than branching inside those.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..domains import RETAIL
from ..io import run_sql_file

# ---------------------------------------------------------------- build ----


def build_layer(con) -> None:
    """Raw -> conformed -> monthly panel."""
    run_sql_file(con, "retail_01_conform.sql")
    run_sql_file(con, "retail_02_monthly_metrics.sql")


def set_window(con, month_from: int, month_to: int) -> None:
    con.execute(
        "CREATE OR REPLACE TABLE analysis_window AS "
        f"SELECT {int(month_from)} AS month_from, {int(month_to)} AS month_to"
    )


def build_360(con, month_from: int, month_to: int) -> pd.DataFrame:
    set_window(con, month_from, month_to)
    run_sql_file(con, "retail_03_customer_360.sql")
    return con.execute("SELECT * FROM customer_360").df()


def conformance_log(con) -> pd.DataFrame:
    """What the cleanse removed, rule by rule, and the reasoning.

    Nothing is dropped silently. The largest single line here -- a fifth of all
    invoice lines discarded for having no customer identifier -- is the kind of
    number that has to be on a page rather than in a footnote, because it
    changes what every customer-level figure downstream can claim to describe.
    """
    def n(sql: str) -> int:
        return int(con.execute(sql).fetchone()[0] or 0)

    landed = n("SELECT count(*) FROM transactions")
    rows = [
        ("transactions", "Landed invoice lines", landed, ""),
        ("transactions", "Dropped — exact duplicate line",
         n("SELECT coalesce(sum(n-1),0) FROM (SELECT invoice_id, stock_code, quantity, "
           "unit_price, invoice_date, count(*) n FROM transactions GROUP BY 1,2,3,4,5 "
           "HAVING count(*)>1)"),
         "A re-run of a load, not a customer buying the same item twice in the same second."),
        ("transactions", "Dropped — no customer identifier",
         n("SELECT count(*) FROM transactions WHERE customer_id IS NULL"),
         "Real revenue, but it cannot be attributed to a relationship, so it cannot "
         "enter a customer-level model. Retained in the revenue tables."),
        ("transactions", "Dropped — non-positive unit price",
         n("SELECT count(*) FROM transactions WHERE unit_price <= 0"),
         "Stock adjustments and write-offs booked through the sales table."),
        ("transactions", "Dropped — no description",
         n("SELECT count(*) FROM transactions WHERE description IS NULL"),
         "Cannot be classified to a product or a category."),
        ("transactions", "Conformed lines", n("SELECT count(*) FROM conformed_transactions"), ""),
        ("transactions", "→ of which sales", n("SELECT count(*) FROM conformed_sales"),
         "Positive-quantity product lines on a sales invoice."),
        ("transactions", "→ of which returns", n("SELECT count(*) FROM conformed_returns"),
         "Separated, not netted off: a customer who buys £10,000 and returns £9,000 is "
         "a different relationship from one who buys £1,000 and returns nothing."),
        ("transactions", "→ of which charges and adjustments",
         n("SELECT count(*) FROM conformed_charges"),
         "Postage, carriage, discounts, samples, bank charges, marketplace commission, "
         "bad debt, gift vouchers and test rows. Real money, not products."),
        ("customers", "Identified accounts", n("SELECT count(*) FROM conformed_customers"), ""),
        ("customers", "Retained — flagged multi-country",
         n("SELECT count(*) FROM conformed_customers WHERE multi_country"),
         "Flagged rather than resolved: with no source of truth, picking one would be a guess."),
        ("products", "Stock codes", n("SELECT count(*) FROM conformed_products"), ""),
        ("products", "Categorised by the derived taxonomy",
         n("SELECT count(*) FROM conformed_products WHERE category <> 'Other giftware'"),
         "Keyword rules over the description text. A judgement, published in "
         "sources/retail.py, not a fact about the data."),
    ]
    return pd.DataFrame(rows, columns=["table", "rule", "rows", "decision"])


# -------------------------------------------------------------- outcomes ----

# An account is "established" if it ordered in at least this many distinct
# months of the twelve-month feature window. Below it, a quiet quarter is
# ordinary trading rather than a lapse: at one month the apparent lapse rate is
# 67%, which is not attrition, it is a wholesaler's normal order cadence.
ESTABLISHED_MIN_ACTIVE_MONTHS = 3


def build_outcomes(
    con,
    feature_months: tuple[int, ...] = RETAIL.train_feature,
    outcome_months: tuple[int, ...] = RETAIL.train_outcome,
) -> pd.DataFrame:
    """Derive the three modelled outcomes over the outcome window.

    Each carries its own eligibility flag. The lapse model is the one that
    needed the most care: "no order in three months" describes two thirds of
    this book and means nothing, so the population is restricted to accounts
    that had actually established a rhythm.
    """
    f_from, f_to = min(feature_months), max(feature_months)
    o_from, o_to = min(outcome_months), max(outcome_months)

    return con.execute(f"""
        WITH feature_activity AS (
            SELECT customer_id,
                   sum(CASE WHEN was_active THEN 1 ELSE 0 END) AS active_months,
                   sum(revenue)                                AS revenue_f,
                   sum(invoices)                               AS invoices_f
            FROM monthly_customer_metrics
            WHERE month_index BETWEEN {f_from} AND {f_to}
            GROUP BY customer_id
        ),
        prior_year_quarter AS (
            -- The same three calendar months, one year earlier.
            SELECT customer_id, sum(revenue) AS revenue_py
            FROM monthly_customer_metrics
            WHERE month_index BETWEEN {o_from} - 12 AND {o_to} - 12
            GROUP BY customer_id
        ),
        outcome_activity AS (
            SELECT customer_id,
                   sum(revenue)  AS revenue_o,
                   sum(invoices) AS invoices_o
            FROM monthly_customer_metrics
            WHERE month_index BETWEEN {o_from} AND {o_to}
            GROUP BY customer_id
        ),
        cats_before AS (
            SELECT s.customer_id, p.category
            FROM conformed_sales s
            JOIN conformed_products p ON s.stock_code = p.stock_code
            WHERE s.month_index BETWEEN {f_from} AND {f_to}
            GROUP BY 1, 2
        ),
        cats_after AS (
            SELECT s.customer_id, p.category
            FROM conformed_sales s
            JOIN conformed_products p ON s.stock_code = p.stock_code
            WHERE s.month_index BETWEEN {o_from} AND {o_to}
            GROUP BY 1, 2
        ),
        expansion AS (
            SELECT a.customer_id, 1 AS took_new_category
            FROM cats_after a
            LEFT JOIN cats_before b
                   ON a.customer_id = b.customer_id AND a.category = b.category
            WHERE b.category IS NULL
            GROUP BY a.customer_id
        )
        SELECT
            f.customer_id,

            -- Lapse: an established account that placed no order at all across
            -- the whole outcome quarter.
            CASE WHEN coalesce(o.invoices_o, 0) = 0 THEN 1 ELSE 0 END        AS lapsed,
            CASE WHEN f.active_months >= {ESTABLISHED_MIN_ACTIVE_MONTHS}
                 THEN 1 ELSE 0 END                                           AS lapse_eligible,

            -- Growth: outcome-quarter spend beat the SAME QUARTER A YEAR EARLIER.
            --
            -- The obvious target -- beating the account's own annual quarterly
            -- run-rate -- is wrong in a seasonal business. The outcome window is
            -- December to February, the low season, so most accounts fall below
            -- their annual average whatever they do, and the model spends its
            -- capacity learning the calendar. Comparing like quarter with like
            -- quarter removes that: ROC-AUC 0.62 -> 0.67 on the same features.
            --
            -- The cost is a smaller eligible population: an account has to have
            -- traded in the prior-year quarter to have anything to be compared
            -- against. That is a real restriction and it is declared, not hidden.
            CASE WHEN coalesce(o.revenue_o, 0) > coalesce(py.revenue_py, 0)
                 THEN 1 ELSE 0 END                                           AS grew,
            CASE WHEN f.active_months >= {ESTABLISHED_MIN_ACTIVE_MONTHS}
                  AND coalesce(py.revenue_py, 0) > 0
                 THEN 1 ELSE 0 END                                           AS growth_eligible,

            -- Category expansion: bought from a category they had never bought.
            coalesce(e.took_new_category, 0)                                 AS took_new_category,
            CASE WHEN f.active_months >= {ESTABLISHED_MIN_ACTIVE_MONTHS}
                 THEN 1 ELSE 0 END                                           AS expansion_eligible
        FROM feature_activity f
        LEFT JOIN outcome_activity o ON f.customer_id = o.customer_id
        LEFT JOIN prior_year_quarter py ON f.customer_id = py.customer_id
        LEFT JOIN expansion       e ON f.customer_id = e.customer_id
        WHERE f.invoices_f > 0
    """).df()


# ---------------------------------------------------------- segmentation ----

SEGMENT_ORDER = [
    "Champions", "Loyal", "Promising", "Needs attention",
    "At risk", "Hibernating", "Lost", "New",
]

SEGMENT_DESCRIPTIONS = {
    "Champions": "Recent, frequent and high-spending. The core of the book, and the most expensive accounts to lose.",
    "Loyal": "Order regularly and reliably. Not the largest spenders, but the most predictable revenue in the ledger.",
    "Promising": "Recent and growing, without the volume yet. The pipeline for the tiers above.",
    "Needs attention": "Solid history, but the ordering rhythm has slipped. Still reachable, and the cheapest group to save.",
    "At risk": "Were valuable and have gone quiet well past their own usual gap. Time-critical.",
    "Hibernating": "Long silent, with real history behind them. Worth one structured win-back attempt, not a campaign.",
    "Lost": "No order for most of the window and little value when active. Serve through self-service.",
    "New": "First ordered too recently to have a rhythm to judge. Treat as onboarding, not as low value.",
}


def rfm_segments(c360: pd.DataFrame) -> pd.Series:
    """RFM segmentation, the standard retail rule set, scored on this book.

    Recency, frequency and monetary value are each cut into quintiles *within
    this population*, then combined by published rules. Quintiles rather than
    fixed thresholds because RFM is inherently comparative: "recent" only means
    anything against how recently everyone else ordered.
    """
    df = c360
    new = df["months_on_book"] <= 3

    def quintile(series: pd.Series, ascending: bool) -> pd.Series:
        ranked = series.rank(pct=True, ascending=ascending)
        return np.ceil(ranked * 5).clip(1, 5)

    r = quintile(df["months_since_last_order"], ascending=False)   # recent -> 5
    f = quintile(df["invoices"], ascending=True)
    m = quintile(df["revenue"], ascending=True)
    fm = (f + m) / 2

    seg = pd.Series("Needs attention", index=df.index, dtype=object)
    seg[(r >= 4) & (fm >= 4)] = "Champions"
    seg[(r >= 3) & (r < 5) & (fm >= 3)] = "Loyal"
    seg[(r >= 4) & (fm < 3)] = "Promising"
    seg[(r == 3) & (fm < 3)] = "Needs attention"
    seg[(r == 2) & (fm >= 3)] = "At risk"
    seg[(r == 2) & (fm < 3)] = "Hibernating"
    seg[(r == 1) & (fm >= 3)] = "At risk"
    seg[(r == 1) & (fm < 3)] = "Lost"
    seg[new] = "New"
    return seg


def segment_profiles(c360: pd.DataFrame, segment_col: str = "segment") -> pd.DataFrame:
    agg = c360.groupby(segment_col, observed=True).agg(
        customers=("customer_id", "size"),
        avg_revenue=("revenue", "mean"),
        median_revenue=("revenue", "median"),
        total_revenue=("revenue", "sum"),
        avg_orders=("invoices", "mean"),
        avg_order_value=("avg_order_value", "mean"),
        avg_months_since_order=("months_since_last_order", "mean"),
        active_months=("active_months", "mean"),
        distinct_categories=("distinct_categories", "mean"),
        return_rate=("return_rate", "mean"),
        avg_cadence=("avg_months_between_orders", "mean"),
    ).reset_index()

    agg["share_of_customers"] = agg["customers"] / agg["customers"].sum() * 100
    agg["share_of_revenue"] = agg["total_revenue"] / agg["total_revenue"].sum() * 100
    agg["description"] = agg[segment_col].map(SEGMENT_DESCRIPTIONS).fillna("")

    order = {name: i for i, name in enumerate(SEGMENT_ORDER)}
    agg["_o"] = agg[segment_col].map(order).fillna(99)
    return agg.sort_values("_o", ignore_index=True).drop(columns="_o")


# ------------------------------------------------------ category holdings ----


def category_holdings(con, months: tuple[int, ...]) -> pd.DataFrame:
    """Boolean account x category matrix over a month window."""
    lo, hi = min(months), max(months)
    long = con.execute(f"""
        SELECT s.customer_id, p.category, count(*) AS lines
        FROM conformed_sales s
        JOIN conformed_products p ON s.stock_code = p.stock_code
        WHERE s.month_index BETWEEN {lo} AND {hi}
        GROUP BY 1, 2
    """).df()
    if long.empty:
        return pd.DataFrame()
    return (long.pivot(index="customer_id", columns="category", values="lines")
                .notna())


def categories_added(
    con, feature_months: tuple[int, ...], outcome_months: tuple[int, ...]
) -> pd.DataFrame:
    """Categories each account bought in the outcome window and never before."""
    before = category_holdings(con, feature_months)
    after = category_holdings(con, outcome_months)
    after = after.reindex(index=before.index, columns=before.columns, fill_value=False)
    return after & ~before


# The recommender is fitted over the most widely-bought products rather than all
# 5,278 stock codes. A long tail bought by one account each carries no
# co-purchase signal, and squaring 5,278 items to get a similarity matrix costs
# 220 MB to learn nothing. This is a stated cut-off, not a silent one.
RECOMMENDER_TOP_PRODUCTS = 400


def product_holdings(
    con, months: tuple[int, ...], top_products: int = RECOMMENDER_TOP_PRODUCTS,
    restrict_to: list[str] | None = None,
) -> pd.DataFrame:
    """Boolean account x product matrix over a month window."""
    lo, hi = min(months), max(months)
    long = con.execute(f"""
        SELECT customer_id, stock_code, count(*) AS lines
        FROM conformed_sales
        WHERE month_index BETWEEN {lo} AND {hi}
        GROUP BY 1, 2
    """).df()
    if long.empty:
        return pd.DataFrame()

    if restrict_to is None:
        keep = (long.groupby("stock_code")["customer_id"].nunique()
                    .nlargest(top_products).index)
    else:
        keep = pd.Index(restrict_to)
    long = long[long["stock_code"].isin(keep)]
    return (long.pivot(index="customer_id", columns="stock_code", values="lines")
                .reindex(columns=keep).notna())


def products_added(
    con, feature_months: tuple[int, ...], outcome_months: tuple[int, ...],
    products: list[str],
) -> pd.DataFrame:
    """Products bought in the outcome window and never before."""
    before = product_holdings(con, feature_months, restrict_to=products)
    after = product_holdings(con, outcome_months, restrict_to=products)
    after = after.reindex(index=before.index, columns=before.columns, fill_value=False)
    return after & ~before

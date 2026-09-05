"""Orchestration of the analytical layer.

Thin Python around the SQL in ``sql/``. The transformations themselves live in
SQL because that is where they belong and where they can be read; this module
sets the analysis window, records what the cleanse removed, and derives the
outcome labels.
"""

from __future__ import annotations

import pandas as pd

from ..config import (
    SCORE_FEATURE_MONTHS, TRAIN_FEATURE_MONTHS, TRAIN_OUTCOME_MONTHS,
)
from ..io import run_sql_file


def build_analytical_layer(con) -> None:
    """Raw -> conformed -> monthly panel."""
    run_sql_file(con, "01_conform.sql")
    run_sql_file(con, "02_monthly_metrics.sql")


def set_window(con, month_from: int, month_to: int) -> None:
    con.execute(
        "CREATE OR REPLACE TABLE analysis_window AS "
        f"SELECT {int(month_from)} AS month_from, {int(month_to)} AS month_to"
    )


def build_customer_360(con, month_from: int, month_to: int) -> pd.DataFrame:
    """Build the 360 view over one month window and return it."""
    set_window(con, month_from, month_to)
    run_sql_file(con, "03_customer_360.sql")
    return con.execute("SELECT * FROM customer_360").df()


def conformance_log(con) -> pd.DataFrame:
    """What the cleanse removed, rule by rule.

    Nothing is dropped silently. Every row that does not survive the conformed
    layer is accounted for here, which is the difference between cleaning data
    and quietly losing it.
    """
    def n(sql: str) -> int:
        return int(con.execute(sql).fetchone()[0] or 0)

    rows = [
        ("customers", "Landed rows", n("SELECT count(*) FROM customers"), ""),
        ("customers", "Dropped — no customer_id",
         n("SELECT count(*) FROM customers WHERE customer_id IS NULL"),
         "A row with no key cannot be joined or attributed."),
        ("customers", "Dropped — duplicate key",
         n("SELECT coalesce(sum(n-1),0) FROM (SELECT customer_id, count(*) n FROM customers "
           "WHERE customer_id IS NOT NULL GROUP BY customer_id HAVING count(*)>1)"),
         "First record retained; later re-lands discarded."),
        ("customers", "Retained — age set to NULL",
         n("SELECT count(*) FROM conformed_customers WHERE age_was_invalid"),
         "Impossible age nulled rather than imputed."),
        ("customers", "Conformed rows", n("SELECT count(*) FROM conformed_customers"), ""),

        ("accounts", "Landed rows", n("SELECT count(*) FROM accounts"), ""),
        ("accounts", "Dropped — orphaned customer_id",
         n("SELECT count(*) FROM accounts a LEFT JOIN conformed_customers c "
           "USING (customer_id) WHERE c.customer_id IS NULL"),
         "No parent customer; cannot be attributed to a relationship."),
        ("accounts", "Retained — status standardised",
         n("SELECT count(*) FROM accounts WHERE status IS NOT NULL AND status NOT IN "
           "('Active','Dormant','Closed')"),
         "Mapped onto the controlled vocabulary, or 'Unknown'."),
        ("accounts", "Conformed rows", n("SELECT count(*) FROM conformed_accounts"), ""),

        ("transactions", "Landed rows", n("SELECT count(*) FROM transactions"), ""),
        ("transactions", "Dropped — duplicate posting",
         n("SELECT coalesce(sum(n-1),0) FROM (SELECT transaction_id, count(*) n "
           "FROM transactions GROUP BY transaction_id HAVING count(*)>1)"),
         "Earliest posting retained."),
        ("transactions", "Dropped — missing or non-positive amount",
         n("SELECT count(*) FROM transactions WHERE amount IS NULL OR amount <= 0"),
         "Contributes no measurable value."),
        ("transactions", "Dropped — anomalous amount",
         n("SELECT count(*) FROM transactions WHERE amount > 5000000"),
         "Quarantined, not capped: a capped value is a fabricated one."),
        ("transactions", "Dropped — dated after reporting close",
         n("SELECT count(*) FROM transactions WHERE transaction_date >= DATE '2026-01-01'"),
         "Outside the reporting window."),
        ("transactions", "Conformed rows", n("SELECT count(*) FROM conformed_transactions"), ""),

        ("products", "Landed rows", n("SELECT count(*) FROM products"), ""),
        ("products", "Retained — code standardised",
         n("SELECT count(*) FROM products WHERE product_type NOT IN "
           "('Current account','Savings','Investment','Personal loan','Asset finance',"
           "'Card','Insurance','Digital wallet')"),
         "Source spellings mapped to the canonical vocabulary."),
        ("products", "Conformed rows", n("SELECT count(*) FROM conformed_products"), ""),
    ]
    return pd.DataFrame(rows, columns=["table", "rule", "rows", "decision"])


# ------------------------------------------------------------- outcomes ----


def build_outcomes(
    con,
    feature_months: tuple[int, ...] = TRAIN_FEATURE_MONTHS,
    outcome_months: tuple[int, ...] = TRAIN_OUTCOME_MONTHS,
) -> pd.DataFrame:
    """Derive the three modelled outcomes over the outcome window.

    Each outcome carries its own eligibility flag. Training on the whole book
    when only part of it could have experienced the outcome is one of the
    commonest ways a propensity model ends up learning eligibility instead of
    propensity.
    """
    f_from, f_to = min(feature_months), max(feature_months)
    o_from, o_to = min(outcome_months), max(outcome_months)

    return con.execute(f"""
        WITH feature_activity AS (
            SELECT customer_id,
                   sum(transaction_count) AS f_txn,
                   avg(total_logins)      AS f_logins
            FROM monthly_customer_metrics
            WHERE month_index BETWEEN {f_from} AND {f_to}
            GROUP BY customer_id
        ),
        outcome_activity AS (
            SELECT customer_id,
                   sum(transaction_count) AS o_txn,
                   avg(total_logins)      AS o_logins
            FROM monthly_customer_metrics
            WHERE month_index BETWEEN {o_from} AND {o_to}
            GROUP BY customer_id
        ),
        held_at_cutoff AS (
            SELECT customer_id,
                   max(CASE WHEN product_type = 'Investment' THEN 1 ELSE 0 END) AS had_investment,
                   max(CASE WHEN product_type IN ('Personal loan', 'Asset finance')
                            THEN 1 ELSE 0 END)                                  AS had_lending
            FROM conformed_products
            WHERE start_month_index <= {f_to}
            GROUP BY customer_id
        ),
        acquired AS (
            SELECT customer_id,
                   max(CASE WHEN product_type = 'Investment' THEN 1 ELSE 0 END) AS took_investment,
                   max(CASE WHEN product_type IN ('Personal loan', 'Asset finance')
                            THEN 1 ELSE 0 END)                                  AS took_lending
            FROM conformed_products
            WHERE start_month_index BETWEEN {o_from} AND {o_to}
            GROUP BY customer_id
        )
        SELECT
            c.customer_id,

            -- Attrition: active during the feature window, then effectively
            -- silent across the whole outcome window. Two conditions, because a
            -- single quiet month is not attrition.
            CASE WHEN coalesce(o.o_txn, 0) < 2
                  AND coalesce(o.o_logins, 0) < 0.25 * coalesce(f.f_logins, 0)
                 THEN 1 ELSE 0 END                            AS churned,
            CASE WHEN coalesce(f.f_txn, 0) >= 3 THEN 1 ELSE 0 END AS churn_eligible,

            coalesce(a.took_investment, 0)                    AS took_investment,
            CASE WHEN coalesce(h.had_investment, 0) = 0 THEN 1 ELSE 0 END AS investment_eligible,

            coalesce(a.took_lending, 0)                       AS took_lending,
            CASE WHEN coalesce(h.had_lending, 0) = 0 THEN 1 ELSE 0 END    AS lending_eligible
        FROM conformed_customers c
        LEFT JOIN feature_activity f ON c.customer_id = f.customer_id
        LEFT JOIN outcome_activity o ON c.customer_id = o.customer_id
        LEFT JOIN held_at_cutoff  h ON c.customer_id = h.customer_id
        LEFT JOIN acquired        a ON c.customer_id = a.customer_id
    """).df()

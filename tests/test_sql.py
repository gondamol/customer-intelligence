"""The SQL layer: that it runs, and that the conformed tables are actually clean."""

from __future__ import annotations

import pytest

from customer_intelligence.config import SQL_DIR


def _count(con, sql: str) -> int:
    return int(con.execute(sql).fetchone()[0] or 0)


def test_all_sql_files_are_present():
    for name in ["01_conform.sql", "02_monthly_metrics.sql", "03_customer_360.sql",
                 "04_analytics.sql"]:
        assert (SQL_DIR / name).exists(), f"{name} is missing"


def test_conformed_customers_have_a_unique_key(analytical):
    assert _count(analytical, "SELECT count(*) FROM conformed_customers WHERE customer_id IS NULL") == 0
    assert _count(analytical,
                  "SELECT count(*) FROM (SELECT customer_id FROM conformed_customers "
                  "GROUP BY customer_id HAVING count(*) > 1)") == 0


def test_conformed_layer_has_no_orphans(analytical):
    for table in ["conformed_accounts", "conformed_transactions", "conformed_loans",
                  "conformed_products", "conformed_service_interactions"]:
        orphans = _count(analytical, f"""
            SELECT count(*) FROM {table} t
            LEFT JOIN conformed_customers c ON t.customer_id = c.customer_id
            WHERE c.customer_id IS NULL
        """)
        assert orphans == 0, f"{table} still has {orphans} orphaned rows"


def test_conformed_transactions_are_clean(analytical):
    assert _count(analytical, "SELECT count(*) FROM conformed_transactions WHERE amount IS NULL") == 0
    assert _count(analytical, "SELECT count(*) FROM conformed_transactions WHERE amount <= 0") == 0
    assert _count(analytical, "SELECT count(*) FROM conformed_transactions WHERE amount > 5000000") == 0
    assert _count(analytical,
                  "SELECT count(*) FROM conformed_transactions "
                  "WHERE transaction_date >= DATE '2026-01-01'") == 0
    assert _count(analytical,
                  "SELECT count(*) FROM (SELECT transaction_id FROM conformed_transactions "
                  "GROUP BY transaction_id HAVING count(*) > 1)") == 0


def test_conformed_statuses_use_the_controlled_vocabulary(analytical):
    assert _count(analytical,
                  "SELECT count(*) FROM conformed_accounts WHERE status NOT IN "
                  "('Active','Dormant','Closed','Unknown')") == 0
    assert _count(analytical,
                  "SELECT count(*) FROM conformed_products WHERE product_type NOT IN "
                  "('Current account','Savings','Investment','Personal loan',"
                  "'Asset finance','Card','Insurance','Digital wallet')") == 0


def test_impossible_ages_are_nulled_not_imputed(analytical):
    """A cleaned age must be missing, never a plausible-looking guess."""
    assert _count(analytical,
                  "SELECT count(*) FROM conformed_customers "
                  "WHERE age IS NOT NULL AND (age < 18 OR age > 100)") == 0
    assert _count(analytical, "SELECT count(*) FROM conformed_customers WHERE age_was_invalid") > 0
    assert _count(analytical,
                  "SELECT count(*) FROM conformed_customers "
                  "WHERE age_was_invalid AND age IS NOT NULL") == 0


def test_monthly_panel_is_a_complete_spine(analytical):
    """Every customer must appear in every month, including silent ones.

    Silence is the strongest attrition signal there is; a panel built by
    aggregating transactions alone would drop exactly the customers that matter.
    """
    customers = _count(analytical, "SELECT count(*) FROM conformed_customers")
    months = _count(analytical, "SELECT count(DISTINCT month_index) FROM monthly_customer_metrics")
    rows = _count(analytical, "SELECT count(*) FROM monthly_customer_metrics")
    assert rows == customers * months

    silent = _count(analytical,
                    "SELECT count(*) FROM monthly_customer_metrics WHERE transaction_count = 0")
    assert silent > 0, "no silent customer-months exist — the spine is not doing its job"


def test_analytics_queries_execute(analytical):
    from customer_intelligence.io import run_sql_file

    run_sql_file(analytical, "04_analytics.sql")
    assert _count(analytical, "SELECT count(*) FROM product_penetration") > 0

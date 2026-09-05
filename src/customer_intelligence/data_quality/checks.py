"""The data quality rule set.

Each check is a declarative record -- a rule in words, and the SQL that measures
it. They run against the raw layer through DuckDB, so the same statements would
run against PostgreSQL with no change.

Checks are grouped by the six dimensions the report scores (completeness,
uniqueness, validity, consistency, integrity, timeliness) plus accuracy for
distributional anomalies. Not every check corresponds to an injected defect:
several exist to demonstrate that the framework is a general rule set, not a
lookup table for the known answers.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# The close of the reporting window. Anything dated after this is not yet
# possible and is treated as a timeliness failure.
REPORTING_CLOSE = "2026-01-01"

# Beyond the plausible retail range by three orders of magnitude. A fixed,
# documented threshold rather than a quantile, so the rule does not move when
# the population does.
AMOUNT_ANOMALY_THRESHOLD = 5_000_000

VALID_ACCOUNT_STATUSES = ("Active", "Dormant", "Closed")
VALID_PRODUCT_TYPES = (
    "Current account", "Savings", "Investment", "Personal loan",
    "Asset finance", "Card", "Insurance", "Digital wallet",
)


@dataclass(frozen=True)
class Check:
    key: str
    table: str
    dimension: str
    severity: str          # high | medium | low
    rule: str              # the rule, in business language
    sql: str               # must return a single column named `failing`
    column: str = ""
    # Optional: the customers touched by this defect. Used to answer "how many
    # customers does this actually affect?", which is the question a business
    # owner asks and a row-level pass rate cannot answer.
    impact_sql: str = ""

    @property
    def label(self) -> str:
        target = f"{self.table}.{self.column}" if self.column else self.table
        return f"{target} — {self.rule}"


def _q(values) -> str:
    return ", ".join(f"'{v}'" for v in values)


CHECKS: tuple[Check, ...] = (
    # ------------------------------------------------------- completeness --
    Check(
        key="customers.customer_id.missing", table="customers", column="customer_id",
        dimension="Completeness", severity="high",
        rule="Every customer record must carry an identifier",
        sql="SELECT count(*) AS failing FROM customers WHERE customer_id IS NULL",
    ),
    Check(
        key="customers.region.missing", table="customers", column="region",
        dimension="Completeness", severity="low",
        rule="Region should be captured at onboarding",
        sql="SELECT count(*) AS failing FROM customers WHERE region IS NULL",
        impact_sql=("SELECT DISTINCT customer_id FROM customers WHERE region IS NULL AND customer_id IS NOT NULL")
    ),
    Check(
        key="customers.income_band.missing", table="customers", column="income_band",
        dimension="Completeness", severity="medium",
        rule="Income band is required for segmentation and affordability",
        sql="SELECT count(*) AS failing FROM customers WHERE income_band IS NULL",
        impact_sql=("SELECT DISTINCT customer_id FROM customers WHERE income_band IS NULL AND customer_id IS NOT NULL")
    ),
    Check(
        key="accounts.opening_date.missing", table="accounts", column="opening_date",
        dimension="Completeness", severity="medium",
        rule="Every account must record when it was opened",
        sql="SELECT count(*) AS failing FROM accounts WHERE opening_date IS NULL",
        impact_sql=("SELECT DISTINCT customer_id FROM accounts WHERE opening_date IS NULL")
    ),
    Check(
        key="transactions.amount.missing", table="transactions", column="amount",
        dimension="Completeness", severity="high",
        rule="A transaction without an amount cannot be measured",
        sql="SELECT count(*) AS failing FROM transactions WHERE amount IS NULL",
        impact_sql=("SELECT DISTINCT customer_id FROM transactions WHERE amount IS NULL")
    ),
    Check(
        key="loans.interest_rate.missing", table="loans", column="interest_rate",
        dimension="Completeness", severity="medium",
        rule="Active facilities must record an interest rate",
        sql=("SELECT count(*) AS failing FROM loans "
             "WHERE interest_rate IS NULL AND repayment_status <> 'Closed'"),
        impact_sql=("SELECT DISTINCT customer_id FROM loans WHERE interest_rate IS NULL AND repayment_status <> 'Closed'")
    ),
    Check(
        key="service_interactions.satisfaction.missing", table="service_interactions",
        column="satisfaction_score", dimension="Completeness", severity="low",
        rule="Post-contact satisfaction should be recorded (missing not at random)",
        sql="SELECT count(*) AS failing FROM service_interactions WHERE satisfaction_score IS NULL",
    ),

    # ---------------------------------------------------------- uniqueness --
    Check(
        key="customers.customer_id.duplicate", table="customers", column="customer_id",
        dimension="Uniqueness", severity="high",
        rule="customer_id must identify exactly one record",
        sql=("SELECT coalesce(sum(n - 1), 0) AS failing FROM ("
             "  SELECT customer_id, count(*) AS n FROM customers "
             "  WHERE customer_id IS NOT NULL GROUP BY customer_id HAVING count(*) > 1)"),
        impact_sql=("SELECT customer_id FROM customers WHERE customer_id IS NOT NULL GROUP BY customer_id HAVING count(*) > 1")
    ),
    Check(
        key="transactions.transaction_id.duplicate", table="transactions",
        column="transaction_id", dimension="Uniqueness", severity="high",
        rule="A transaction must be posted exactly once",
        sql=("SELECT coalesce(sum(n - 1), 0) AS failing FROM ("
             "  SELECT transaction_id, count(*) AS n FROM transactions "
             "  GROUP BY transaction_id HAVING count(*) > 1)"),
        impact_sql=("SELECT DISTINCT customer_id FROM transactions WHERE transaction_id IN (SELECT transaction_id FROM transactions GROUP BY transaction_id HAVING count(*) > 1)")
    ),
    Check(
        key="accounts.account_id.duplicate", table="accounts", column="account_id",
        dimension="Uniqueness", severity="high",
        rule="account_id must identify exactly one account",
        sql=("SELECT coalesce(sum(n - 1), 0) AS failing FROM ("
             "  SELECT account_id, count(*) AS n FROM accounts "
             "  GROUP BY account_id HAVING count(*) > 1)"),
    ),

    # ------------------------------------------------------------ validity --
    Check(
        key="customers.age.invalid", table="customers", column="age",
        dimension="Validity", severity="high",
        rule="Age must fall within 18–100",
        sql="SELECT count(*) AS failing FROM customers WHERE age < 18 OR age > 100",
        impact_sql=("SELECT DISTINCT customer_id FROM customers WHERE (age < 18 OR age > 100) AND customer_id IS NOT NULL")
    ),
    Check(
        key="accounts.balance.invalid", table="accounts", column="current_balance",
        dimension="Validity", severity="high",
        rule="Savings and fixed deposit balances cannot be negative",
        sql=("SELECT count(*) AS failing FROM accounts "
             "WHERE account_type IN ('Savings', 'Fixed deposit') AND current_balance < 0"),
        impact_sql=("SELECT DISTINCT customer_id FROM accounts WHERE account_type IN ('Savings', 'Fixed deposit') AND current_balance < 0")
    ),
    Check(
        key="transactions.amount.nonpositive", table="transactions", column="amount",
        dimension="Validity", severity="medium",
        rule="Transaction amounts must be strictly positive",
        sql="SELECT count(*) AS failing FROM transactions WHERE amount <= 0",
    ),
    Check(
        key="loans.outstanding.invalid", table="loans", column="outstanding_balance",
        dimension="Validity", severity="medium",
        rule="Outstanding balance cannot exceed the principal advanced",
        sql="SELECT count(*) AS failing FROM loans WHERE outstanding_balance > principal * 1.5",
    ),

    # --------------------------------------------------------- consistency --
    Check(
        key="accounts.status.invalid", table="accounts", column="status",
        dimension="Consistency", severity="medium",
        rule=f"Status must be one of {', '.join(VALID_ACCOUNT_STATUSES)}",
        sql=(f"SELECT count(*) AS failing FROM accounts "
             f"WHERE status IS NULL OR status NOT IN ({_q(VALID_ACCOUNT_STATUSES)})"),
        impact_sql=("SELECT DISTINCT customer_id FROM accounts WHERE status IS NULL OR status NOT IN ('Active', 'Dormant', 'Closed')")
    ),
    Check(
        key="products.product_type.inconsistent", table="products", column="product_type",
        dimension="Consistency", severity="medium",
        rule="Product type must use the canonical product vocabulary",
        sql=(f"SELECT count(*) AS failing FROM products "
             f"WHERE product_type IS NULL OR product_type NOT IN ({_q(VALID_PRODUCT_TYPES)})"),
        impact_sql=("SELECT DISTINCT customer_id FROM products WHERE product_type IS NULL OR product_type NOT IN ('Current account', 'Savings', 'Investment', 'Personal loan', 'Asset finance', 'Card', 'Insurance', 'Digital wallet')")
    ),

    # ----------------------------------------------- referential integrity --
    Check(
        key="accounts.customer_id.orphan", table="accounts", column="customer_id",
        dimension="Integrity", severity="high",
        rule="Every account must belong to a known customer",
        sql=("SELECT count(*) AS failing FROM accounts a "
             "LEFT JOIN (SELECT DISTINCT customer_id FROM customers) c USING (customer_id) "
             "WHERE c.customer_id IS NULL"),
    ),
    Check(
        key="transactions.customer_id.orphan", table="transactions", column="customer_id",
        dimension="Integrity", severity="high",
        rule="Every transaction must belong to a known customer",
        sql=("SELECT count(*) AS failing FROM transactions t "
             "LEFT JOIN (SELECT DISTINCT customer_id FROM customers) c USING (customer_id) "
             "WHERE c.customer_id IS NULL"),
    ),
    Check(
        key="loans.customer_id.orphan", table="loans", column="customer_id",
        dimension="Integrity", severity="high",
        rule="Every loan must belong to a known customer",
        sql=("SELECT count(*) AS failing FROM loans l "
             "LEFT JOIN (SELECT DISTINCT customer_id FROM customers) c USING (customer_id) "
             "WHERE c.customer_id IS NULL"),
    ),

    # ---------------------------------------------------------- timeliness --
    Check(
        key="transactions.date.future", table="transactions", column="transaction_date",
        dimension="Timeliness", severity="high",
        rule=f"No transaction may be dated after the reporting close ({REPORTING_CLOSE})",
        sql=(f"SELECT count(*) AS failing FROM transactions "
             f"WHERE transaction_date >= DATE '{REPORTING_CLOSE}'"),
        impact_sql=("SELECT DISTINCT customer_id FROM transactions WHERE transaction_date >= DATE '2026-01-01'")
    ),
    Check(
        key="accounts.opening_date.future", table="accounts", column="opening_date",
        dimension="Timeliness", severity="medium",
        rule="An account cannot be opened after the reporting close",
        sql=(f"SELECT count(*) AS failing FROM accounts "
             f"WHERE opening_date >= DATE '{REPORTING_CLOSE}'"),
    ),

    # ------------------------------------------------------------ accuracy --
    Check(
        key="transactions.amount.anomaly", table="transactions", column="amount",
        dimension="Accuracy", severity="high",
        rule=f"Transaction amount beyond the plausible range (> {AMOUNT_ANOMALY_THRESHOLD:,})",
        sql=(f"SELECT count(*) AS failing FROM transactions "
             f"WHERE amount > {AMOUNT_ANOMALY_THRESHOLD}"),
        impact_sql=("SELECT DISTINCT customer_id FROM transactions WHERE amount > 5000000")
    ),
)


def run_checks(con, checks: tuple[Check, ...] = CHECKS) -> pd.DataFrame:
    """Execute every check and return one row per check.

    ``failing`` is the number of offending rows; ``total`` the table's row count;
    ``pass_rate`` the share of rows that satisfy the rule.
    """
    totals: dict[str, int] = {}
    rows = []
    for check in checks:
        if check.table not in totals:
            totals[check.table] = int(con.execute(f"SELECT count(*) FROM {check.table}").fetchone()[0])
        total = totals[check.table]
        failing = int(con.execute(check.sql).fetchone()[0] or 0)
        rows.append({
            "key": check.key,
            "table": check.table,
            "column": check.column,
            "dimension": check.dimension,
            "severity": check.severity,
            "rule": check.rule,
            "failing": failing,
            "total": total,
            "pass_rate": 1.0 if total == 0 else 1.0 - failing / total,
            "status": "Pass" if failing == 0 else ("Fail" if check.severity == "high" else "Warn"),
        })
    return pd.DataFrame(rows)

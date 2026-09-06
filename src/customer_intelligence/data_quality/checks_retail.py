"""Quality rules for the Online Retail II schema.

Nothing here was injected. Every rule below fires on defects that are really in
a published, widely-used dataset, which is the difference between demonstrating
a quality framework and demonstrating that one was needed:

  * 243,007 invoice lines with no customer identifier;
  * 34,335 exact duplicate rows;
  * 22,950 negative quantities (returns booked through the sales table);
  * 6,207 lines priced at zero or below;
  * 4,382 lines with no description;
  * a row described as "This is a test product";
  * an Amazon commission of -£260,764 and a bad-debt write-off of -£147,614,
    both booked as ordinary invoice lines;
  * the same manual-adjustment code entered as both `M` and `m`.

Because the answer is not known in advance, these rules cannot be graded the way
the synthetic harness grades its own. That is what checks_synthetic.py is for:
it proves the framework detects what it claims to, against defects whose counts
are known, before the same framework is pointed at data where they are not.
"""

from __future__ import annotations

from .checks import Check

# The panel is 24 whole months. December 2011 is a partial month in the source
# and is excluded upstream; anything dated into it here is out of window.
REPORTING_CLOSE = "2011-12-01"
PANEL_OPEN = "2009-12-01"

# Three orders of magnitude above the median line value. A fixed, documented
# threshold rather than a quantile, so the rule does not move when the data does.
LINE_VALUE_ANOMALY = 50_000

VALID_LINE_TYPES = (
    "Product", "Postage", "Carriage", "Manual adjustment", "Discount", "Samples",
    "Bad debt adjustment", "Bank charges", "Marketplace commission",
    "Charity commission", "Test data", "Gift voucher",
)


def _q(values) -> str:
    return ", ".join(f"'{v}'" for v in values)


RETAIL_CHECKS: tuple[Check, ...] = (
    # ------------------------------------------------------- completeness --
    Check(
        key="transactions.customer_id.missing", table="transactions", column="customer_id",
        dimension="Completeness", severity="high",
        rule="Every invoice line should be attributable to a customer",
        sql="SELECT count(*) AS failing FROM transactions WHERE customer_id IS NULL",
    ),
    Check(
        key="transactions.description.missing", table="transactions", column="description",
        dimension="Completeness", severity="medium",
        rule="Every line should name what was sold",
        sql="SELECT count(*) AS failing FROM transactions WHERE description IS NULL",
        impact_sql=("SELECT DISTINCT customer_id FROM transactions "
                    "WHERE description IS NULL AND customer_id IS NOT NULL"),
    ),
    Check(
        key="products.description.missing", table="products", column="description",
        dimension="Completeness", severity="low",
        rule="Every stock code should carry a description",
        sql="SELECT count(*) AS failing FROM products WHERE description IS NULL",
    ),
    Check(
        key="customers.country.missing", table="customers", column="country",
        dimension="Completeness", severity="low",
        rule="Every customer should have a billing country",
        sql="SELECT count(*) AS failing FROM customers WHERE country IS NULL",
    ),

    # ---------------------------------------------------------- uniqueness --
    Check(
        key="transactions.duplicate_lines", table="transactions", column="line_id",
        dimension="Uniqueness", severity="high",
        rule="An identical line should not be posted twice to the same invoice",
        sql=("SELECT coalesce(sum(n - 1), 0) AS failing FROM ("
             "  SELECT invoice_id, stock_code, quantity, unit_price, invoice_date, "
             "         count(*) AS n FROM transactions "
             "  GROUP BY 1,2,3,4,5 HAVING count(*) > 1)"),
        impact_sql=("SELECT DISTINCT t.customer_id FROM transactions t JOIN ("
                    "  SELECT invoice_id, stock_code, quantity, unit_price, invoice_date "
                    "  FROM transactions GROUP BY 1,2,3,4,5 HAVING count(*) > 1) d "
                    "USING (invoice_id, stock_code, quantity, unit_price, invoice_date) "
                    "WHERE t.customer_id IS NOT NULL"),
    ),
    Check(
        key="customers.customer_id.duplicate", table="customers", column="customer_id",
        dimension="Uniqueness", severity="high",
        rule="customer_id must identify exactly one account",
        sql=("SELECT coalesce(sum(n - 1), 0) AS failing FROM ("
             "  SELECT customer_id, count(*) AS n FROM customers GROUP BY 1 HAVING count(*) > 1)"),
    ),
    Check(
        key="products.stock_code.duplicate", table="products", column="stock_code",
        dimension="Uniqueness", severity="medium",
        rule="stock_code must identify exactly one product",
        sql=("SELECT coalesce(sum(n - 1), 0) AS failing FROM ("
             "  SELECT stock_code, count(*) AS n FROM products GROUP BY 1 HAVING count(*) > 1)"),
    ),

    # ------------------------------------------------------------ validity --
    Check(
        key="transactions.unit_price.nonpositive", table="transactions", column="unit_price",
        dimension="Validity", severity="high",
        rule="A sale line must carry a positive unit price",
        sql=("SELECT count(*) AS failing FROM transactions "
             "WHERE unit_price <= 0 AND NOT is_return"),
        impact_sql=("SELECT DISTINCT customer_id FROM transactions "
                    "WHERE unit_price <= 0 AND NOT is_return AND customer_id IS NOT NULL"),
    ),
    Check(
        key="transactions.quantity.negative_on_sale", table="transactions", column="quantity",
        dimension="Validity", severity="high",
        rule="A negative quantity must be a credit note, not a sales invoice",
        sql=("SELECT count(*) AS failing FROM transactions "
             "WHERE quantity < 0 AND NOT is_return"),
        impact_sql=("SELECT DISTINCT customer_id FROM transactions "
                    "WHERE quantity < 0 AND NOT is_return AND customer_id IS NOT NULL"),
    ),
    Check(
        key="transactions.quantity.zero", table="transactions", column="quantity",
        dimension="Validity", severity="medium",
        rule="A line with zero quantity moves no stock and no money",
        sql="SELECT count(*) AS failing FROM transactions WHERE quantity = 0",
    ),

    # --------------------------------------------------------- consistency --
    Check(
        key="transactions.line_type.invalid", table="transactions", column="line_type",
        dimension="Consistency", severity="medium",
        rule=f"Line type must be one of the {len(VALID_LINE_TYPES)} classified types",
        sql=(f"SELECT count(*) AS failing FROM transactions "
             f"WHERE line_type IS NULL OR line_type NOT IN ({_q(VALID_LINE_TYPES)})"),
    ),
    Check(
        key="transactions.test_rows", table="transactions", column="stock_code",
        dimension="Consistency", severity="high",
        rule="Test rows must not be present in a production extract",
        sql="SELECT count(*) AS failing FROM transactions WHERE line_type = 'Test data'",
    ),
    Check(
        key="products.description.inconsistent", table="products", column="description",
        dimension="Consistency", severity="low",
        rule="One stock code should map to one product description",
        sql=("SELECT count(*) AS failing FROM ("
             "  SELECT stock_code FROM transactions WHERE description IS NOT NULL "
             "  GROUP BY stock_code HAVING count(DISTINCT description) > 1)"),
    ),
    Check(
        key="customers.country.inconsistent", table="customers", column="country",
        dimension="Consistency", severity="low",
        rule="A customer account should invoice to a single country",
        sql="SELECT count(*) AS failing FROM customers WHERE countries_seen > 1",
        impact_sql="SELECT DISTINCT customer_id FROM customers WHERE countries_seen > 1",
    ),

    # ----------------------------------------------- referential integrity --
    Check(
        key="transactions.stock_code.orphan", table="transactions", column="stock_code",
        dimension="Integrity", severity="high",
        rule="Every product line must reference a known stock code",
        sql=("SELECT count(*) AS failing FROM transactions t "
             "LEFT JOIN products p USING (stock_code) "
             "WHERE t.is_product AND p.stock_code IS NULL"),
    ),
    Check(
        key="transactions.customer_id.orphan", table="transactions", column="customer_id",
        dimension="Integrity", severity="high",
        rule="Every attributed line must reference a known customer",
        sql=("SELECT count(*) AS failing FROM transactions t "
             "LEFT JOIN customers c USING (customer_id) "
             "WHERE t.customer_id IS NOT NULL AND c.customer_id IS NULL"),
    ),
    Check(
        key="invoices.customer_id.orphan", table="invoices", column="customer_id",
        dimension="Integrity", severity="medium",
        rule="Every attributed invoice must reference a known customer",
        sql=("SELECT count(*) AS failing FROM invoices i "
             "LEFT JOIN customers c USING (customer_id) "
             "WHERE i.customer_id IS NOT NULL AND c.customer_id IS NULL"),
    ),

    # ---------------------------------------------------------- timeliness --
    Check(
        key="transactions.date.out_of_window", table="transactions", column="invoice_date",
        dimension="Timeliness", severity="high",
        rule=f"Every line must fall inside the panel ({PANEL_OPEN} to {REPORTING_CLOSE})",
        sql=(f"SELECT count(*) AS failing FROM transactions "
             f"WHERE invoice_date < DATE '{PANEL_OPEN}' "
             f"   OR invoice_date >= DATE '{REPORTING_CLOSE}'"),
    ),
    Check(
        key="transactions.month_index.invalid", table="transactions", column="month_index",
        dimension="Timeliness", severity="medium",
        rule="Derived month index must fall within the 24-month panel",
        sql=("SELECT count(*) AS failing FROM transactions "
             "WHERE month_index < 1 OR month_index > 24"),
    ),

    # ------------------------------------------------------------ accuracy --
    Check(
        key="transactions.line_value.anomaly", table="transactions", column="line_value",
        dimension="Accuracy", severity="high",
        rule=f"Line value beyond the plausible range (|value| > {LINE_VALUE_ANOMALY:,})",
        sql=(f"SELECT count(*) AS failing FROM transactions "
             f"WHERE abs(line_value) > {LINE_VALUE_ANOMALY}"),
        impact_sql=(f"SELECT DISTINCT customer_id FROM transactions "
                    f"WHERE abs(line_value) > {LINE_VALUE_ANOMALY} AND customer_id IS NOT NULL"),
    ),
    Check(
        key="transactions.quantity.anomaly", table="transactions", column="quantity",
        dimension="Accuracy", severity="medium",
        rule="Order quantity beyond any plausible wholesale line (> 20,000 units)",
        sql="SELECT count(*) AS failing FROM transactions WHERE abs(quantity) > 20000",
    ),
    Check(
        key="transactions.negative_revenue_lines", table="transactions", column="line_value",
        dimension="Accuracy", severity="medium",
        rule="Write-offs and commissions should not be booked as invoice lines",
        sql=("SELECT count(*) AS failing FROM transactions WHERE line_type IN "
             "('Bad debt adjustment', 'Marketplace commission', 'Bank charges')"),
    ),
)

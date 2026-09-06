-- =============================================================================
-- RETAIL 02 — MONTHLY CUSTOMER METRICS
-- One row per customer per month across the whole panel, including the months
-- in which a customer bought nothing.
--
-- The spine is the point. A wholesale customer who stops ordering simply stops
-- appearing in an invoice table, and an aggregate built from invoices alone
-- would drop exactly the customers the lapse model exists to find. Silence has
-- to be recorded to be modelled.
-- =============================================================================

CREATE OR REPLACE TABLE monthly_customer_metrics AS
WITH months AS (
    SELECT DISTINCT month_index, month FROM conformed_transactions
),
spine AS (
    SELECT c.customer_id, m.month_index, m.month
    FROM conformed_customers c
    CROSS JOIN months m
    -- Only from the month the relationship actually began. Carrying a customer
    -- back to the start of the panel would invent a year of silence and make
    -- every new account look like it was lapsing.
    WHERE m.month_index >= c.first_month_index
),
sales AS (
    SELECT
        s.customer_id, s.month_index,
        count(DISTINCT s.invoice_id)          AS invoices,
        count(*)                              AS lines,
        sum(s.quantity)                       AS units,
        sum(s.line_value)                     AS revenue,
        avg(s.unit_price)                     AS avg_unit_price,
        count(DISTINCT s.stock_code)          AS distinct_products,
        count(DISTINCT p.category)            AS distinct_categories
    FROM conformed_sales s
    LEFT JOIN conformed_products p ON s.stock_code = p.stock_code
    GROUP BY s.customer_id, s.month_index
),
returns AS (
    SELECT customer_id, month_index,
           count(*)                  AS return_lines,
           abs(sum(line_value))      AS return_value
    FROM conformed_returns GROUP BY customer_id, month_index
),
charges AS (
    SELECT customer_id, month_index,
           sum(CASE WHEN line_type IN ('Postage','Carriage') THEN line_value ELSE 0 END) AS postage,
           sum(CASE WHEN line_type = 'Discount' THEN abs(line_value) ELSE 0 END)         AS discounts
    FROM conformed_charges GROUP BY customer_id, month_index
)
SELECT
    sp.customer_id,
    sp.month_index,
    sp.month,
    coalesce(s.invoices, 0)            AS invoices,
    coalesce(s.lines, 0)               AS lines,
    coalesce(s.units, 0)               AS units,
    coalesce(s.revenue, 0)             AS revenue,
    s.avg_unit_price,
    coalesce(s.distinct_products, 0)   AS distinct_products,
    coalesce(s.distinct_categories, 0) AS distinct_categories,
    coalesce(r.return_lines, 0)        AS return_lines,
    coalesce(r.return_value, 0)        AS return_value,
    coalesce(ch.postage, 0)            AS postage,
    coalesce(ch.discounts, 0)          AS discounts,
    (coalesce(s.invoices, 0) > 0)      AS was_active
FROM spine sp
LEFT JOIN sales   s  ON sp.customer_id = s.customer_id  AND sp.month_index = s.month_index
LEFT JOIN returns r  ON sp.customer_id = r.customer_id  AND sp.month_index = r.month_index
LEFT JOIN charges ch ON sp.customer_id = ch.customer_id AND sp.month_index = ch.month_index;

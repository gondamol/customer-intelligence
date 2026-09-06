-- =============================================================================
-- RETAIL 01 — CONFORMED LAYER
-- Online Retail II, as landed, into something analysis can stand on.
--
-- Every rule below is a decision about a defect that is REALLY IN THE DATA.
-- None of it was injected. src/customer_intelligence/analytics/build.py counts
-- what each rule removed and writes it to the conformance log the application
-- shows, so nothing is dropped silently.
-- =============================================================================

CREATE OR REPLACE TABLE panel_origin AS SELECT DATE '2009-12-01' AS month_1;

-- -----------------------------------------------------------------------------
-- transactions — the decisions, in order of how much they matter
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE conformed_transactions AS
WITH deduplicated AS (
    -- 34,335 rows are exact duplicates across every field. In an invoice-line
    -- extract that is a re-run of a load, not a customer buying the same item
    -- twice on the same second at the same price -- those would share an
    -- invoice but differ in line. Keep one of each.
    SELECT *, row_number() OVER (
        PARTITION BY invoice_id, stock_code, quantity, unit_price, invoice_date
        ORDER BY line_id
    ) AS rn
    FROM transactions
)
SELECT
    line_id,
    invoice_id,
    customer_id,
    stock_code,
    description,
    quantity,
    unit_price,
    line_value,
    invoice_date,
    month,
    month_index,
    country,
    is_return,
    is_product,
    line_type
FROM deduplicated
WHERE rn = 1
  -- 243,007 lines carry no customer identifier. They are real revenue and are
  -- kept in the revenue tables, but they cannot be attributed to a
  -- relationship, so they cannot enter a customer-level model. Excluded here
  -- and counted in the conformance log rather than quietly ignored.
  AND customer_id IS NOT NULL
  -- A zero or negative unit price on a sale line is not a price. These are
  -- stock adjustments and write-offs that were booked through the same table.
  AND unit_price > 0
  AND description IS NOT NULL;

-- -----------------------------------------------------------------------------
-- sales vs returns
--
-- Returns are not negative sales to be netted off and forgotten. A customer who
-- buys £10,000 and returns £9,000 is a different relationship from one who buys
-- £1,000 and returns nothing, and netting makes them identical. Sales and
-- returns are therefore separated, and the return rate becomes a feature.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE conformed_sales AS
SELECT * FROM conformed_transactions
WHERE NOT is_return AND quantity > 0 AND is_product;

CREATE OR REPLACE TABLE conformed_returns AS
SELECT * FROM conformed_transactions
WHERE (is_return OR quantity < 0) AND is_product;

-- Postage, carriage, manual adjustments, discounts, bank charges, marketplace
-- commission, bad debt, gift vouchers and test rows. Real money, not products.
CREATE OR REPLACE TABLE conformed_charges AS
SELECT * FROM conformed_transactions WHERE NOT is_product;

-- -----------------------------------------------------------------------------
-- products — the derived category taxonomy joins here
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE conformed_products AS
SELECT
    stock_code, description, category, median_price,
    lines, units, revenue, first_month, last_month
FROM products;

-- -----------------------------------------------------------------------------
-- customers — one row per identified account
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE conformed_customers AS
SELECT
    c.customer_id,
    c.country,
    c.countries_seen,
    c.first_invoice_date,
    c.first_month_index,
    c.last_month_index,
    -- A handful of accounts invoice to more than one country. Flagged rather
    -- than resolved: without a source of truth, picking one would be a guess.
    (c.countries_seen > 1) AS multi_country
FROM customers c
WHERE c.customer_id IS NOT NULL;

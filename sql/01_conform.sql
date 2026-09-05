-- =============================================================================
-- 01 — CONFORMED LAYER
-- Raw, as-landed data in; trustworthy, analysis-ready tables out.
--
-- Every rule here is a decision about what to do with a defect the quality
-- checks found. Nothing is silently discarded: src/customer_intelligence/
-- analytics/clean.py counts what each rule removed and writes it to the
-- conformance log shown in the application.
--
-- Portability: plain ANSI SQL apart from the parquet views created by io.connect().
-- The analysis window is a one-row table rather than an engine-specific
-- variable, so the same statements run on DuckDB or PostgreSQL.
-- =============================================================================

-- The month index origin. Month 1 is the first month of the panel.
CREATE OR REPLACE TABLE panel_origin AS SELECT DATE '2024-10-01' AS month_1;

-- -----------------------------------------------------------------------------
-- customers — drop keyless rows, de-duplicate, quarantine impossible ages
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE conformed_customers AS
WITH deduplicated AS (
    SELECT *, row_number() OVER (PARTITION BY customer_id ORDER BY customer_id) AS rn
    FROM customers
    WHERE customer_id IS NOT NULL          -- a row with no key cannot be used
)
SELECT
    customer_id,
    -- An impossible age is set to NULL rather than guessed at. Downstream
    -- consumers see missing, not a fabricated value.
    CASE WHEN age BETWEEN 18 AND 100 THEN age END        AS age,
    gender,
    coalesce(region, 'Unknown')                          AS region,
    employment_type,
    coalesce(income_band, 'Unknown')                     AS income_band,
    monthly_income,
    tenure_months,
    customer_segment                                     AS declared_segment,
    join_date,
    (age IS NULL OR age < 18 OR age > 100)               AS age_was_invalid,
    (region IS NULL)                                     AS region_was_missing,
    (income_band IS NULL)                                AS income_band_was_missing
FROM deduplicated
WHERE rn = 1;

-- -----------------------------------------------------------------------------
-- accounts — drop orphans, standardise status, null impossible balances
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE conformed_accounts AS
SELECT
    a.account_id,
    a.customer_id,
    a.account_type,
    a.opening_date,
    CASE
        WHEN upper(trim(a.status)) IN ('ACTIVE', 'A')  THEN 'Active'
        WHEN upper(trim(a.status)) IN ('DORMANT', 'D') THEN 'Dormant'
        WHEN upper(trim(a.status)) IN ('CLOSED', 'C')  THEN 'Closed'
        ELSE 'Unknown'
    END                                                  AS status,
    a.average_balance,
    -- A negative balance on a deposit product is not a real overdraft; it is a
    -- sign error. Excluded from value measures rather than counted as debt.
    CASE
        WHEN a.account_type IN ('Savings', 'Fixed deposit') AND a.current_balance < 0
        THEN NULL ELSE a.current_balance
    END                                                  AS current_balance
FROM accounts a
INNER JOIN conformed_customers c ON a.customer_id = c.customer_id;

-- -----------------------------------------------------------------------------
-- transactions — de-duplicate, drop future dates, quarantine anomalies
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE conformed_transactions AS
WITH deduplicated AS (
    SELECT *, row_number() OVER (PARTITION BY transaction_id ORDER BY transaction_date) AS rn
    FROM transactions
)
SELECT
    t.transaction_id,
    t.customer_id,
    t.account_id,
    t.transaction_date,
    (date_diff('month', p.month_1, t.transaction_date) + 1) AS month_index,
    t.transaction_type,
    t.amount,
    t.channel,
    t.merchant_category,
    t.channel IN ('Mobile app', 'Internet banking')        AS is_digital
FROM deduplicated t
CROSS JOIN panel_origin p
INNER JOIN conformed_customers c ON t.customer_id = c.customer_id
WHERE t.rn = 1                                  -- one posting per transaction
  AND t.amount IS NOT NULL                      -- no amount, no measurable value
  AND t.amount > 0
  AND t.amount <= 5000000                       -- anomalies quarantined, not capped
  AND t.transaction_date < DATE '2026-01-01';   -- nothing after the reporting close

-- -----------------------------------------------------------------------------
-- products — map source spellings onto the canonical vocabulary
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE conformed_products AS
SELECT
    pr.customer_id,
    CASE lower(trim(pr.product_type))
        WHEN 'current_acct'   THEN 'Current account'
        WHEN 'savings'        THEN 'Savings'
        WHEN 'investment'     THEN 'Investment'
        WHEN 'pers-loan'      THEN 'Personal loan'
        WHEN 'asset_finance'  THEN 'Asset finance'
        WHEN 'card'           THEN 'Card'
        WHEN 'ins'            THEN 'Insurance'
        WHEN 'wallet'         THEN 'Digital wallet'
        ELSE pr.product_type
    END                                                    AS product_type,
    pr.start_date,
    pr.start_month_index,
    pr.status
FROM products pr
INNER JOIN conformed_customers c ON pr.customer_id = c.customer_id;

-- -----------------------------------------------------------------------------
-- loans / digital activity / service interactions — integrity filter only
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE conformed_loans AS
SELECT
    l.*,
    (date_diff('month', p.month_1, l.origination_date) + 1) AS origination_month_index,
    l.repayment_status IN ('Late 1-30', 'Late 31-90', 'Default') AS in_arrears
FROM loans l
CROSS JOIN panel_origin p
INNER JOIN conformed_customers c ON l.customer_id = c.customer_id;

CREATE OR REPLACE TABLE conformed_digital_activity AS
SELECT d.*
FROM digital_activity d
INNER JOIN conformed_customers c ON d.customer_id = c.customer_id;

CREATE OR REPLACE TABLE conformed_service_interactions AS
SELECT
    s.*,
    (date_diff('month', p.month_1, s.interaction_date) + 1) AS month_index
FROM service_interactions s
CROSS JOIN panel_origin p
INNER JOIN conformed_customers c ON s.customer_id = c.customer_id;

-- -----------------------------------------------------------------------------
-- account monthly balances — integrity filter, inherits the account cleanse
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE conformed_account_balances AS
SELECT b.*
FROM account_monthly_balances b
INNER JOIN conformed_accounts a ON b.account_id = a.account_id;

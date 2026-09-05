-- =============================================================================
-- 04 — BUSINESS ANALYTICS
-- The aggregate questions an organisation asks of a customer book. Kept in SQL
-- rather than pandas because these are the queries an analyst would be handed
-- and asked to change, and SQL is where that conversation happens.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Product penetration: how deep the relationship goes, by segment
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE product_penetration AS
WITH base AS (SELECT count(*) AS total_customers FROM conformed_customers)
SELECT
    p.product_type,
    count(DISTINCT p.customer_id)                                        AS customers,
    count(DISTINCT p.customer_id) * 100.0 / max(b.total_customers)       AS penetration_pct
FROM conformed_products p
CROSS JOIN base b
WHERE p.status = 'Active'
GROUP BY p.product_type
ORDER BY customers DESC;

-- -----------------------------------------------------------------------------
-- Monthly balance and activity trend across the whole book
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE monthly_book_trend AS
SELECT
    month_index,
    min(month)                                                AS month,
    count(*)                                                  AS customers,
    sum(closing_balance)                                      AS total_balance,
    avg(closing_balance)                                      AS avg_balance,
    sum(transaction_count)                                    AS transactions,
    sum(transaction_value)                                    AS transaction_value,
    sum(CASE WHEN was_active THEN 1 ELSE 0 END)               AS active_customers,
    sum(CASE WHEN was_active THEN 1 ELSE 0 END) * 100.0 / count(*) AS active_pct,
    sum(complaints)                                           AS complaints
FROM monthly_customer_metrics
GROUP BY month_index
ORDER BY month_index;

-- -----------------------------------------------------------------------------
-- Channel mix: where activity actually happens
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE channel_mix AS
SELECT
    channel,
    count(*)                                              AS transactions,
    sum(amount)                                           AS value,
    avg(amount)                                           AS avg_value,
    count(DISTINCT customer_id)                           AS customers,
    count(*) * 100.0 / sum(count(*)) OVER ()              AS share_of_transactions
FROM conformed_transactions
GROUP BY channel
ORDER BY transactions DESC;

-- -----------------------------------------------------------------------------
-- Credit quality: exposure and arrears by loan type
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE credit_quality AS
SELECT
    loan_type,
    count(*)                                                     AS facilities,
    sum(outstanding_balance)                                     AS exposure,
    avg(interest_rate)                                           AS avg_rate,
    sum(CASE WHEN in_arrears THEN 1 ELSE 0 END)                  AS in_arrears,
    sum(CASE WHEN in_arrears THEN 1 ELSE 0 END) * 100.0 / count(*) AS arrears_rate,
    sum(CASE WHEN in_arrears THEN outstanding_balance ELSE 0 END) AS exposure_in_arrears
FROM conformed_loans
GROUP BY loan_type
ORDER BY exposure DESC;

-- -----------------------------------------------------------------------------
-- Activity cohorts: how engagement varies with tenure
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE tenure_cohorts AS
SELECT
    CASE
        WHEN c.tenure_months < 12  THEN '0–11 months'
        WHEN c.tenure_months < 36  THEN '1–2 years'
        WHEN c.tenure_months < 60  THEN '3–4 years'
        WHEN c.tenure_months < 120 THEN '5–9 years'
        ELSE '10+ years'
    END                                                     AS tenure_cohort,
    count(*)                                                AS customers,
    avg(m.avg_balance)                                      AS avg_balance,
    avg(m.active_months)                                    AS avg_active_months,
    avg(m.total_transactions)                               AS avg_transactions
FROM conformed_customers c
JOIN (
    SELECT
        customer_id,
        avg(closing_balance)                        AS avg_balance,
        sum(CASE WHEN was_active THEN 1 ELSE 0 END) AS active_months,
        sum(transaction_count)                      AS total_transactions
    FROM monthly_customer_metrics
    GROUP BY customer_id
) m ON c.customer_id = m.customer_id
GROUP BY tenure_cohort
ORDER BY customers DESC;

-- -----------------------------------------------------------------------------
-- The churn feature set, expressed in SQL rather than pandas.
-- Illustrates that the whole feature layer is expressible in the warehouse --
-- which is where it would live in production, close to the data.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE churn_feature_preview AS
WITH w AS (SELECT month_from, month_to FROM analysis_window),
recent AS (
    SELECT m.customer_id,
           avg(m.closing_balance)   AS bal_recent,
           avg(m.transaction_count) AS txn_recent,
           avg(m.total_logins)      AS log_recent
    FROM monthly_customer_metrics m CROSS JOIN w
    WHERE m.month_index > w.month_to - 3 AND m.month_index <= w.month_to
    GROUP BY m.customer_id
),
early AS (
    SELECT m.customer_id,
           avg(m.closing_balance)   AS bal_early,
           avg(m.transaction_count) AS txn_early,
           avg(m.total_logins)      AS log_early
    FROM monthly_customer_metrics m CROSS JOIN w
    WHERE m.month_index >= w.month_from AND m.month_index <= w.month_from + 2
    GROUP BY m.customer_id
)
SELECT
    r.customer_id,
    r.bal_recent, e.bal_early,
    CASE WHEN e.bal_early > 0 THEN r.bal_recent / e.bal_early ELSE 1.0 END AS balance_trend,
    CASE WHEN e.txn_early > 0 THEN r.txn_recent / e.txn_early ELSE 1.0 END AS transaction_trend,
    CASE WHEN e.log_early > 0 THEN r.log_recent / e.log_early ELSE 1.0 END AS digital_trend
FROM recent r
JOIN early e ON r.customer_id = e.customer_id;

SELECT * FROM product_penetration;

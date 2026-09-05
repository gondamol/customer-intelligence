-- =============================================================================
-- 03 — CUSTOMER 360
-- One row per customer, computed over a bounded month window.
--
-- The window matters more than anything else in this file. Training features
-- come from months 1-12 and the outcome is observed in 13-15; scoring features
-- come from months 4-15 and the outcome is the unobserved future. Both are
-- produced by *this same SQL* with a different analysis_window, which is what
-- guarantees the training and scoring feature definitions cannot drift apart.
--
-- Nothing in this file may reference a month outside [month_from, month_to].
-- That is the leakage rule, and tests/test_leakage.py enforces it.
-- =============================================================================

CREATE OR REPLACE TABLE customer_360 AS
WITH w AS (SELECT month_from, month_to FROM analysis_window),

-- ---- behavioural aggregates over the window --------------------------------
panel AS (
    SELECT m.* FROM monthly_customer_metrics m CROSS JOIN w
    WHERE m.month_index BETWEEN w.month_from AND w.month_to
),
activity AS (
    SELECT
        customer_id,
        sum(transaction_count)                        AS transaction_count,
        sum(transaction_value)                        AS transaction_value,
        avg(transaction_count)                        AS avg_monthly_transactions,
        CASE WHEN sum(transaction_count) > 0
             THEN sum(transaction_value) / sum(transaction_count) END AS avg_transaction_value,
        avg(closing_balance)                          AS avg_monthly_balance,
        max(closing_balance)                          AS peak_balance,
        sum(inflow)                                   AS total_inflow,
        sum(outflow)                                  AS total_outflow,
        sum(total_logins)                             AS total_logins,
        avg(total_logins)                             AS avg_monthly_logins,
        sum(digital_transaction_count)                AS digital_transactions,
        sum(failed_logins)                            AS failed_logins,
        sum(service_interactions)                     AS service_interactions,
        sum(complaints)                               AS complaints,
        sum(unresolved_interactions)                  AS unresolved_interactions,
        avg(avg_satisfaction)                         AS avg_satisfaction,
        sum(CASE WHEN was_active THEN 1 ELSE 0 END)   AS active_months,
        max(CASE WHEN was_active THEN month_index END) AS last_active_month,
        max(channels_used)                            AS max_channels_used
    FROM panel GROUP BY customer_id
),

-- ---- trend: the last quarter of the window against the first ---------------
-- Direction of travel, not level. A high-balance customer in steep decline and
-- a low-balance customer growing steadily look identical on level alone.
edges AS (
    SELECT
        p.customer_id,
        avg(CASE WHEN p.month_index >  w.month_to - 3 THEN p.closing_balance END)   AS bal_recent,
        avg(CASE WHEN p.month_index <= w.month_from + 2 THEN p.closing_balance END)  AS bal_early,
        avg(CASE WHEN p.month_index >  w.month_to - 3 THEN p.transaction_count END) AS txn_recent,
        avg(CASE WHEN p.month_index <= w.month_from + 2 THEN p.transaction_count END) AS txn_early,
        avg(CASE WHEN p.month_index >  w.month_to - 3 THEN p.total_logins END)      AS log_recent,
        avg(CASE WHEN p.month_index <= w.month_from + 2 THEN p.total_logins END)     AS log_early,
        sum(CASE WHEN p.month_index >  w.month_to - 3 THEN p.complaints ELSE 0 END) AS complaints_recent
    FROM panel p CROSS JOIN w
    GROUP BY p.customer_id
),

-- ---- holdings as at the close of the window --------------------------------
prod AS (
    SELECT
        p.customer_id,
        count(*)                                                                AS product_count,
        max(CASE WHEN p.product_type = 'Investment'    THEN 1 ELSE 0 END)       AS has_investment,
        max(CASE WHEN p.product_type = 'Savings'       THEN 1 ELSE 0 END)       AS has_savings,
        max(CASE WHEN p.product_type IN ('Personal loan', 'Asset finance')
                 THEN 1 ELSE 0 END)                                             AS has_lending,
        max(CASE WHEN p.product_type = 'Card'          THEN 1 ELSE 0 END)       AS has_card,
        max(CASE WHEN p.product_type = 'Insurance'     THEN 1 ELSE 0 END)       AS has_insurance,
        max(CASE WHEN p.product_type = 'Digital wallet' THEN 1 ELSE 0 END)      AS has_wallet
    FROM conformed_products p CROSS JOIN w
    WHERE p.status = 'Active'
      AND p.start_month_index <= w.month_to     -- never count a future holding
    GROUP BY p.customer_id
),
acct AS (
    SELECT
        a.customer_id,
        count(*)                                                       AS account_count,
        sum(CASE WHEN a.status = 'Active' THEN 1 ELSE 0 END)           AS active_accounts,
        sum(CASE WHEN a.status = 'Dormant' THEN 1 ELSE 0 END)          AS dormant_accounts,
        count(DISTINCT a.account_type)                                 AS account_types_held
    FROM conformed_accounts a GROUP BY a.customer_id
),
loan AS (
    SELECT
        l.customer_id,
        count(*)                                                       AS loan_count,
        sum(l.outstanding_balance)                                     AS loan_exposure,
        avg(l.interest_rate)                                           AS avg_interest_rate,
        max(CASE WHEN l.in_arrears THEN 1 ELSE 0 END)                  AS ever_in_arrears,
        max(CASE WHEN l.repayment_status = 'Default' THEN 1 ELSE 0 END) AS ever_defaulted
    FROM conformed_loans l CROSS JOIN w
    WHERE l.origination_month_index <= w.month_to
    GROUP BY l.customer_id
)

SELECT
    c.customer_id,
    -- ---- who they are -------------------------------------------------------
    c.age, c.gender, c.region, c.employment_type, c.income_band, c.monthly_income,
    c.declared_segment,
    -- Tenure as at the close of the window, not as at today.
    c.tenure_months - (SELECT max(month_index) FROM monthly_customer_metrics)
                    + (SELECT month_to FROM w)                          AS tenure_months,

    -- ---- what they hold -----------------------------------------------------
    coalesce(ac.account_count, 0)        AS account_count,
    coalesce(ac.active_accounts, 0)      AS active_accounts,
    coalesce(ac.dormant_accounts, 0)     AS dormant_accounts,
    coalesce(ac.account_types_held, 0)   AS account_types_held,
    coalesce(pr.product_count, 0)        AS product_count,
    coalesce(pr.has_investment, 0)       AS has_investment,
    coalesce(pr.has_savings, 0)          AS has_savings,
    coalesce(pr.has_lending, 0)          AS has_lending,
    coalesce(pr.has_card, 0)             AS has_card,
    coalesce(pr.has_insurance, 0)        AS has_insurance,
    coalesce(pr.has_wallet, 0)           AS has_wallet,

    -- ---- what they do -------------------------------------------------------
    coalesce(a.avg_monthly_balance, 0)   AS avg_monthly_balance,
    coalesce(a.peak_balance, 0)          AS peak_balance,
    coalesce(a.transaction_count, 0)     AS transaction_count,
    coalesce(a.transaction_value, 0)     AS transaction_value,
    coalesce(a.avg_monthly_transactions, 0) AS avg_monthly_transactions,
    a.avg_transaction_value,
    coalesce(a.total_inflow, 0)          AS total_inflow,
    coalesce(a.total_outflow, 0)         AS total_outflow,
    -- Ratios, not levels. How much is held against income, and how much leaves
    -- against what arrives, say more about financial posture than either
    -- absolute figure does on its own.
    CASE WHEN c.monthly_income > 0
         THEN coalesce(a.avg_monthly_balance, 0) / c.monthly_income END        AS balance_to_income,
    CASE WHEN coalesce(a.total_inflow, 0) > 0
         THEN a.total_outflow / a.total_inflow END                             AS outflow_to_inflow,
    coalesce(a.total_logins, 0)          AS total_logins,
    coalesce(a.avg_monthly_logins, 0)    AS avg_monthly_logins,
    coalesce(a.digital_transactions, 0)  AS digital_transactions,
    coalesce(a.failed_logins, 0)         AS failed_logins,
    coalesce(a.max_channels_used, 0)     AS max_channels_used,
    CASE WHEN coalesce(a.transaction_count, 0) > 0
         THEN a.digital_transactions * 1.0 / a.transaction_count END   AS digital_share,
    coalesce(a.active_months, 0)         AS active_months,
    (SELECT month_to FROM w) - coalesce(a.last_active_month, (SELECT month_from FROM w) - 1)
                                         AS months_since_last_activity,

    -- ---- service experience -------------------------------------------------
    coalesce(a.service_interactions, 0)  AS service_interactions,
    coalesce(a.complaints, 0)            AS complaints,
    coalesce(a.unresolved_interactions, 0) AS unresolved_interactions,
    a.avg_satisfaction,

    -- ---- credit -------------------------------------------------------------
    coalesce(l.loan_count, 0)            AS loan_count,
    coalesce(l.loan_exposure, 0)         AS loan_exposure,
    l.avg_interest_rate,
    coalesce(l.ever_in_arrears, 0)       AS ever_in_arrears,
    coalesce(l.ever_defaulted, 0)        AS ever_defaulted,

    -- ---- direction of travel ------------------------------------------------
    -- Ratios of recent to early, floored so a customer going to zero produces
    -- 0.0 rather than a division error or a null the models must impute.
    CASE WHEN coalesce(e.bal_early, 0) > 0 THEN e.bal_recent / e.bal_early ELSE 1.0 END AS balance_trend,
    CASE WHEN coalesce(e.txn_early, 0) > 0 THEN e.txn_recent / e.txn_early ELSE 1.0 END AS transaction_trend,
    CASE WHEN coalesce(e.log_early, 0) > 0 THEN e.log_recent / e.log_early ELSE 1.0 END AS digital_trend,
    coalesce(e.complaints_recent, 0)     AS complaints_recent,

    (SELECT month_from FROM w)           AS window_from,
    (SELECT month_to FROM w)             AS window_to
FROM conformed_customers c
LEFT JOIN activity a  ON c.customer_id = a.customer_id
LEFT JOIN edges e     ON c.customer_id = e.customer_id
LEFT JOIN prod pr     ON c.customer_id = pr.customer_id
LEFT JOIN acct ac     ON c.customer_id = ac.customer_id
LEFT JOIN loan l      ON c.customer_id = l.customer_id;

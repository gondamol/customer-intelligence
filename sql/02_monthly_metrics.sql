-- =============================================================================
-- 02 — MONTHLY CUSTOMER METRICS
-- One row per customer per month: the analytical panel every feature is built
-- from. Derived, not landed -- which is why it lives here and not in raw.
-- =============================================================================

CREATE OR REPLACE TABLE monthly_customer_metrics AS
WITH months AS (
    SELECT DISTINCT month_index, month FROM conformed_digital_activity
),
spine AS (
    -- Every customer appears in every month, including months with no activity.
    -- Without the spine, a customer who stops transacting simply disappears and
    -- their silence -- the strongest churn signal there is -- is never recorded.
    SELECT c.customer_id, m.month_index, m.month
    FROM conformed_customers c
    CROSS JOIN months m
),
txn AS (
    SELECT
        customer_id, month_index,
        count(*)                                     AS transaction_count,
        sum(amount)                                  AS transaction_value,
        avg(amount)                                  AS avg_transaction_value,
        count(DISTINCT channel)                      AS channels_used,
        sum(CASE WHEN is_digital THEN 1 ELSE 0 END)  AS digital_transaction_count,
        sum(CASE WHEN transaction_type = 'Credit' THEN amount ELSE 0 END) AS inflow,
        sum(CASE WHEN transaction_type = 'Debit'  THEN amount ELSE 0 END) AS outflow
    FROM conformed_transactions
    GROUP BY customer_id, month_index
),
bal AS (
    SELECT customer_id, month_index, sum(closing_balance) AS closing_balance
    FROM conformed_account_balances
    GROUP BY customer_id, month_index
),
svc AS (
    SELECT
        customer_id, month_index,
        count(*)                                                          AS interactions,
        sum(CASE WHEN interaction_type = 'Complaint' THEN 1 ELSE 0 END)   AS complaints,
        sum(CASE WHEN resolution_status IN ('Unresolved', 'Escalated')
                 THEN 1 ELSE 0 END)                                       AS unresolved,
        avg(satisfaction_score)                                           AS avg_satisfaction
    FROM conformed_service_interactions
    GROUP BY customer_id, month_index
)
SELECT
    s.customer_id,
    s.month_index,
    s.month,
    coalesce(b.closing_balance, 0)            AS closing_balance,
    coalesce(t.transaction_count, 0)          AS transaction_count,
    coalesce(t.transaction_value, 0)          AS transaction_value,
    t.avg_transaction_value,
    coalesce(t.channels_used, 0)              AS channels_used,
    coalesce(t.digital_transaction_count, 0)  AS digital_transaction_count,
    coalesce(t.inflow, 0)                     AS inflow,
    coalesce(t.outflow, 0)                    AS outflow,
    coalesce(d.mobile_logins, 0)              AS mobile_logins,
    coalesce(d.online_logins, 0)              AS online_logins,
    coalesce(d.failed_logins, 0)              AS failed_logins,
    coalesce(d.mobile_logins, 0) + coalesce(d.online_logins, 0) AS total_logins,
    coalesce(v.interactions, 0)               AS service_interactions,
    coalesce(v.complaints, 0)                 AS complaints,
    coalesce(v.unresolved, 0)                 AS unresolved_interactions,
    v.avg_satisfaction,
    (coalesce(t.transaction_count, 0) > 0)    AS was_active
FROM spine s
LEFT JOIN bal b ON s.customer_id = b.customer_id AND s.month_index = b.month_index
LEFT JOIN txn t ON s.customer_id = t.customer_id AND s.month_index = t.month_index
LEFT JOIN conformed_digital_activity d ON s.customer_id = d.customer_id AND s.month_index = d.month_index
LEFT JOIN svc v ON s.customer_id = v.customer_id AND s.month_index = v.month_index;

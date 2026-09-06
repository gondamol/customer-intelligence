-- =============================================================================
-- RETAIL 03 — CUSTOMER 360
-- One row per customer, over a bounded month window.
--
-- The window rule is the same as it was on synthetic data and matters more here,
-- because the outcome is real: training features come from months 1–12 with the
-- outcome observed in 13–15, and scoring features come from months 13–24 with
-- the outcome unobserved. Both are produced by THIS FILE with a different
-- analysis_window, which is what stops the two definitions drifting apart.
--
-- Nothing here may reference a month outside [month_from, month_to].
-- tests/test_leakage.py enforces it on the text of this file.
-- =============================================================================

CREATE OR REPLACE TABLE customer_360 AS
WITH w AS (SELECT month_from, month_to FROM analysis_window),

panel AS (
    SELECT m.* FROM monthly_customer_metrics m CROSS JOIN w
    WHERE m.month_index BETWEEN w.month_from AND w.month_to
),

-- ---- RFM and the rest of the behavioural aggregate ------------------------
activity AS (
    SELECT
        customer_id,
        sum(invoices)                                  AS invoices,
        sum(lines)                                     AS lines,
        sum(units)                                     AS units,
        sum(revenue)                                   AS revenue,
        avg(revenue)                                   AS avg_monthly_revenue,
        max(revenue)                                   AS peak_monthly_revenue,
        avg(avg_unit_price)                            AS avg_unit_price,
        sum(return_lines)                              AS return_lines,
        sum(return_value)                              AS return_value,
        sum(postage)                                   AS postage,
        sum(discounts)                                 AS discounts,
        sum(CASE WHEN was_active THEN 1 ELSE 0 END)    AS active_months,
        max(CASE WHEN was_active THEN month_index END) AS last_active_month,
        min(CASE WHEN was_active THEN month_index END) AS first_active_month
    FROM panel GROUP BY customer_id
),

-- ---- breadth, counted over the window rather than over all time ----------
breadth AS (
    SELECT
        s.customer_id,
        count(DISTINCT s.stock_code)   AS distinct_products,
        count(DISTINCT p.category)     AS distinct_categories,
        count(DISTINCT s.invoice_id)   AS distinct_invoices
    FROM conformed_sales s
    CROSS JOIN w
    LEFT JOIN conformed_products p ON s.stock_code = p.stock_code
    WHERE s.month_index BETWEEN w.month_from AND w.month_to
    GROUP BY s.customer_id
),

-- ---- category mix: what this account actually buys ------------------------
-- Shares, not amounts. A large account and a small one that buy the same mix
-- should look alike on mix and differ on value, and only shares do that.
category_mix AS (
    SELECT
        s.customer_id,
        sum(CASE WHEN p.category = 'Christmas & seasonal'    THEN s.line_value ELSE 0 END) AS cat_christmas,
        sum(CASE WHEN p.category = 'Home décor'              THEN s.line_value ELSE 0 END) AS cat_home,
        sum(CASE WHEN p.category = 'Kitchen & dining'        THEN s.line_value ELSE 0 END) AS cat_kitchen,
        sum(CASE WHEN p.category = 'Bags & luggage'          THEN s.line_value ELSE 0 END) AS cat_bags,
        sum(CASE WHEN p.category = 'Lighting & candles'      THEN s.line_value ELSE 0 END) AS cat_lighting,
        sum(CASE WHEN p.category = 'Party & celebration'     THEN s.line_value ELSE 0 END) AS cat_party,
        sum(CASE WHEN p.category = 'Stationery & wrap'       THEN s.line_value ELSE 0 END) AS cat_stationery,
        sum(CASE WHEN p.category = 'Storage & household'     THEN s.line_value ELSE 0 END) AS cat_storage,
        sum(CASE WHEN p.category = 'Garden & outdoor'        THEN s.line_value ELSE 0 END) AS cat_garden,
        sum(CASE WHEN p.category = 'Toys & games'            THEN s.line_value ELSE 0 END) AS cat_toys,
        sum(CASE WHEN p.category = 'Jewellery & accessories' THEN s.line_value ELSE 0 END) AS cat_jewellery,
        sum(s.line_value)                                                                  AS cat_total
    FROM conformed_sales s
    CROSS JOIN w
    LEFT JOIN conformed_products p ON s.stock_code = p.stock_code
    WHERE s.month_index BETWEEN w.month_from AND w.month_to
    GROUP BY s.customer_id
),

-- ---- direction of travel: the last quarter against the first -------------
edges AS (
    SELECT
        p.customer_id,
        avg(CASE WHEN p.month_index >  w.month_to - 3   THEN p.revenue END)  AS rev_recent,
        avg(CASE WHEN p.month_index <= w.month_from + 2 THEN p.revenue END)  AS rev_early,
        avg(CASE WHEN p.month_index >  w.month_to - 3   THEN p.invoices END) AS inv_recent,
        avg(CASE WHEN p.month_index <= w.month_from + 2 THEN p.invoices END) AS inv_early,
        sum(CASE WHEN p.month_index >  w.month_to - 3   THEN p.revenue ELSE 0 END) AS revenue_last_quarter
    FROM panel p CROSS JOIN w
    GROUP BY p.customer_id
),

-- ---- ordering rhythm ------------------------------------------------------
-- A wholesale account has a cadence. Whether it has slipped past its own usual
-- gap says more than an absolute recency figure, because a quarterly buyer and
-- a weekly buyer who both last ordered six weeks ago are in different states.
cadence AS (
    SELECT
        customer_id,
        avg(gap)     AS avg_months_between_orders,
        max(gap)     AS max_months_between_orders,
        stddev(gap)  AS gap_variability
    FROM (
        SELECT
            customer_id,
            month_index - lag(month_index) OVER (
                PARTITION BY customer_id ORDER BY month_index
            ) AS gap
        FROM panel WHERE was_active
    ) g
    WHERE gap IS NOT NULL
    GROUP BY customer_id
)

SELECT
    c.customer_id,
    c.country,
    c.multi_country,

    -- ---- recency, frequency, monetary --------------------------------------
    (SELECT month_to FROM w) - coalesce(a.last_active_month, (SELECT month_from FROM w) - 1)
                                                AS months_since_last_order,
    coalesce(a.active_months, 0)                AS active_months,
    coalesce(a.invoices, 0)                     AS invoices,
    coalesce(a.revenue, 0)                      AS revenue,
    coalesce(a.avg_monthly_revenue, 0)          AS avg_monthly_revenue,
    coalesce(a.peak_monthly_revenue, 0)         AS peak_monthly_revenue,
    CASE WHEN coalesce(a.invoices, 0) > 0 THEN a.revenue / a.invoices END AS avg_order_value,
    coalesce(a.units, 0)                        AS units,
    coalesce(a.lines, 0)                        AS lines,
    CASE WHEN coalesce(a.invoices, 0) > 0 THEN a.lines * 1.0 / a.invoices END AS lines_per_order,
    a.avg_unit_price,

    -- ---- tenure within the window ------------------------------------------
    (SELECT month_to FROM w) - greatest(c.first_month_index, (SELECT month_from FROM w)) + 1
                                                AS months_on_book,

    -- ---- breadth ------------------------------------------------------------
    coalesce(b.distinct_products, 0)            AS distinct_products,
    coalesce(b.distinct_categories, 0)          AS distinct_categories,

    -- ---- returns and friction ----------------------------------------------
    coalesce(a.return_lines, 0)                 AS return_lines,
    coalesce(a.return_value, 0)                 AS return_value,
    CASE WHEN coalesce(a.revenue, 0) > 0
         THEN a.return_value / a.revenue END    AS return_rate,
    coalesce(a.postage, 0)                      AS postage,
    coalesce(a.discounts, 0)                    AS discounts,
    CASE WHEN coalesce(a.revenue, 0) > 0
         THEN a.discounts / a.revenue END       AS discount_rate,

    -- ---- category mix, as shares -------------------------------------------
    CASE WHEN coalesce(m.cat_total,0) > 0 THEN m.cat_christmas  / m.cat_total END AS share_christmas,
    CASE WHEN coalesce(m.cat_total,0) > 0 THEN m.cat_home       / m.cat_total END AS share_home,
    CASE WHEN coalesce(m.cat_total,0) > 0 THEN m.cat_kitchen    / m.cat_total END AS share_kitchen,
    CASE WHEN coalesce(m.cat_total,0) > 0 THEN m.cat_bags       / m.cat_total END AS share_bags,
    CASE WHEN coalesce(m.cat_total,0) > 0 THEN m.cat_lighting   / m.cat_total END AS share_lighting,
    CASE WHEN coalesce(m.cat_total,0) > 0 THEN m.cat_party      / m.cat_total END AS share_party,
    CASE WHEN coalesce(m.cat_total,0) > 0 THEN m.cat_stationery / m.cat_total END AS share_stationery,
    CASE WHEN coalesce(m.cat_total,0) > 0 THEN m.cat_storage    / m.cat_total END AS share_storage,
    CASE WHEN coalesce(m.cat_total,0) > 0 THEN m.cat_garden     / m.cat_total END AS share_garden,
    CASE WHEN coalesce(m.cat_total,0) > 0 THEN m.cat_toys       / m.cat_total END AS share_toys,
    CASE WHEN coalesce(m.cat_total,0) > 0 THEN m.cat_jewellery  / m.cat_total END AS share_jewellery,

    -- ---- ordering rhythm ----------------------------------------------------
    e2.avg_months_between_orders,
    e2.max_months_between_orders,
    e2.gap_variability,
    -- How far past their own usual gap this account has drifted. 1.0 means
    -- exactly on cadence; 3.0 means three times their normal silence.
    CASE WHEN coalesce(e2.avg_months_between_orders, 0) > 0
         THEN ((SELECT month_to FROM w) - a.last_active_month) / e2.avg_months_between_orders
    END                                          AS cadence_overdue,

    -- ---- direction of travel -------------------------------------------------
    CASE WHEN coalesce(e.rev_early, 0) > 0 THEN e.rev_recent / e.rev_early ELSE 1.0 END AS revenue_trend,
    CASE WHEN coalesce(e.inv_early, 0) > 0 THEN e.inv_recent / e.inv_early ELSE 1.0 END AS order_trend,
    coalesce(e.revenue_last_quarter, 0)          AS revenue_last_quarter,

    (SELECT month_from FROM w)                   AS window_from,
    (SELECT month_to FROM w)                     AS window_to
FROM conformed_customers c
LEFT JOIN activity     a  ON c.customer_id = a.customer_id
LEFT JOIN breadth      b  ON c.customer_id = b.customer_id
LEFT JOIN category_mix m  ON c.customer_id = m.customer_id
LEFT JOIN edges        e  ON c.customer_id = e.customer_id
LEFT JOIN cadence      e2 ON c.customer_id = e2.customer_id
-- Only customers that existed by the close of the window.
WHERE c.first_month_index <= (SELECT month_to FROM w);

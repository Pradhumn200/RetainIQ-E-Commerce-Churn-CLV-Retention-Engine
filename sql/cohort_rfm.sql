-- cohort_rfm.sql
-- Phase 1 SQL Foundation for RetainIQ

-- 1. Create Customer-Order Fact Table View
DROP VIEW IF EXISTS customer_order_fact;
CREATE VIEW customer_order_fact AS
WITH order_prices AS (
    SELECT 
        order_id,
        SUM(price) AS order_value
    FROM olist_order_items
    GROUP BY order_id
),
order_categories AS (
    -- Get the category of the first item (lowest order_item_id) in the order
    SELECT 
        oi.order_id,
        COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS category
    FROM olist_order_items oi
    JOIN olist_products p ON oi.product_id = p.product_id
    LEFT JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
    WHERE oi.order_item_id = 1
)
SELECT 
    o.order_id,
    c.customer_unique_id,
    o.order_purchase_timestamp,
    date(o.order_purchase_timestamp) AS order_date,
    COALESCE(op.order_value, 0) AS order_value,
    COALESCE(oc.category, 'unknown') AS category
FROM olist_orders o
JOIN olist_customers c ON o.customer_id = c.customer_id
LEFT JOIN order_prices op ON o.order_id = op.order_id
LEFT JOIN order_categories oc ON o.order_id = oc.order_id
WHERE o.order_status = 'delivered'; -- Focus on completed transactions


-- 2. Cohort Retention Query with Explicit CASTs
DROP TABLE IF EXISTS cohort_retention;
CREATE TABLE cohort_retention AS
WITH customer_first_order AS (
    SELECT 
        customer_unique_id,
        MIN(order_date) AS first_order_date,
        strftime('%Y-%m', MIN(order_date)) || '-01' AS cohort_month
    FROM customer_order_fact
    GROUP BY customer_unique_id
),
customer_activity AS (
    SELECT DISTINCT
        f.customer_unique_id,
        f.cohort_month,
        strftime('%Y-%m', o.order_date) || '-01' AS activity_month
    FROM customer_order_fact o
    JOIN customer_first_order f ON o.customer_unique_id = f.customer_unique_id
),
month_diffs AS (
    SELECT 
        customer_unique_id,
        cohort_month,
        (CAST(strftime('%Y', activity_month) AS INTEGER) - CAST(strftime('%Y', cohort_month) AS INTEGER)) * 12 +
        (CAST(strftime('%m', activity_month) AS INTEGER) - CAST(strftime('%m', cohort_month) AS INTEGER)) AS period_diff
    FROM customer_activity
),
cohort_sizes AS (
    SELECT 
        cohort_month,
        COUNT(DISTINCT customer_unique_id) AS cohort_size
    FROM customer_first_order
    GROUP BY cohort_month
),
retention_counts AS (
    SELECT 
        cohort_month,
        period_diff,
        COUNT(DISTINCT customer_unique_id) AS active_customers
    FROM month_diffs
    GROUP BY cohort_month, period_diff
)
SELECT 
    r.cohort_month,
    s.cohort_size,
    r.period_diff,
    r.active_customers,
    ROUND(CAST(r.active_customers AS REAL) / s.cohort_size, 4) AS retention_rate
FROM retention_counts r
JOIN cohort_sizes s ON r.cohort_month = s.cohort_month
ORDER BY r.cohort_month, r.period_diff;


-- 3. RFM Query
DROP TABLE IF EXISTS rfm_table;
CREATE TABLE rfm_table AS
WITH snapshot AS (
    SELECT date(MAX(order_date), '+1 day') AS snapshot_date FROM customer_order_fact
),
customer_metrics AS (
    SELECT 
        customer_unique_id,
        MAX(order_date) AS last_order_date,
        COUNT(order_id) AS frequency,
        SUM(order_value) AS monetary
    FROM customer_order_fact
    GROUP BY customer_unique_id
)
SELECT 
    m.customer_unique_id,
    CAST(julianday((SELECT snapshot_date FROM snapshot)) - julianday(m.last_order_date) AS INTEGER) AS recency,
    m.frequency,
    m.monetary,
    m.last_order_date
FROM customer_metrics m;


-- 4. Dynamic Purchase Gaps using LAG() Window Function
DROP TABLE IF EXISTS customer_purchase_gaps;
CREATE TABLE customer_purchase_gaps AS
WITH ordered_purchases AS (
    SELECT 
        customer_unique_id,
        order_date,
        LAG(order_date, 1) OVER (PARTITION BY customer_unique_id ORDER BY order_date) AS prev_order_date
    FROM customer_order_fact
)
SELECT 
    customer_unique_id,
    order_date,
    prev_order_date,
    CAST(julianday(order_date) - julianday(prev_order_date) AS INTEGER) AS gap
FROM ordered_purchases
WHERE prev_order_date IS NOT NULL;

-- Product lever: what the customer holds. Churn rate by NumOfProducts, credit card,
-- zero vs non-zero balance, and balance band.
SELECT 'Product' AS lever, 'Number of products' AS dimension, CAST(NumOfProducts AS VARCHAR) AS segment,
       NumOfProducts AS sort_key, COUNT(*) AS customers, SUM(Exited) AS churners,
       ROUND(100.0 * SUM(Exited) / COUNT(*), 1) AS churn_rate_pct
FROM customers GROUP BY NumOfProducts
UNION ALL
SELECT 'Product', 'Credit card', card_label,
       -COUNT(*), COUNT(*), SUM(Exited), ROUND(100.0 * SUM(Exited) / COUNT(*), 1)
FROM customers GROUP BY card_label
UNION ALL
SELECT 'Product', 'Zero vs non-zero balance', balance_flag,
       -COUNT(*), COUNT(*), SUM(Exited), ROUND(100.0 * SUM(Exited) / COUNT(*), 1)
FROM customers GROUP BY balance_flag
UNION ALL
SELECT 'Product', 'Balance band', balance_band,
       MIN(Balance), COUNT(*), SUM(Exited), ROUND(100.0 * SUM(Exited) / COUNT(*), 1)
FROM customers GROUP BY balance_band
ORDER BY dimension, sort_key;

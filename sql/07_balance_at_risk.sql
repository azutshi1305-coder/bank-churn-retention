-- Balance at risk: money held by customers who churned (Exited = 1), by segment.
--   pct_of_segment_balance  = churned balance / that segment's total balance
--   share_of_total_balance  = churned balance / the bank's total balance (all customers)
-- Segments of one dimension add up to the same total, so shares are comparable across dimensions.
WITH totals AS (SELECT SUM(Balance) AS total_balance FROM customers)

SELECT 'Overall' AS dimension, 'All customers' AS segment, 0 AS sort_key,
       COUNT(*) AS customers, SUM(Exited) AS churners,
       ROUND(SUM(Balance), 0) AS segment_balance,
       ROUND(SUM(Balance) FILTER (WHERE Exited = 1), 0) AS churned_balance,
       ROUND(100.0 * SUM(Balance) FILTER (WHERE Exited = 1) / SUM(Balance), 1) AS pct_of_segment_balance,
       ROUND(100.0 * SUM(Balance) FILTER (WHERE Exited = 1) / ANY_VALUE(total_balance), 1) AS share_of_total_balance
FROM customers, totals
UNION ALL
SELECT 'Geography', Geography, -COUNT(*), COUNT(*), SUM(Exited),
       ROUND(SUM(Balance), 0), ROUND(SUM(Balance) FILTER (WHERE Exited = 1), 0),
       ROUND(100.0 * SUM(Balance) FILTER (WHERE Exited = 1) / SUM(Balance), 1),
       ROUND(100.0 * SUM(Balance) FILTER (WHERE Exited = 1) / ANY_VALUE(total_balance), 1)
FROM customers, totals GROUP BY Geography
UNION ALL
SELECT 'Age band', age_band, MIN(Age), COUNT(*), SUM(Exited),
       ROUND(SUM(Balance), 0), ROUND(SUM(Balance) FILTER (WHERE Exited = 1), 0),
       ROUND(100.0 * SUM(Balance) FILTER (WHERE Exited = 1) / SUM(Balance), 1),
       ROUND(100.0 * SUM(Balance) FILTER (WHERE Exited = 1) / ANY_VALUE(total_balance), 1)
FROM customers, totals GROUP BY age_band
UNION ALL
SELECT 'Number of products', CAST(NumOfProducts AS VARCHAR), NumOfProducts, COUNT(*), SUM(Exited),
       ROUND(SUM(Balance), 0), ROUND(SUM(Balance) FILTER (WHERE Exited = 1), 0),
       ROUND(100.0 * SUM(Balance) FILTER (WHERE Exited = 1) / SUM(Balance), 1),
       ROUND(100.0 * SUM(Balance) FILTER (WHERE Exited = 1) / ANY_VALUE(total_balance), 1)
FROM customers, totals GROUP BY NumOfProducts
UNION ALL
SELECT 'Active member', activity_label, -COUNT(*), COUNT(*), SUM(Exited),
       ROUND(SUM(Balance), 0), ROUND(SUM(Balance) FILTER (WHERE Exited = 1), 0),
       ROUND(100.0 * SUM(Balance) FILTER (WHERE Exited = 1) / SUM(Balance), 1),
       ROUND(100.0 * SUM(Balance) FILTER (WHERE Exited = 1) / ANY_VALUE(total_balance), 1)
FROM customers, totals GROUP BY activity_label
UNION ALL
SELECT 'Balance band', balance_band, MIN(Balance), COUNT(*), SUM(Exited),
       ROUND(SUM(Balance), 0), ROUND(SUM(Balance) FILTER (WHERE Exited = 1), 0),
       -- the Zero band holds no money, so its percentages are undefined
       ROUND(100.0 * SUM(Balance) FILTER (WHERE Exited = 1) / NULLIF(SUM(Balance), 0), 1),
       ROUND(100.0 * SUM(Balance) FILTER (WHERE Exited = 1) / ANY_VALUE(total_balance), 1)
FROM customers, totals GROUP BY balance_band
ORDER BY (dimension <> 'Overall'), dimension, sort_key;

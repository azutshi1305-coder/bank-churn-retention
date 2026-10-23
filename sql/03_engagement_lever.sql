-- Engagement lever: how the customer interacts. Churn rate by IsActiveMember and tenure band.
SELECT 'Engagement' AS lever, 'Active member' AS dimension, activity_label AS segment,
       -COUNT(*) AS sort_key, COUNT(*) AS customers, SUM(Exited) AS churners,
       ROUND(100.0 * SUM(Exited) / COUNT(*), 1) AS churn_rate_pct
FROM customers GROUP BY activity_label
UNION ALL
SELECT 'Engagement', 'Tenure band', tenure_band,
       MIN(Tenure), COUNT(*), SUM(Exited), ROUND(100.0 * SUM(Exited) / COUNT(*), 1)
FROM customers GROUP BY tenure_band
ORDER BY dimension, sort_key;

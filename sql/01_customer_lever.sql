-- Customer lever: who the customer is. Churn rate by Geography, Gender, age band, credit score band.
-- sort_key orders bands numerically (MIN of the underlying value) and categories by size.
SELECT 'Customer' AS lever, 'Geography' AS dimension, Geography AS segment,
       -COUNT(*) AS sort_key, COUNT(*) AS customers, SUM(Exited) AS churners,
       ROUND(100.0 * SUM(Exited) / COUNT(*), 1) AS churn_rate_pct
FROM customers GROUP BY Geography
UNION ALL
SELECT 'Customer', 'Gender', Gender,
       -COUNT(*), COUNT(*), SUM(Exited), ROUND(100.0 * SUM(Exited) / COUNT(*), 1)
FROM customers GROUP BY Gender
UNION ALL
SELECT 'Customer', 'Age band', age_band,
       MIN(Age), COUNT(*), SUM(Exited), ROUND(100.0 * SUM(Exited) / COUNT(*), 1)
FROM customers GROUP BY age_band
UNION ALL
SELECT 'Customer', 'Credit score band', credit_band,
       MIN(CreditScore), COUNT(*), SUM(Exited), ROUND(100.0 * SUM(Exited) / COUNT(*), 1)
FROM customers GROUP BY credit_band
ORDER BY dimension, sort_key;

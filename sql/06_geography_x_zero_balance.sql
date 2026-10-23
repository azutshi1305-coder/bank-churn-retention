-- Two-way cut: Geography x zero vs non-zero balance
SELECT Geography, balance_flag,
       COUNT(*) AS customers, SUM(Exited) AS churners,
       ROUND(100.0 * SUM(Exited) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY Geography, balance_flag
ORDER BY Geography, balance_flag;

-- Two-way cut: Geography x IsActiveMember
SELECT Geography, activity_label,
       COUNT(*) AS customers, SUM(Exited) AS churners,
       ROUND(100.0 * SUM(Exited) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY Geography, activity_label
ORDER BY Geography, activity_label;

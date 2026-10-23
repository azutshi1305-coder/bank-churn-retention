-- Two-way cut: age band x NumOfProducts.
-- 3 and 4 products are pooled into '3-4' here only: 4 products has just 60 customers,
-- so splitting it across five age bands would leave cells of a handful of people.
SELECT age_band,
       CASE WHEN NumOfProducts >= 3 THEN '3-4' ELSE CAST(NumOfProducts AS VARCHAR) END AS products,
       COUNT(*) AS customers, SUM(Exited) AS churners,
       ROUND(100.0 * SUM(Exited) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY age_band, products
ORDER BY age_band, products;

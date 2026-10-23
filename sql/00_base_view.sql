-- Base view: reads the raw CSV, drops identifiers (RowNumber, CustomerId, Surname),
-- and defines every band once so all other queries share the same definitions.
-- Run from the repo root (path below is relative to it).
CREATE OR REPLACE VIEW customers AS
SELECT
    CreditScore,
    Geography,
    Gender,
    Age,
    Tenure,
    Balance,
    NumOfProducts,
    HasCrCard,
    IsActiveMember,
    EstimatedSalary,
    Exited,

    CASE WHEN Age < 30 THEN '18-29'
         WHEN Age < 40 THEN '30-39'
         WHEN Age < 50 THEN '40-49'
         WHEN Age < 60 THEN '50-59'
         ELSE '60+' END AS age_band,

    -- FICO-style cut points; the lowest range is split so it is not one huge bucket
    CASE WHEN CreditScore < 500 THEN '<500'
         WHEN CreditScore < 580 THEN '500-579'
         WHEN CreditScore < 670 THEN '580-669'
         WHEN CreditScore < 740 THEN '670-739'
         WHEN CreditScore < 800 THEN '740-799'
         ELSE '800+' END AS credit_band,

    CASE WHEN HasCrCard = 1 THEN 'Has credit card' ELSE 'No credit card' END AS card_label,
    CASE WHEN IsActiveMember = 1 THEN 'Active' ELSE 'Inactive' END AS activity_label,
    CASE WHEN Balance = 0 THEN 'Zero balance' ELSE 'Non-zero balance' END AS balance_flag,

    CASE WHEN Balance = 0 THEN 'Zero'
         WHEN Balance < 100000 THEN '1-100k'
         WHEN Balance < 125000 THEN '100-125k'
         WHEN Balance < 150000 THEN '125-150k'
         ELSE '150k+' END AS balance_band,

    CASE WHEN Tenure <= 2 THEN '0-2 yrs'
         WHEN Tenure <= 5 THEN '3-5 yrs'
         WHEN Tenure <= 8 THEN '6-8 yrs'
         ELSE '9-10 yrs' END AS tenure_band
FROM read_csv_auto('data/raw/Churn_Modelling.csv');

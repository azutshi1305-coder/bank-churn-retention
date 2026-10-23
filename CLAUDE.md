# Bank Churn Prediction & Retention — project context

Author: Annapurna Zutshi. Portfolio project for data science and analytics roles.

## Goal
Predict churn for bank customers, explain drivers (customer, product, engagement levers),
size retention ROI under best/base/worst scenarios, and deploy the model as an API.

## Stack
Python, DuckDB (SQL EDA), PySpark (feature pipeline), scikit-learn + XGBoost,
MLflow (tracking + model registry), SHAP, FastAPI + Pydantic, Docker, AWS (ECR + EC2/App Runner),
Tableau Public (dashboard).

## Data
Kaggle "Bank Customer Churn Prediction" (10k rows), stored at data/raw/. Target: Exited.
Drop identifiers (RowNumber, CustomerId, Surname) before modelling.
Dataset has no time column. Drift monitoring is SIMULATED and must be labelled as such.

## Working rules
- Work one stage at a time; don't jump ahead or build later stages unprompted.
- Explain key decisions briefly so I can defend them in interviews.
- Keep code modular in src/; notebooks are for exploration only.
- Never commit data/, mlruns/, .venv/, or secrets.
- Do not add Claude/AI co-author lines or attribution to commits, code, or docs.

## Stages
1. Setup & data  2. SQL EDA  3. PySpark features  4. Models + MLflow
5. SHAP + threshold + ROI  6. FastAPI  7. Docker + AWS  8. Drift (PSI), Tableau, README
# Tableau data

Five CSVs for the dashboard. Rebuild them from the repo root with `python -m src.reporting.tableau`.
None contains `CustomerId`, `Surname` or `RowNumber` (the writer refuses to save a file that does). `test_index` is an anonymous row number within the test set.

**Read before building charts**
- **Money is in "units" and rests on assumptions** (`src/config/retention_assumptions.yaml`): margin, fee per product, offer cost and the
  retention success rate are illustrative placeholders, not bank figures. The dataset has no treatment data, so "saved" is a modelled expectation, not an observed result.
- **Drift rows marked `simulated = 1` are SIMULATED.** The dataset has no time column; the only real drift comparison is a random train/test split.
- Test-set files (`scored_customers`, `scenarios`, `drivers`) cover the 2,000 held-out customers. `segment_summary` covers all 10,000.

## segment_summary.csv: churn and balance at risk by segment (14 rows, all 10,000 customers)
One row per segment of one dimension. Each dimension's rows add up to all customers, so filter to **one `dimension`** at a time.

| Column | Meaning |
|---|---|
| `dimension` | `Geography`, `Age band`, `Number of products`, `Active member` |
| `segment` | Value within the dimension (e.g. Germany, 40-49, 3, Inactive) |
| `customers`, `churners` | Customers in the segment and how many churned (`Exited = 1`) |
| `churn_rate_pct` | `100 * churners / customers` |
| `segment_balance` | Sum of `Balance` for all customers in the segment |
| `balance_at_risk` | Sum of `Balance` for the segment's churned customers (same definition as `sql/07_balance_at_risk.sql`) |
| `pct_of_segment_balance_at_risk` | `balance_at_risk / segment_balance`, in percent (empty if the segment holds no balance) |
| `share_of_total_balance_at_risk` | `balance_at_risk` / the bank's total balance, in percent; comparable across dimensions |

## drivers.csv: what drives the model (16 rows)
Mean absolute SHAP value per model input on the test set, from the champion's trees (log-odds scale).

| Column | Meaning |
|---|---|
| `rank` | 1 = largest mean &#124;SHAP&#124; |
| `feature` | Model input |
| `lever` | `Customer`, `Product` or `Engagement` (from `src/models/drivers.py`) |
| `mean_abs_shap` | Average size of the feature's effect on a customer's score |
| `share_pct` | Share of the total mean &#124;SHAP&#124;. Correlated features split credit between them, so sum by lever only as a rough guide |

## scored_customers.csv: test customers scored and targeted (6,000 rows = 2,000 customers x 3 scenarios)
Long format: each customer appears once per cost scenario. **Filter to one `scenario`** (or use it as a parameter) or customer counts triple.

| Column | Meaning |
|---|---|
| `test_index` | Anonymous row number in the test set (0-1999) |
| `risk` | Calibrated churn probability from the champion model |
| `risk_band` | `low` (< 20%), `medium` (20% to < 50%), `high` (>= 50%): the API's bands |
| `churned_actual` | Observed `Exited` (1 = churned) |
| `Geography`, `Gender`, `age_band`, `Age`, `NumOfProducts`, `IsActiveMember`, `HasCrCard`, `zero_balance`, `Balance`, `Tenure`, `CreditScore` | Customer attributes. `Gender` is **not** a model input; it is here only to check the campaign is not skewed by gender |
| `scenario` | `best`, `base` or `worst` cost scenario |
| `customer_value` | One year of value: `Balance x margin + NumOfProducts x fee` (scenario assumptions) |
| `expected_net_benefit` | `risk x retention_success_rate x customer_value - offer_cost` |
| `target_unconstrained` | 1 if net benefit > 0 (contact everyone worth contacting) |
| `target_capacity_10pct`, `target_capacity_15pct` | 1 if selected when contact capacity is capped at 10% / 15% of customers, highest net benefit first, never a customer with benefit <= 0 |

## scenarios.csv: campaign results (18 rows = 3 scenarios x 3 capacities x 2 bases)
Filter `basis` to **one value**: `test_2000` is the actual 2,000-customer test set, `per_1000` is the same results scaled to 1,000 customers.

| Column | Meaning |
|---|---|
| `cost_scenario` | `best`, `base`, `worst` (the assumption set) |
| `capacity`, `capacity_rule` | `10%`, `15%` or `Unconstrained` (rule names `capacity_10pct`, `capacity_15pct`, `cost_based`) |
| `basis` | `test_2000` or `per_1000` |
| `customers_contacted`, `pct_targeted` | Customers contacted, and as a percent of customers |
| `expected_churners_reached` | Sum of churn probabilities of those contacted |
| `actual_churners_reached` | How many of them really churned in the test data |
| `expected_customers_saved` | `expected_churners_reached x retention_success_rate` (modelled) |
| `value_retained` | Expected one-year value of the customers saved |
| `campaign_cost` | `customers_contacted x offer_cost` |
| `net_value` | `value_retained - campaign_cost` |
| `roi_pct` | `100 * net_value / campaign_cost` |

## drift.csv: PSI per input and for the score (68 rows = 4 scenarios x 17)
PSI compares the current population with the train set (reference): below 0.10 stable, 0.10 to 0.25 moderate, above 0.25 significant. The **score** is the model's predicted probability.

| Column | Meaning |
|---|---|
| `scenario` | `actual_train_vs_test`, `sim_older_customers`, `sim_fewer_active_members`, `sim_germany_share_doubles` |
| `description` | Plain-language label; simulated ones start with `SIMULATED:` |
| `simulated` | **1 = SIMULATED** (edited test customers), 0 = actual comparison (a random split, so ~0 expected) |
| `kind` | `feature` or `score` |
| `name` | The model input, or `predicted_score` |
| `psi`, `band` | The PSI value and its band (`stable`, `moderate`, `significant`) |

The simulated scenarios are: every customer 5 years older; 30% of active members lapse to inactive; Germany's share of customers doubles (about 25% to 50%, by resampling).

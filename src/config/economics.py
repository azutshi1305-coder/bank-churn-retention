"""Customer value and expected net benefit, plus the loader for retention_assumptions.yaml.

Lives next to the YAML so the API image (which ships only src/features, src/api and src/config) can use it
without importing the training/tracking code in src/models. All amounts are in "units"; the inputs in the
YAML are illustrative assumptions.
"""
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "retention_assumptions.yaml"


def load_assumptions(path=CONFIG_PATH):
    return yaml.safe_load(Path(path).read_text())["scenarios"]


def customer_value(balance, n_products, net_interest_margin, annual_fee_per_product):
    """One year of value: Balance x margin + NumOfProducts x fee."""
    return balance * net_interest_margin + n_products * annual_fee_per_product


def expected_net_benefit(p_churn, value, retention_success_rate, offer_cost):
    return p_churn * retention_success_rate * value - offer_cost

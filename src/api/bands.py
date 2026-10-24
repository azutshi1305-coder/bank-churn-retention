"""Serving policy applied on top of the registered model: probability clipping and risk bands."""
import numpy as np

# Served probabilities are clipped to this range in the service layer (the registered model is unchanged).
# Isotonic calibration can return exactly 0 or 1, which overstates certainty about an individual customer.
PROB_CLIP = (0.005, 0.995)

LOW_BELOW = 0.20     # risk bands are relative to the ~20% base churn rate: below average ...
HIGH_FROM = 0.50     # ... and at least a coin-flip chance of churning

BAND_DESCRIPTIONS = {
    "low": f"churn probability below {LOW_BELOW:.0%} (below the ~20% average churn rate)",
    "medium": f"churn probability from {LOW_BELOW:.0%} up to {HIGH_FROM:.0%}",
    "high": f"churn probability of {HIGH_FROM:.0%} or more",
}


def clip_probabilities(p):
    return np.clip(p, *PROB_CLIP)


def risk_band(probability: float) -> str:
    if probability < LOW_BELOW:
        return "low"
    return "medium" if probability < HIGH_FROM else "high"

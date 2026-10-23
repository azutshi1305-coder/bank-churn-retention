import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from src.features.transform import FEATURE_COLUMNS
from src.models.drivers import LEVER_OF, LEVERS, champion_shap


def test_every_feature_has_exactly_one_lever():
    tagged = [f for features in LEVERS.values() for f in features]
    assert sorted(tagged) == sorted(FEATURE_COLUMNS)
    assert len(tagged) == len(set(tagged))
    assert set(LEVER_OF.values()) == {"Customer", "Product", "Engagement"}


def test_champion_shap_is_additive():
    """SHAP values plus the base value must reproduce the model's log-odds (regression test: the base
    value reads as 0 from a TreeExplainer that has not yet run shap_values)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
    y = (X["a"] + 0.5 * X["b"] + rng.normal(scale=0.5, size=400) > 0.8).astype(int)
    champion = CalibratedClassifierCV(XGBClassifier(n_estimators=20, max_depth=3), method="isotonic", cv=3).fit(X, y)
    values, base, margin = champion_shap(champion, X)
    assert base != 0
    assert np.abs(values.sum(axis=1) + base - margin).max() < 1e-4

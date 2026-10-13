import warnings

import numpy as np
import pandas as pd
import pytest

from neer_match_utilities.baseline_models import GradientBoostingModel


@pytest.fixture
def fitted():
    rng = np.random.default_rng(0)
    match = np.array([0, 1] * 30)
    df = pd.DataFrame({"col_a": match + rng.normal(0, 0.5, 60), "match": match})
    model = GradientBoostingModel()
    model.model.set_params(n_estimators=10)
    return model.fit(df), df


def test_best_threshold_stores_by_default(fitted):
    model, df = fitted

    best_t, _ = model.best_threshold(df)

    assert model.best_threshold_ == best_t


def test_store_threshold_false_leaves_the_model_alone(fitted):
    model, df = fitted

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the correct spelling must not warn
        model.best_threshold(df, store_threshold=False)

    assert model.best_threshold_ is None


def test_misspelled_store_treshold_still_works_with_a_warning(fitted):
    model, df = fitted

    with pytest.warns(FutureWarning, match="store_threshold"):
        model.best_threshold(df, store_treshold=False)

    assert model.best_threshold_ is None


def test_store_threshold_keeps_its_position(fitted):
    model, df = fitted

    # df_val, match_col, feature_prefix, metric, thresholds, store_threshold
    model.best_threshold(df, "match", "col_", "mcc", None, False)

    assert model.best_threshold_ is None

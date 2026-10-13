import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import neer_match_utilities  # noqa: F401  (whatever importing the package installs)
from neer_match_utilities.baseline_models import ProbitMatchingModel
from neer_match_utilities.feature_selection import FeatureSelector


def test_import_leaves_callers_warnings_alone():
    with warnings.catch_warnings(record=True) as caught:
        warnings.warn("raised by the caller's own code", RuntimeWarning)
        np.log(np.zeros(1))  # divide by zero

    messages = [str(w.message) for w in caught]
    assert "raised by the caller's own code" in messages
    assert any("divide by zero" in m for m in messages)


def test_statsmodels_fit_warnings_stay_silenced(monkeypatch):
    original = sm.Probit.fit_regularized

    def noisy_fit(self, *args, **kwargs):
        warnings.warn("did not converge", ConvergenceWarning)
        np.exp(np.array([1000.0]))  # overflow
        return original(self, *args, **kwargs)
    monkeypatch.setattr(sm.Probit, "fit_regularized", noisy_fit)

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"col_a": rng.random(60), "match": [0, 1] * 30})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ProbitMatchingModel().fit(df)
        np.exp(np.array([1000.0]))  # the caller's own overflow, after the fit

    categories = [w.category for w in caught]
    assert ConvergenceWarning not in categories
    assert categories.count(RuntimeWarning) == 1  # only the caller's


def test_feature_selection_fit_warnings_stay_silenced():
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    left = pd.DataFrame({"id": range(6), "name": names})
    right = pd.DataFrame({"id": range(6), "name": [n[::-1] if i % 2 else n for i, n in enumerate(names)]})
    matches = pd.DataFrame({"left": range(6), "right": range(6)})

    selector = FeatureSelector(
        {"name": ["levenshtein", "jaro_winkler"]},
        (left, right, matches),
        cv=2, Cs=2, l1_ratios=(0.9,), n_jobs=1,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selector.execute()

    # scikit-learn 1.8 deprecates `penalty`, so the fits emit at least that
    leaked = [
        w for w in caught
        if issubclass(w.category, (FutureWarning, SklearnConvergenceWarning, RuntimeWarning))
    ]
    assert leaked == []

import numpy as np
import pandas as pd
import pytest

from neer_match_utilities.baseline_io import ModelBaseline
from neer_match_utilities.baseline_models import (
    LogitMatchingModel,
    ProbitMatchingModel,
    _linearly_independent,
)
from neer_match_utilities.baseline_training import BaselineTrainingPipe

MODELS = [LogitMatchingModel, ProbitMatchingModel]


def dependent_features(n=400, seed=0):
    """Similarity features with the exact dependencies similarity maps produce."""
    rng = np.random.default_rng(seed)
    match = (rng.random(n) < 0.3).astype(int)
    a = np.clip(0.4 * match + rng.normal(0.3, 0.2, n), 0, 1)
    b = np.clip(0.3 * match + rng.normal(0.4, 0.2, n), 0, 1)
    return pd.DataFrame({
        "col_a": a,
        "col_a_again": a,          # e.g. token_sort_ratio and token_set_ratio on one word
        "col_b": b,
        "col_mean": (a + b) / 2,   # a combination of two others
        "col_one_minus_a": 1 - a,  # a combination with the intercept
        "match": match,
    })


def repeated_initials(n=400, copies=8, seed=0):
    """
    A name feature and middle initials compared by eight string similarities.

    On single letters every string similarity only tells whether they are equal,
    so the eight initial features are one feature eight times. Before dependent
    columns were dropped, logit and probit both failed to fit this.
    """
    rng = np.random.default_rng(seed)
    match = (rng.random(n) < 0.3).astype(int)
    same_initial = ((rng.random(n) < 0.15) | (match & (rng.random(n) < 0.7))).astype(float)
    name = np.clip(0.4 * match + rng.normal(0.3, 0.2, n), 0, 1)
    initials = {f"col_initial_{k}": same_initial for k in range(copies)}
    return pd.DataFrame({"col_name": name, **initials, "match": match})


def rank_with_intercept(X):
    return np.linalg.matrix_rank(np.column_stack([np.ones(len(X)), X.to_numpy()]))


def test_dependent_columns_are_dropped_and_the_first_of_them_kept():
    X = dependent_features().drop(columns="match")

    kept = _linearly_independent(X)

    assert kept == ["col_a", "col_b"]
    assert rank_with_intercept(X[kept]) == rank_with_intercept(X) == 3


def test_columns_that_differ_in_a_few_rows_are_kept():
    X = dependent_features()[["col_a", "col_b"]].copy()
    X["col_a_nearly"] = X["col_a"]
    X.loc[:2, "col_a_nearly"] += 0.1  # e.g. three two-word names among one-word ones

    assert _linearly_independent(X) == ["col_a", "col_b", "col_a_nearly"]


def test_independent_columns_are_all_kept():
    X = dependent_features()[["col_a", "col_b"]]

    assert _linearly_independent(X) == ["col_a", "col_b"]


@pytest.mark.parametrize("model_cls", MODELS)
def test_repeated_features_are_fit_once(model_cls, capsys):
    df = repeated_initials()

    model = model_cls().fit(df)

    assert "Dropped 7 feature(s) that are linear combinations" in capsys.readouterr().out
    assert model.feature_cols == ["col_initial_0", "col_name"]


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize("make", [repeated_initials, dependent_features])
def test_predictions_equal_those_without_the_dependent_columns(model_cls, make):
    df = make()

    model = model_cls().fit(df)

    reference = model_cls().fit(df[model.feature_cols + ["match"]])
    np.testing.assert_allclose(model.predict_proba(df), reference.predict_proba(df))


def test_logit_needs_no_regularization_once_they_are_dropped(capsys):
    LogitMatchingModel().fit(repeated_initials())

    assert "regularization" not in capsys.readouterr().out


@pytest.mark.parametrize("model_cls", MODELS)
def test_reloaded_model_uses_the_same_features(model_cls, tmp_path):
    df = repeated_initials()
    model = model_cls().fit(df)
    expected = model.predict_proba(df)

    ModelBaseline.save(model, tmp_path, "m")
    loaded = ModelBaseline.load(tmp_path / "m")

    assert loaded.feature_cols == model.feature_cols
    np.testing.assert_allclose(loaded.predict_proba(df), expected)


@pytest.mark.parametrize("kind", ["logit", "probit"])
def test_pipeline_fits_duplicate_similarities_of_one_word_names_once(kind, tmp_path):
    """On one-word names, token_sort_ratio and token_set_ratio are the same feature."""
    rng = np.random.default_rng(1)
    n = 60
    names = ["".join(rng.choice(list("abcdefghijk"), rng.integers(4, 9))) for _ in range(n)]
    order = rng.permutation(n)
    left = pd.DataFrame({"id": range(n), "name": names})
    right = pd.DataFrame({
        "id": range(n),
        "name": [names[i][:-1] + rng.choice(list("xyz")) if rng.random() < 0.5 else names[i] for i in order],
    })
    matches = pd.DataFrame({"left": order, "right": np.arange(n)})
    data = (left, right, matches)

    pipe = BaselineTrainingPipe(
        model_name="m",
        similarity_map={"name": ["token_sort_ratio", "token_set_ratio", "jaro_winkler"]},
        training_data=data,
        testing_data=data,
        model_kind=kind,
        base_dir=tmp_path,
    )
    model = pipe.execute()

    assert model.feature_cols == ["col_name_name_jaro_winkler", "col_name_name_token_set_ratio"]
    assert pipe.metrics_test_["F1"] > 0

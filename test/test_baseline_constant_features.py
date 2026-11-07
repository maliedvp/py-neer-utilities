import numpy as np
import pandas as pd
import pytest

from neer_match_utilities.baseline_io import ModelBaseline
from neer_match_utilities.baseline_models import LogitMatchingModel, ProbitMatchingModel
from neer_match_utilities.baseline_training import BaselineTrainingPipe
from neer_match_utilities.feature_selection import FeatureSelector

MODELS = [LogitMatchingModel, ProbitMatchingModel]


def constant_feature(n=200, n_matches=10):
    """One feature with the same value for every pair, e.g. notzero on a code every record has."""
    return pd.DataFrame({
        "col_code_code_notzero": np.ones(n),
        "match": (np.arange(n) < n_matches).astype(int),
    })


@pytest.mark.parametrize("model_cls", MODELS)
def test_without_a_varying_feature_the_model_predicts_the_share_of_matches(model_cls, capsys):
    df = constant_feature()

    model = model_cls().fit(df)

    assert model.feature_cols == []
    assert "No feature varies in the training data" in capsys.readouterr().out
    # the probit is fit with a slight penalty (alpha 1e-4), which moves it by a few millionths
    np.testing.assert_allclose(model.predict_proba(df), df["match"].mean(), rtol=1e-4)
    assert model.evaluate(df)["F1"] == 0.0  # no pair reaches 0.5


@pytest.mark.parametrize("model_cls", MODELS)
def test_the_reloaded_intercept_only_model_predicts_the_same(model_cls, tmp_path):
    df = constant_feature()
    model = model_cls().fit(df)

    ModelBaseline.save(model, tmp_path, "m")
    loaded = ModelBaseline.load(tmp_path / "m")

    np.testing.assert_allclose(loaded.predict_proba(df), model.predict_proba(df))

    loaded.result = None  # as when the statsmodels pickle cannot be read: from meta.pkl alone
    np.testing.assert_allclose(loaded.predict_proba(df), model.predict_proba(df))


@pytest.mark.parametrize("model_cls", MODELS)
def test_a_model_loaded_without_feature_cols_still_takes_the_frames_features(model_cls):
    """Archives that did not store feature_cols: an empty list stands for the frame's columns."""
    rng = np.random.default_rng(0)
    match = (rng.random(300) < 0.3).astype(int)
    df = pd.DataFrame({
        "col_a": np.clip(0.4 * match + rng.normal(0.3, 0.2, 300), 0, 1),
        "col_b": np.clip(0.3 * match + rng.normal(0.4, 0.2, 300), 0, 1),
        "match": match,
    })
    model = model_cls().fit(df)
    expected = model.predict_proba(df)

    model.feature_cols = []

    np.testing.assert_allclose(model.predict_proba(df), expected)


def code_every_record_has(n=30):
    """The census runs: notzero alone on a code that every record has, so it is 1 for every pair."""
    left = pd.DataFrame({"id": range(n), "code": np.arange(1, n + 1)})
    right = pd.DataFrame({"id": range(n), "code": np.arange(1, n + 1)})
    matches = pd.DataFrame({"left": range(n), "right": range(n)})
    return left, right, matches


CONSTANT_MAP = {"code": ["notzero"]}


def test_feature_selection_keeps_the_map_when_no_feature_varies(capsys):
    result = FeatureSelector(CONSTANT_MAP, code_every_record_has(), cv=2, Cs=3, n_jobs=1).execute()

    assert result.updated_similarity_map == CONSTANT_MAP
    assert result.meta["did_fallback"] is True
    assert result.selected_feature_columns == []
    assert "No feature varies in the training data" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["logit", "probit"])
@pytest.mark.parametrize("feature_selection", [False, True])
def test_pipeline_with_only_a_constant_feature_scores_f1_zero(kind, feature_selection, tmp_path):
    """As the random experiments run: feature selection first, if at all, then the pipeline."""
    data = code_every_record_has()
    similarity_map = CONSTANT_MAP
    if feature_selection:
        similarity_map = FeatureSelector(similarity_map, data, cv=2, Cs=3, n_jobs=1).execute().updated_similarity_map

    pipe = BaselineTrainingPipe(
        model_name="m",
        similarity_map=similarity_map,
        training_data=data,
        testing_data=data,
        model_kind=kind,
        base_dir=tmp_path,
    )
    model = pipe.execute()

    assert model.feature_cols == []
    assert pipe.metrics_train_["F1"] == pipe.metrics_test_["F1"] == 0.0
    assert pipe.metrics_test_["TP"] == pipe.metrics_test_["FP"] == 0

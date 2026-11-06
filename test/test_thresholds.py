import pickle

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from neer_match_utilities.baseline_models import GradientBoostingModel
from neer_match_utilities.model import Model, ValidationMetrics
from neer_match_utilities.panel import GenerateID
from neer_match_utilities.thresholds import pair_labels, scores_at, tune_threshold
from neer_match_utilities.training import TrainingPipe, focal_loss, soft_f1_loss


def make_data(suffix="x"):
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    left = pd.DataFrame({"id_unique": range(6), "name": names})
    right = pd.DataFrame({"id_unique": range(6), "name": [n + suffix for n in names]})
    matches = pd.DataFrame({"left": range(6), "right": range(6)})
    return left, right, matches


def make_pipe(**kwargs):
    kwargs.setdefault("stage_2", False)
    return TrainingPipe(
        model_name="m",
        training_data=make_data(),
        testing_data=make_data("y"),
        similarity_map={"name": ["levenshtein", "jaro_winkler"]},
        epochs_1=kwargs.pop("epochs_1", 2), mismatch_share_1=1.0,
        no_tm_pbatch=2,
        **kwargs,
    )


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """TrainingPipe writes under the working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path / "m"


def scripted_logs(monkeypatch, **values):
    """Log the given values per epoch instead of scoring the validation pairs."""
    values = {name: iter(v) for name, v in values.items()}

    def on_epoch_end(self, epoch, logs=None):
        for name, v in values.items():
            logs[name] = next(v)
    monkeypatch.setattr(ValidationMetrics, "on_epoch_end", on_epoch_end)


def loaded_paths(monkeypatch):
    """Record which saved models the pipeline loads."""
    loaded, original = [], Model.load

    def load(model_directory):
        loaded.append(model_directory)
        return original(model_directory)
    monkeypatch.setattr(Model, "load", staticmethod(load))
    return loaded


# ---------- Choosing the threshold ----------

@pytest.mark.parametrize("metric", ["mcc", "f1"])
def test_tuning_chooses_as_gradient_boosting_does(metric):
    rng = np.random.default_rng(0)
    labels = (rng.random(3000) < 0.05).astype(int)
    # float32 values, as the network predicts them
    probabilities = np.clip(rng.normal(0.3 + 0.3 * labels, 0.15), 0, 1).astype(np.float32)

    threshold, scores = tune_threshold(labels, probabilities, metric)
    expected_threshold, expected = GradientBoostingModel()._best_threshold_from_proba(
        labels, probabilities.astype(float), metric=metric
    )

    assert threshold == pytest.approx(expected_threshold)
    assert {k: scores[k] for k in expected} == pytest.approx(expected)


def test_undefined_rates_are_zero():
    scores = scores_at(np.array([1, 0, 0]), np.array([0.1, 0.2, 0.3]), 0.5)

    assert (scores["TP"], scores["FP"], scores["TN"], scores["FN"]) == (0, 0, 2, 1)
    assert scores["Precision"] == scores["F1"] == scores["MCC"] == 0.0


def test_pair_labels_follow_predict_order():
    matches = pd.DataFrame({"left": [0, 2], "right": [1, 0]})

    # left 0 with right 0, 1, 2; left 1 with ...; left 2 with ...
    assert pair_labels(matches, 3, 3).tolist() == [0, 1, 0, 0, 0, 0, 1, 0, 0]


# ---------- TrainingPipe ----------

def test_tuned_threshold_is_saved_and_used(run_dir):
    pipe = make_pipe(validation_data=make_data("z"))
    pipe.execute()

    threshold = pipe.model.best_threshold_
    assert 0.0 < threshold < 1.0
    performance = pd.read_csv(run_dir / "performance.csv").set_index("data")
    assert (performance["threshold"] == threshold).all()

    # Reloaded, the model evaluates at it without being told
    loaded = Model.load(run_dir)
    assert loaded.best_threshold_ == threshold
    test = loaded.evaluate(*make_data("y"), batch_size=1024)
    for count in ("TP", "FP", "TN", "FN"):
        assert test[count] == performance.loc["test", count]


def test_without_tuning_the_threshold_is_fixed(run_dir):
    pipe = make_pipe(validation_data=make_data("z"), tune_threshold=False, threshold=0.4)
    pipe.execute()

    performance = pd.read_csv(run_dir / "performance.csv")
    assert (performance["threshold"] == 0.4).all()
    assert pipe.model.best_threshold_ is None
    assert Model.load(run_dir).best_threshold_ is None


def test_without_validation_data_nothing_is_tuned(run_dir):
    make_pipe().execute()

    assert (pd.read_csv(run_dir / "performance.csv")["threshold"] == 0.5).all()


def test_epochs_are_chosen_by_the_tuned_score(run_dir, monkeypatch):
    # val_F1 at the fixed threshold would choose epoch 1
    scripted_logs(monkeypatch, val_F1=[0.9, 0.1, 0.1], val_tuned_MCC=[0.2, 0.9, 0.4])
    loaded = loaded_paths(monkeypatch)

    make_pipe(epochs_1=3, validation_data=make_data("z")).execute()

    assert loaded[0].name == "epoch_02"


def stage_scores(tuned_mcc, threshold=0.3):
    """Scores as _validation_scores returns them."""
    return {"val_F1": 0.5, "val_MCC": 0.5, "val_threshold": threshold, "val_tuned_F1": 0.5,
            "val_tuned_MCC": tuned_mcc}, np.zeros(36)


def test_stage_2_keeps_stage_1_model_when_no_epoch_beats_it(run_dir, monkeypatch):
    scripted_logs(monkeypatch, val_tuned_MCC=[0.5, 0.6, 0.7, 0.8])  # 2 epochs per stage
    calls = iter([stage_scores(0.95), stage_scores(0.95, threshold=0.37)])
    monkeypatch.setattr(TrainingPipe, "_validation_scores", lambda self: next(calls))
    loaded = loaded_paths(monkeypatch)

    make_pipe(
        validation_data=make_data("z"),
        stage_2=True, epochs_2=2, mismatch_share_2=1.0, gamma=1.5, max_alpha=0.9,
    ).execute()

    # stage 1 chose its epoch 2; stage 2 kept it rather than its own epochs
    assert [p.parent.name for p in loaded[:2]] == ["checkpoints", "checkpoints_stage_1"]
    assert loaded[1].name == "epoch_02"
    overview = pd.read_csv(run_dir / "checkpoints_stage_2" / "epoch_overview.csv")
    assert overview["epoch"].tolist() == [0, 1, 2]
    assert (pd.read_csv(run_dir / "performance.csv")["threshold"] == 0.37).all()


def test_stage_2_takes_its_own_epoch_when_it_is_better(run_dir, monkeypatch):
    scripted_logs(monkeypatch, val_tuned_MCC=[0.5, 0.6, 0.7, 0.8])
    calls = iter([stage_scores(0.65), stage_scores(0.8)])
    monkeypatch.setattr(TrainingPipe, "_validation_scores", lambda self: next(calls))
    loaded = loaded_paths(monkeypatch)

    make_pipe(
        validation_data=make_data("z"),
        stage_2=True, epochs_2=2, mismatch_share_2=1.0, gamma=1.5, max_alpha=0.9,
    ).execute()

    assert (loaded[1].parent.name, loaded[1].name) == ("checkpoints", "epoch_02")


def test_a_collapsed_model_is_reported(run_dir, monkeypatch):
    calls = iter([stage_scores(0.0), stage_scores(0.0)])
    monkeypatch.setattr(TrainingPipe, "_validation_scores", lambda self: next(calls))

    with pytest.warns(RuntimeWarning, match="collapsed"):
        make_pipe(validation_data=make_data("z")).execute()


@pytest.mark.parametrize(
    "kwargs, match",
    [({"threshold": 1.0}, "threshold"), ({"tune_metric": "auc"}, "tune_metric")],
)
def test_invalid_threshold_settings_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_pipe(**kwargs)


def test_models_saved_before_thresholds_were_saved_still_load(run_dir):
    make_pipe().execute()
    info_path = run_dir / "model" / "model_info.pkl"
    info = pickle.loads(info_path.read_bytes())
    del info["best_threshold"]
    info_path.write_bytes(pickle.dumps(info))

    assert Model.load(run_dir).best_threshold_ is None


# ---------- Losses ----------

@pytest.mark.parametrize("make_loss", [lambda: focal_loss(alpha=0.8, gamma=1.0), soft_f1_loss])
def test_losses_score_each_pair_once(make_loss):
    loss = make_loss()
    y_true = tf.constant([1.0, 0.0, 0.0, 1.0])
    y_pred = tf.constant([0.9, 0.2, 0.6, 0.4])

    # neer-match's evaluate passes predictions as a column
    as_column = float(loss(y_true, tf.reshape(y_pred, (-1, 1))))
    assert as_column == pytest.approx(float(loss(y_true, y_pred)))


def test_focal_loss_is_the_mean_over_pairs():
    y_true = np.array([1.0, 0.0, 0.0, 1.0])
    y_pred = np.array([0.9, 0.2, 0.6, 0.4])
    alpha, gamma = 0.8, 1.0
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    expected = np.mean(alpha_t * (1 - p_t) ** gamma * -np.log(p_t))

    value = focal_loss(alpha=alpha, gamma=gamma)(tf.constant(y_true), tf.reshape(tf.constant(y_pred), (-1, 1)))
    assert float(value) == pytest.approx(expected, rel=1e-5)


# ---------- Baselines and GenerateID ----------

def scripted_gb(probabilities, best_threshold=None):
    model = GradientBoostingModel()
    model._check_fitted = lambda: None
    model.predict_proba = lambda df, feature_prefix="col_": np.asarray(probabilities)
    model.best_threshold_ = best_threshold
    return model


def test_gradient_boosting_evaluates_at_its_tuned_threshold():
    pairs = pd.DataFrame({"col_x": [0, 0, 0, 0], "match": [1, 1, 0, 0]})
    model = scripted_gb([0.9, 0.35, 0.3, 0.1], best_threshold=0.32)

    assert model.evaluate(pairs)["TP"] == 2  # at 0.32
    assert model.evaluate(pairs, threshold=0.5)["TP"] == 1
    model.best_threshold_ = None
    assert model.evaluate(pairs)["TP"] == 1  # at 0.5


class Suggests:
    def __init__(self, best_threshold=None):
        if best_threshold is not None:
            self.best_threshold_ = best_threshold

    def suggest(self, left, right, count=1, verbose=0):
        return pd.DataFrame({"left": [], "right": [], "prediction": []})


@pytest.mark.parametrize(
    "model, kwargs, expected",
    [
        (Suggests(0.27), {}, 0.27),
        (Suggests(), {}, 0.9),
        (Suggests(0.27), {"prediction_threshold": 0.6}, 0.6),
        (Suggests(), {"prediction_threshold": None}, 0.5),  # a dict map, as before
    ],
)
def test_generate_id_defaults_to_the_tuned_threshold(model, kwargs, expected):
    panel = pd.DataFrame({"year": [1900, 1910], "name": ["A", "A"]})

    generator = GenerateID(
        panel, panel_var="pid", time_var="year", model=model,
        similarity_map={"name": ["levenshtein"]}, **kwargs,
    )

    assert generator.prediction_threshold == expected

import pandas as pd
import pytest

from neer_match_utilities.model import Model, ValidationMetrics
from neer_match_utilities.training import TrainingPipe


def make_data(suffix="x"):
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    left = pd.DataFrame({"id_unique": range(6), "name": names})
    right = pd.DataFrame({"id_unique": range(6), "name": [n + suffix for n in names]})
    matches = pd.DataFrame({"left": range(6), "right": range(6)})
    return left, right, matches


def stage_1_pipe(epochs, **kwargs):
    return TrainingPipe(
        model_name="m",
        training_data=make_data(),
        testing_data=make_data(),
        similarity_map={"name": ["levenshtein"]},
        stage_2=False,
        epochs_1=epochs,
        mismatch_share_1=1.0,
        no_tm_pbatch=2,
        **kwargs,
    )


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """TrainingPipe writes under the working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path / "m"


def scripted_val_f1(monkeypatch, values):
    """Log the given val_F1 per epoch instead of evaluating."""
    values = iter(values)

    def on_epoch_end(self, epoch, logs=None):
        logs["val_F1"] = next(values)
    monkeypatch.setattr(ValidationMetrics, "on_epoch_end", on_epoch_end)


def loaded_epochs(monkeypatch):
    """Record which checkpoints the pipeline loads."""
    loaded, original = [], Model.load

    def load(model_directory):
        loaded.append(model_directory.name)
        return original(model_directory)
    monkeypatch.setattr(Model, "load", staticmethod(load))
    return loaded


def test_validation_metrics_are_logged_each_epoch(run_dir):
    stage_1_pipe(epochs=2, validation_data=make_data("y")).execute()

    overview = pd.read_csv(run_dir / "checkpoints_stage_1" / "epoch_overview.csv")
    assert {"val_F1", "val_MCC", "val_Loss"} <= set(overview.columns)
    assert len(overview) == 2


def test_best_epoch_is_chosen_by_validation_f1(run_dir, monkeypatch):
    scripted_val_f1(monkeypatch, [0.2, 0.9, 0.4])
    loaded = loaded_epochs(monkeypatch)

    stage_1_pipe(epochs=3, validation_data=make_data("y")).execute()

    assert loaded[0] == "epoch_02"


def test_early_stopping_ends_a_stage(run_dir, monkeypatch):
    scripted_val_f1(monkeypatch, [0.9, 0.5, 0.4, 0.3, 0.2])
    loaded = loaded_epochs(monkeypatch)

    stage_1_pipe(epochs=5, validation_data=make_data("y"), early_stopping_patience=1).execute()

    overview = pd.read_csv(run_dir / "checkpoints_stage_1" / "epoch_overview.csv")
    assert len(overview) == 2  # no improvement in epoch 2, so it stops there
    assert loaded[0] == "epoch_01"


def test_without_validation_nothing_changes(run_dir):
    stage_1_pipe(epochs=2).execute()

    overview = pd.read_csv(run_dir / "checkpoints_stage_1" / "epoch_overview.csv")
    assert not [c for c in overview.columns if c.startswith("val_")]


def test_early_stopping_requires_validation_data():
    with pytest.raises(ValueError, match="validation_data"):
        stage_1_pipe(epochs=2, early_stopping_patience=1)

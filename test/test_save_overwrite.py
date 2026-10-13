import builtins

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from neer_match.matching_model import DLMatchingModel
from neer_match.similarity_map import SimilarityMap

from neer_match_utilities.baseline_io import ModelBaseline
from neer_match_utilities.baseline_models import GradientBoostingModel
from neer_match_utilities.baseline_training import BaselineTrainingPipe
from neer_match_utilities.model import Model
from neer_match_utilities.training import Training, TrainingPipe


@pytest.fixture(autouse=True)
def no_prompt(monkeypatch):
    """Saving must never wait on stdin."""
    def _fail(*args, **kwargs):
        raise AssertionError("input() was called")
    monkeypatch.setattr(builtins, "input", _fail)


@pytest.fixture
def dl_model():
    smap = SimilarityMap({"name": ["levenshtein"]})
    model = DLMatchingModel(similarity_map=smap)
    sizes = smap.association_sizes()
    model.build(input_shapes=[tf.TensorShape([None, s]) for s in sizes])
    model([tf.zeros((1, s)) for s in sizes])
    return model


@pytest.fixture
def gb_model():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "col_name_name_levenshtein": rng.random(40),
        "match": [0, 1] * 20,
    })
    model = GradientBoostingModel()
    model.model.set_params(n_estimators=5)
    return model.fit(df)


def test_model_save_refuses_existing_directory(tmp_path, dl_model):
    Model.save(model=dl_model, target_directory=tmp_path, name="m")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        Model.save(model=dl_model, target_directory=tmp_path, name="m")


def test_model_save_overwrite_replaces_directory(tmp_path, dl_model):
    Model.save(model=dl_model, target_directory=tmp_path, name="m")
    stale = tmp_path / "m" / "model" / "stale.txt"
    stale.write_text("from a previous save")

    Model.save(model=dl_model, target_directory=tmp_path, name="m", overwrite=True)

    assert not stale.exists()
    assert isinstance(Model.load(tmp_path / "m"), DLMatchingModel)


def test_baseline_save_refuses_existing_directory(tmp_path, gb_model):
    ModelBaseline.save(model=gb_model, target_directory=tmp_path, name="m")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        ModelBaseline.save(model=gb_model, target_directory=tmp_path, name="m")


def test_baseline_save_overwrite_replaces_directory(tmp_path, gb_model):
    ModelBaseline.save(model=gb_model, target_directory=tmp_path, name="m")
    stale = tmp_path / "m" / "model" / "stale.txt"
    stale.write_text("from a previous save")

    ModelBaseline.save(model=gb_model, target_directory=tmp_path, name="m", overwrite=True)

    assert not stale.exists()
    assert isinstance(ModelBaseline.load(tmp_path / "m"), GradientBoostingModel)


def test_export_refuses_before_writing_metrics(tmp_path, gb_model):
    ModelBaseline.save(model=gb_model, target_directory=tmp_path, name="m")
    performance = tmp_path / "m" / "performance.csv"
    performance.write_text("metrics of the saved model")

    training = Training(similarity_map={"name": ["levenshtein"]})
    with pytest.raises(FileExistsError):
        training.performance_statistics_export(
            model=gb_model,
            model_name="m",
            target_directory=tmp_path,
            evaluation_train={"F1": 0.0},
            evaluation_test={"F1": 0.0},
            export_model=True,
        )

    assert performance.read_text() == "metrics of the saved model"


def test_baseline_pipe_refuses_before_building_features(tmp_path):
    (tmp_path / "m" / "model").mkdir(parents=True)

    pipe = BaselineTrainingPipe(
        model_name="m",
        similarity_map={"name": ["levenshtein"]},
        training_data=None,  # never reached
        testing_data=None,
        base_dir=tmp_path,
    )
    with pytest.raises(FileExistsError, match="overwrite=True"):
        pipe.execute()


def test_training_pipe_refuses_before_training(tmp_path, monkeypatch):
    (tmp_path / "m" / "model").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)  # TrainingPipe saves under the working directory

    frame = pd.DataFrame({"id_unique": [0, 1], "name": ["a", "b"]})
    matches = pd.DataFrame({"left": [0], "right": [0]})
    pipe = TrainingPipe(
        model_name="m",
        training_data=(frame, frame, matches),
        testing_data=(frame, frame, matches),
        similarity_map={"name": ["levenshtein"]},
        stage_2=False,
        epochs_1=1,
        mismatch_share_1=1.0,
        no_tm_pbatch=1,
    )
    with pytest.raises(FileExistsError, match="overwrite=True"):
        pipe.execute()

    # The final save would raise too; no checkpoints shows it failed up front
    assert [p.name for p in (tmp_path / "m").iterdir()] == ["model"]

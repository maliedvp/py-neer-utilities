import numpy as np
import pandas as pd
import pytest
from neer_match.matching_model import DLMatchingModel

from neer_match_utilities.training import TrainingPipe


def make_data(suffix="x"):
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    left = pd.DataFrame({"id_unique": range(6), "name": names})
    right = pd.DataFrame({"id_unique": range(6), "name": [n + suffix for n in names]})
    matches = pd.DataFrame({"left": range(6), "right": range(6)})
    return left, right, matches


def make_pipe(**kwargs):
    return TrainingPipe(
        model_name="m",
        training_data=make_data(),
        testing_data=make_data("y"),
        similarity_map={"name": ["levenshtein", "jaro_winkler"]},
        epochs_1=1, mismatch_share_1=1.0,
        no_tm_pbatch=2,
        **kwargs,
    )


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """TrainingPipe writes under the working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path / "m"


def test_every_evaluation_uses_evaluation_batch_size(run_dir, monkeypatch):
    original = DLMatchingModel.evaluate
    sizes = []

    def spy(self, left=None, *args, **kwargs):
        if isinstance(left, pd.DataFrame):  # the pipeline's calls, not Keras's
            sizes.append(kwargs.get("batch_size"))
        return original(self, left, *args, **kwargs)
    monkeypatch.setattr(DLMatchingModel, "evaluate", spy)

    make_pipe(
        validation_data=make_data("z"),
        epochs_2=1, mismatch_share_2=1.0, gamma=1.5, max_alpha=0.9,
        evaluation_batch_size=5,
    ).execute()

    # validation after stage 1's epoch, the check after stage 1, validation after
    # stage 2's epoch, and the final evaluation of the training and test pairs
    assert sizes == [5] * 5


def test_scores_do_not_depend_on_evaluation_batch_size(run_dir):
    pipe = make_pipe(stage_2=False)
    pipe.execute()
    model = pipe.model

    # At 0.5 this barely trained model predicts no match at all. At its median
    # prediction about half the training pairs count as matches, so that the
    # counts compared below are not all zero.
    probabilities = model.predict(pipe.left_train, pipe.right_train, batch_size=1024).ravel()
    model.compile(loss=model.loss, threshold=float(np.median(probabilities)))

    # 36 pairs per split: one batch at the default, three at neer-match's 16
    for split, (left, right, matches) in (
        ("train", (pipe.left_train, pipe.right_train, pipe.matches_train)),
        ("test", (pipe.left_test, pipe.right_test, pipe.matches_test)),
    ):
        scored = model.evaluate(
            left, right, matches, mismatch_share=1.0, batch_size=pipe.evaluation_batch_size
        )
        expected = model.evaluate(left, right, matches, mismatch_share=1.0, batch_size=16)
        if split == "train":
            assert 0 < scored["TP"] + scored["FP"] < 36
        for count in ("TP", "FP", "TN", "FN"):
            assert scored[count] == expected[count], (split, count)
        for rate in ("Accuracy", "Recall", "Precision", "F1", "MCC"):
            assert scored[rate] == pytest.approx(expected[rate], abs=1e-9, nan_ok=True), (split, rate)
        # Not Loss: neer-match sums the loss over the batches, and soft F1, the
        # loss this model was compiled with, is computed per batch


@pytest.mark.parametrize("value", [0, -3, 2.5, True, "1024"])
def test_evaluation_batch_size_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="evaluation_batch_size"):
        make_pipe(stage_2=False, evaluation_batch_size=value)

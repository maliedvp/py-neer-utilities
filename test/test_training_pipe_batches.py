import pandas as pd
from neer_match.matching_model import DLMatchingModel

from neer_match_utilities.training import TrainingPipe


def make_pipe(**kwargs):
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    left = pd.DataFrame({"id_unique": range(6), "name": names})
    right = pd.DataFrame({"id_unique": range(6), "name": [n + "x" for n in names]})
    matches = pd.DataFrame({"left": range(6), "right": range(6)})
    return TrainingPipe(
        model_name="m",
        training_data=(left, right, matches),
        testing_data=(left, right, matches),
        similarity_map={"name": ["levenshtein"]},
        **kwargs,
    )


def test_stage_1_only_keeps_no_tm_pbatch():
    pipe = make_pipe(stage_2=False, epochs_1=1, mismatch_share_1=1.0, no_tm_pbatch=3)

    assert pipe.no_tm_pbatch == 3


def test_every_pass_targets_no_tm_pbatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # TrainingPipe writes under the working directory

    original = TrainingPipe.required_batch_size
    targets = []

    def spy(*args, desired_pos_per_batch, **kwargs):
        targets.append(desired_pos_per_batch)
        return original(*args, desired_pos_per_batch=desired_pos_per_batch, **kwargs)
    monkeypatch.setattr(TrainingPipe, "required_batch_size", staticmethod(spy))

    make_pipe(
        epochs_1=1, mismatch_share_1=1.0,
        epochs_2=1, mismatch_share_2=1.0, gamma=1.5, max_alpha=0.9,
        no_tm_pbatch=3,
    ).execute()

    # warm-up pass, stage 1, stage 2
    assert targets == [3, 3, 3]


def test_every_pass_is_sized_for_the_share_it_trains_with(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # TrainingPipe writes under the working directory

    original_size, original_fit = TrainingPipe.required_batch_size, DLMatchingModel.fit
    sized_for, trained_with = [], []

    def size_spy(num_matches, num_left, num_right, mismatch_share, **kwargs):
        sized_for.append(mismatch_share)
        return original_size(num_matches, num_left, num_right, mismatch_share, **kwargs)

    def fit_spy(self, *args, **kwargs):
        trained_with.append(kwargs["mismatch_share"])
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(TrainingPipe, "required_batch_size", staticmethod(size_spy))
    monkeypatch.setattr(DLMatchingModel, "fit", fit_spy)

    make_pipe(
        epochs_1=1, mismatch_share_1=1.0,
        epochs_2=1, mismatch_share_2=0.5, gamma=1.5, max_alpha=0.9,
        no_tm_pbatch=3,
    ).execute()

    # warm-up pass, stage 1, stage 2
    assert sized_for == trained_with == [TrainingPipe.WARMUP_MISMATCH_SHARE, 1.0, 0.5]

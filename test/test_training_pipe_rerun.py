import pandas as pd
import pytest

from neer_match_utilities.training import TrainingPipe


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """TrainingPipe writes under the working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path / "m"


def stage_1_pipe(**kwargs):
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    left = pd.DataFrame({"id_unique": range(6), "name": names})
    right = pd.DataFrame({"id_unique": range(6), "name": [n + "x" for n in names]})
    matches = pd.DataFrame({"left": range(6), "right": range(6)})
    return TrainingPipe(
        model_name="m",
        training_data=(left, right, matches),
        testing_data=(left, right, matches),
        similarity_map={"name": ["levenshtein"]},
        stage_2=False,
        epochs_1=1,
        mismatch_share_1=1.0,
        no_tm_pbatch=2,
        **kwargs,
    )


def leave_stale_checkpoint(run_dir, stage_dir):
    stale = run_dir / stage_dir / "epoch_09"
    stale.mkdir(parents=True)
    return stale


def test_rerun_replaces_stage_1_checkpoints(run_dir):
    # As left by an earlier run that got past stage 1 but never saved a model
    stale = leave_stale_checkpoint(run_dir, "checkpoints_stage_1")

    stage_1_pipe().execute()

    assert (run_dir / "checkpoints_stage_1" / "epoch_01").exists()
    assert not stale.exists()


def test_final_save_removes_checkpoints_of_skipped_stage(run_dir):
    leave_stale_checkpoint(run_dir, "checkpoints_stage_2")

    stage_1_pipe().execute()

    assert (run_dir / "model").exists()
    assert not (run_dir / "checkpoints_stage_2").exists()


def test_failed_run_leaves_previous_files(run_dir, monkeypatch):
    (run_dir / "model").mkdir(parents=True)
    stale = leave_stale_checkpoint(run_dir, "checkpoints_stage_2")

    def fail(self, P):
        raise RuntimeError("training failed")
    monkeypatch.setattr(TrainingPipe, "_round1", fail)

    with pytest.raises(RuntimeError, match="training failed"):
        stage_1_pipe(overwrite=True).execute()

    assert (run_dir / "model").exists()
    assert stale.exists()

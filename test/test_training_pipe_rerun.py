import pandas as pd
import pytest

from neer_match_utilities.training import TrainingPipe


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """TrainingPipe writes under the working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path / "m"


def single_stage_pipe(stage=1, **kwargs):
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    left = pd.DataFrame({"id_unique": range(6), "name": names})
    right = pd.DataFrame({"id_unique": range(6), "name": [n + "x" for n in names]})
    matches = pd.DataFrame({"left": range(6), "right": range(6)})
    if stage == 1:
        stage_args = dict(stage_2=False, epochs_1=1, mismatch_share_1=1.0)
    else:
        stage_args = dict(
            stage_1=False, epochs_2=1, mismatch_share_2=1.0, gamma=1.5, max_alpha=0.9
        )
    return TrainingPipe(
        model_name="m",
        training_data=(left, right, matches),
        testing_data=(left, right, matches),
        similarity_map={"name": ["levenshtein"]},
        no_tm_pbatch=2,
        **stage_args,
        **kwargs,
    )


def leave_stale_checkpoint(run_dir, stage_dir):
    stale = run_dir / stage_dir / "epoch_09"
    stale.mkdir(parents=True)
    return stale


def test_rerun_replaces_stage_1_checkpoints(run_dir):
    # As left by an earlier run that got past stage 1 but never saved a model
    stale = leave_stale_checkpoint(run_dir, "checkpoints_stage_1")

    single_stage_pipe().execute()

    assert (run_dir / "checkpoints_stage_1" / "epoch_01").exists()
    assert not stale.exists()


@pytest.mark.parametrize("stage", [1, 2])
def test_stage_starts_from_empty_checkpoints(run_dir, stage):
    # As left by a run interrupted mid-stage, before the rename
    stale = leave_stale_checkpoint(run_dir, "checkpoints")

    single_stage_pipe(stage).execute()

    stage_dir = run_dir / f"checkpoints_stage_{stage}"
    assert (stage_dir / "epoch_01").exists()
    assert not (stage_dir / stale.name).exists()


def test_final_save_removes_checkpoints_of_skipped_stage(run_dir):
    leave_stale_checkpoint(run_dir, "checkpoints_stage_2")

    single_stage_pipe().execute()

    assert (run_dir / "model").exists()
    assert not (run_dir / "checkpoints_stage_2").exists()


def test_failed_run_leaves_previous_files(run_dir, monkeypatch):
    (run_dir / "model").mkdir(parents=True)
    stale = leave_stale_checkpoint(run_dir, "checkpoints_stage_2")

    def fail(self, P):
        raise RuntimeError("training failed")
    monkeypatch.setattr(TrainingPipe, "_round1", fail)

    with pytest.raises(RuntimeError, match="training failed"):
        single_stage_pipe(overwrite=True).execute()

    assert (run_dir / "model").exists()
    assert stale.exists()

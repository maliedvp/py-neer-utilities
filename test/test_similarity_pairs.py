import numpy as np
import pandas as pd
import pytest
from neer_match.similarity_map import SimilarityMap

from neer_match_utilities.baseline_models import GradientBoostingModel
from neer_match_utilities.baseline_training import BaselineTrainingPipe
from neer_match_utilities.feature_selection import FeatureSelector
from neer_match_utilities.similarity_features import (
    SimilarityFeatures,
    subsample_non_match_positions,
    subsample_non_matches,
)

SMAP = {"name": ["levenshtein", "jaro_winkler"], "city": ["levenshtein", "notmissing"]}


def make_data(n=12, seed=0):
    """Right row j is a slightly edited copy of left row order[j]."""
    rng = np.random.default_rng(seed)
    names = ["".join(rng.choice(list("abcdefgh"), 6)) for _ in range(n)]
    cities = rng.choice(np.array(["x", "y", "z", None], dtype=object), n)
    order = rng.permutation(n)
    left = pd.DataFrame({"id": [f"L{i}" for i in range(n)], "name": names, "city": cities})
    right = pd.DataFrame({
        "id": [f"R{i}" for i in range(n)],
        "name": [names[i][:-1] + "q" for i in order],
        "city": cities[order],
    })
    matches = pd.DataFrame({"left": order, "right": np.arange(n)})
    return left, right, matches


def make_pairs(matches_are_indices=True):
    left, right, matches = make_data()
    if not matches_are_indices:
        matches = pd.DataFrame({
            "left": left["id"].to_numpy()[matches["left"]],
            "right": right["id"].to_numpy()[matches["right"]],
        })
    return SimilarityFeatures(SimilarityMap(SMAP)).pairs(
        left, right, matches, "id", "id", matches_are_indices=matches_are_indices
    )


@pytest.mark.parametrize("matches_are_indices", [True, False])
@pytest.mark.parametrize("pairs_per_chunk", [1, 7, 12, 13, 10**6])
def test_chunks_concatenate_to_frame(matches_are_indices, pairs_per_chunk):
    pairs = make_pairs(matches_are_indices)

    blocks = pd.concat(pairs.chunks(pairs_per_chunk))

    pd.testing.assert_frame_equal(blocks, pairs.frame())


def test_frame_has_one_matched_row_per_match():
    pairs = make_pairs()
    frame = pairs.frame()

    assert len(frame) == 12 * 12
    assert frame["match"].sum() == 12


def test_rows_are_the_frames_rows_in_the_requested_order():
    pairs = make_pairs()
    positions = np.random.default_rng(1).permutation(len(pairs))[:20]

    pd.testing.assert_frame_equal(pairs.rows(positions), pairs.frame().iloc[positions])


def test_match_flags_are_the_frames_match_column():
    pairs = make_pairs()

    np.testing.assert_array_equal(pairs.match_flags(5), pairs.frame()["match"].to_numpy())


@pytest.mark.parametrize("mismatch_share", [1.0, 0.3])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("random_state", [42, None])
def test_positions_reproduce_subsample_non_matches(mismatch_share, shuffle, random_state):
    frame = make_pairs().frame()
    args = dict(mismatch_share=mismatch_share, random_state=random_state, shuffle=shuffle)

    np.random.seed(7)  # what random_state=None draws from
    expected = subsample_non_matches(frame, **args)
    np.random.seed(7)
    positions = subsample_non_match_positions(frame["match"].to_numpy(), **args)

    pd.testing.assert_frame_equal(frame.iloc[positions], expected)


def test_duplicate_matches_are_counted_once():
    left, right, matches = make_data()
    twice = pd.concat([matches, matches], ignore_index=True)

    frame = SimilarityFeatures(SimilarityMap(SMAP)).pairwise_similarity_dataframe(
        left, right, twice, "id", "id"
    )

    assert len(frame) == 12 * 12
    assert frame["match"].sum() == 12


@pytest.mark.parametrize("model_kind", ["gb", "logit"])
def test_pipeline_does_not_depend_on_chunk_size(tmp_path, model_kind):
    def run(pairs_per_chunk):
        pipe = BaselineTrainingPipe(
            model_name=f"m_{pairs_per_chunk}",
            similarity_map=SMAP,
            training_data=make_data(seed=0),
            validation_data=make_data(seed=1),
            testing_data=make_data(seed=2),
            model_kind=model_kind,
            mismatch_share_fit=0.5,
            base_dir=tmp_path,
            pairs_per_chunk=pairs_per_chunk,
        )
        model = pipe.execute()
        fitted = (
            model.model.feature_importances_.tolist() if model_kind == "gb"
            else np.asarray(model.result.params).tolist()
        )
        return pipe.metrics_train_, pipe.metrics_test_, pipe.metrics_val_, pipe.best_threshold_, fitted

    assert run(7) == run(10**6)


@pytest.mark.parametrize("ids", [None, "id"])
def test_suggest_does_not_depend_on_chunk_size(ids):
    left, right, matches = make_data()
    model = GradientBoostingModel()
    model.model.set_params(n_estimators=20)
    model.fit(SimilarityFeatures(SimilarityMap(SMAP)).pairwise_similarity_dataframe(
        left, right, matches, "id", "id"
    ))
    model.similarity_map = SimilarityMap(SMAP)

    # With real IDs, the result is ordered by ID, not by row
    kwargs = dict(count=3, left_id_col=ids, right_id_col=ids)
    pd.testing.assert_frame_equal(
        model.suggest(left, right, pairs_per_chunk=5, **kwargs),
        model.suggest(left, right, **kwargs),
    )


def test_feature_selection_does_not_depend_on_chunk_size():
    def run(pairs_per_chunk):
        result = FeatureSelector(
            SMAP, make_data(n=20), cv=2, Cs=3, l1_ratios=(0.9,), n_jobs=1,
            pairs_per_chunk=pairs_per_chunk,
        ).execute()
        return (
            result.updated_similarity_map,
            result.selected_feature_columns,
            result.coef_by_feature.to_dict(),
            result.meta,
        )

    assert run(7) == run(10**6)

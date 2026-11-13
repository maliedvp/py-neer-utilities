"""synth_mismatches: row labels, n_cols, random_state, candidate_weights, variants, fallback."""
import random

import numpy as np
import pandas as pd
import pytest

from neer_match_utilities.prepare import synth_mismatches
from neer_match_utilities.split import split_test_train

ARGS = dict(str_metric="levenshtein", str_similarity_range=(0.0, 1.0), pct_diff_range=(0.1, 0.3))


@pytest.fixture
def right():
    """Forty records, a unique key, two string columns and a numeric one; labels 10, 20, ..."""
    n = 40
    return pd.DataFrame({
        "key": [f"k{i}" for i in range(n)],
        "city": [f"city{i % 13}" for i in range(n)],
        "job": [f"job{i % 7}" for i in range(n)],
        "income": [100.0 + 10 * i for i in range(n)],
    }, index=range(10, 10 * (n + 1), 10))


def sources(right, out):
    """Pairs (synthetic row, its source row), found through the unchanged key."""
    by_key = right.reset_index().set_index("key")
    return [(row, right.loc[by_key.at[row["key"], "index"]]) for _, row in out.iloc[len(right):].iterrows()]


def test_row_labels_are_kept(right):
    out = synth_mismatches(right, ["key"], ["city", "job"], n_cols=1, random_state=0, **ARGS)
    pd.testing.assert_frame_equal(out.loc[right.index], right)
    assert (out.index[len(right):] > right.index.max()).all()
    assert out.index.is_unique


def test_range_index_as_before(right):
    r = right.reset_index(drop=True)
    out = synth_mismatches(r, ["key"], ["city"], n_cols=1, random_state=0, **ARGS)
    assert list(out.index) == list(range(len(out)))


@pytest.mark.parametrize("n_cols", [1, 2])
def test_n_cols_columns_change_per_row(right, n_cols):
    out = synth_mismatches(right, ["key"], ["city", "job", "income"], n_cols=n_cols, random_state=1, **ARGS)
    pairs = sources(right, out)
    assert pairs
    for new, orig in pairs:
        changed = [c for c in ("city", "job", "income") if new[c] != orig[c]]
        assert len(changed) == n_cols


def test_random_state_is_reproducible_and_local(right):
    random.seed(1); np.random.seed(1)
    a = synth_mismatches(right, ["key"], ["city", "job"], n_cols=1, sample_share=0.5, random_state=3, **ARGS)
    random.seed(2); np.random.seed(2)
    b = synth_mismatches(right, ["key"], ["city", "job"], n_cols=1, sample_share=0.5, random_state=3, **ARGS)
    c = synth_mismatches(right, ["key"], ["city", "job"], n_cols=1, sample_share=0.5, random_state=4, **ARGS)
    assert a.equals(b)
    assert not a.equals(c)


def test_frequency_weights_draw_common_values_more_often():
    n = 200
    r = pd.DataFrame({"key": [f"k{i}" for i in range(n)],
                      "name": ["COMMON"] * 150 + [f"RARE{i}" for i in range(50)]})
    def share_common(weights):
        out = synth_mismatches(r, ["key"], ["name"], n_cols=1, random_state=0, candidate_weights=weights, **ARGS)
        new = out.iloc[n:]
        rare_sources = new["key"].isin(r["key"].iloc[150:])  # sources whose own value is rare
        return (new.loc[rare_sources, "name"] == "COMMON").mean()
    assert share_common("uniform") < 0.15
    assert share_common("frequency") > 0.5


def test_variants_are_never_used_as_replacement():
    r = pd.DataFrame({"key": [f"k{i}" for i in range(30)],
                      "first": ["JOHANN", "JOHANNES", "JOHAN", "HANS", "JAN"] * 6})
    variants = {"first": {"JOHANN": {"JOHANNES", "JOHAN"}, "JOHANNES": {"JOHANN"}, "JOHAN": {"JOHANN"}}}
    for seed in range(10):
        out = synth_mismatches(r, ["key"], ["first"], n_cols=1, n_mismatches=3, random_state=seed,
                               variants=variants, **ARGS)
        for new, orig in sources(r, out):
            assert new["first"] not in variants["first"].get(orig["first"], set())


def test_fallback_skip_leaves_the_cell(right):
    # no other value is within this similarity range, so nothing qualifies
    kw = dict(ARGS, str_similarity_range=(0.0, 0.0))
    out = synth_mismatches(right, ["key"], ["city"], n_cols=1, random_state=0, fallback="skip", **kw)
    assert len(out) == len(right)  # unchanged copies are duplicates and are dropped
    out = synth_mismatches(right, ["key"], ["city"], n_cols=1, random_state=0, fallback="random", **kw)
    assert len(out) > len(right)


@pytest.fixture
def persons():
    """Standardized and written names that belong together, with sex and commonness."""
    std = ["JOHAN", "JOHANNES", "KARL", "KARIN", "MARIA", "MARTA", "ANNA", "ANNIKA", "ERIK", "ERIKA"] * 4
    sex = [1, 1, 1, 2, 2, 2, 2, 2, 1, 2] * 4
    return pd.DataFrame({
        "key": [f"k{i}" for i in range(40)],
        "sex": sex,
        "first_name_std": std,
        "first_name": [s.capitalize() + "." for s in std],   # as written, derived from the standard form
        "commonness": [len(s) / 10 for s in std],
    })


def test_linked_columns_come_from_the_donor(persons):
    out = synth_mismatches(persons, ["key", "sex"], ["first_name_std"], n_cols=1, random_state=0,
                           linked_columns={"first_name_std": ["first_name", "commonness"]}, **ARGS)
    new = out.iloc[len(persons):]
    assert len(new) > 0
    # every synthetic row is internally consistent: written name and commonness fit the new standard name
    assert (new["first_name"] == new["first_name_std"].str.capitalize() + ".").all()
    assert np.allclose(new["commonness"], new["first_name_std"].str.len() / 10)


def test_without_linked_columns_rows_can_be_inconsistent(persons):
    out = synth_mismatches(persons, ["key", "sex"], ["first_name_std"], n_cols=1, random_state=0, **ARGS)
    new = out.iloc[len(persons):]
    assert (new["first_name"] != new["first_name_std"].str.capitalize() + ".").all()


def test_within_draws_from_the_same_group(persons):
    by_name_sex = dict(zip(persons["first_name_std"], persons["sex"]))
    for seed in range(5):
        out = synth_mismatches(persons, ["key", "sex"], ["first_name_std"], n_cols=1, random_state=seed,
                               within=["sex"], linked_columns={"first_name_std": ["first_name"]}, **ARGS)
        new = out.iloc[len(persons):]
        assert len(new) > 0
        assert (new["first_name_std"].map(by_name_sex) == new["sex"]).all()


def test_linked_and_within_are_checked(persons):
    with pytest.raises(ValueError, match="keys of linked_columns"):
        synth_mismatches(persons, ["key"], ["first_name_std"], n_cols=1,
                         linked_columns={"first_name": ["commonness"]}, **ARGS)
    with pytest.raises(ValueError, match="fixed"):
        synth_mismatches(persons, ["key", "first_name"], ["first_name_std"], n_cols=1,
                         linked_columns={"first_name_std": ["first_name"]}, **ARGS)
    with pytest.raises(ValueError, match="not in right"):
        synth_mismatches(persons, ["key"], ["first_name_std"], n_cols=1, within=["parish"], **ARGS)


def test_invalid_options(right):
    with pytest.raises(ValueError, match="candidate_weights"):
        synth_mismatches(right, ["key"], ["city"], n_cols=1, candidate_weights="zipf", **ARGS)
    with pytest.raises(ValueError, match="fallback"):
        synth_mismatches(right, ["key"], ["city"], n_cols=1, fallback="none", **ARGS)


def test_after_split_original_labels_are_missing(right):
    left = right.copy()
    matches = pd.DataFrame({"left": right.index, "right": right.index})
    lt, rt, mt, *_ = split_test_train(left, right, matches, 0.25, 0.0, random_state=0)
    out = synth_mismatches(rt, ["key"], ["city"], n_cols=1, random_state=0, **ARGS)
    new = out.index[len(rt):]
    assert len(new) > 0
    assert out.loc[new, "index_original"].isna().all()
    assert out.loc[rt.index, "index_original"].tolist() == rt["index_original"].tolist()

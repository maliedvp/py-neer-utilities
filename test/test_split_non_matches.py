"""split_test_train(non_matches=..., drop_partner_share=...) and pairs_in_subset.

Labelled non-matches: pairs checked by hand and known not to match. Their
records are kept (also those without any match), and both records of a pair
land in the same subset, so that the pair is a non-match inside it.

Partner removal: for a share of the test matches the right record is removed,
so that the left record has no counterpart and "no match" is the right answer.
"""
import warnings

import pandas as pd
import pytest

from neer_match_utilities.split import pairs_in_subset, split_test_train

SUBSETS = ("train", "validation", "test")


@pytest.fixture
def data():
    """30 one-to-one matches; left 130-133 and right 530-531 only in non_matches; 134-139 / 532-539 in neither."""
    left = pd.DataFrame({"entity": [f"L{i}" for i in range(40)]}, index=range(100, 140))
    right = pd.DataFrame({"entity": [f"R{i}" for i in range(40)]}, index=range(500, 540))
    matches = pd.DataFrame({"left": range(100, 130), "right": range(500, 530)})
    non_matches = pd.DataFrame({
        "left": [100, 102, 130, 131, 132, 133],
        "right": [501, 503, 500, 530, 531, 505],
    })
    return left, right, matches, non_matches


def by_subset(result):
    lt, rt, mt, lv, rv, mv, ls, rs, ms = result
    return {"train": (lt, rt, mt), "validation": (lv, rv, mv), "test": (ls, rs, ms)}


def original(df):
    return set(df["index_original"])


def original_matches(left, right, matches):
    lo, ro = left["index_original"].to_dict(), right["index_original"].to_dict()
    return {(lo[l], ro[r]) for l, r in zip(matches["left"], matches["right"])}


def test_without_non_matches_unmatched_records_are_dropped(data):
    left, right, matches, _ = data
    s = by_subset(split_test_train(left, right, matches, random_state=1))
    kept_l = set().union(*(original(l) for l, _, _ in s.values()))
    kept_r = set().union(*(original(r) for _, r, _ in s.values()))
    assert kept_l == set(range(100, 130)) and kept_r == set(range(500, 530))


@pytest.mark.parametrize("seed", range(10))
def test_records_of_non_matches_are_kept(data, seed):
    left, right, matches, nm = data
    s = by_subset(split_test_train(left, right, matches, 0.3, 0.2, random_state=seed, non_matches=nm))
    kept_l = set().union(*(original(l) for l, _, _ in s.values()))
    kept_r = set().union(*(original(r) for _, r, _ in s.values()))
    assert kept_l == set(range(100, 134))
    assert kept_r == set(range(500, 532))


@pytest.mark.parametrize("seed", range(10))
def test_each_non_match_pair_in_one_subset(data, seed):
    left, right, matches, nm = data
    s = by_subset(split_test_train(left, right, matches, 0.3, 0.2, random_state=seed, non_matches=nm))
    found = []
    for name, (l, r, m) in s.items():
        inside = pairs_in_subset(nm, l, r)
        # mapped back, the pairs are the labelled ones, and none is listed as a match
        lo, ro = l["index_original"], r["index_original"]
        found += [(lo[a], ro[b]) for a, b in zip(inside["left"], inside["right"])]
        assert not set(zip(inside["left"], inside["right"])) & set(zip(m["left"], m["right"]))
    assert sorted(found) == sorted(zip(nm["left"], nm["right"]))


@pytest.mark.parametrize("seed", range(10))
def test_no_record_in_two_subsets_and_all_matches_kept(data, seed):
    left, right, matches, nm = data
    s = by_subset(split_test_train(left, right, matches, 0.3, 0.2, random_state=seed, non_matches=nm))
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not original(s[a][0]) & original(s[b][0])
        assert not original(s[a][1]) & original(s[b][1])
    pairs = set().union(*(original_matches(*v) for v in s.values()))
    assert pairs == set(zip(matches["left"], matches["right"]))
    assert sum(len(v[2]) for v in s.values()) == len(matches)


def test_reproducible(data):
    left, right, matches, nm = data
    a = split_test_train(left, right, matches, random_state=7, non_matches=nm)
    b = split_test_train(left, right, matches, random_state=7, non_matches=nm)
    assert all(x.equals(y) for x, y in zip(a, b))


def test_non_matches_must_refer_to_existing_rows(data):
    left, right, matches, nm = data
    bad = pd.concat([nm, pd.DataFrame({"left": [999], "right": [500]})])
    with pytest.raises(ValueError, match="not in"):
        split_test_train(left, right, matches, non_matches=bad)


def test_pair_cannot_be_match_and_non_match(data):
    left, right, matches, nm = data
    bad = pd.concat([nm, pd.DataFrame({"left": [100], "right": [500]})])
    with pytest.raises(ValueError, match="both"):
        split_test_train(left, right, matches, non_matches=bad)


def test_non_matches_need_left_and_right(data):
    left, right, matches, nm = data
    with pytest.raises(ValueError, match="columns"):
        split_test_train(left, right, matches, non_matches=nm.rename(columns={"left": "l"}))


def test_pairs_in_subset_needs_a_subset(data):
    left, right, _, nm = data
    with pytest.raises(ValueError, match="index_original"):
        pairs_in_subset(nm, left, right)


# --- drop_partner_share -------------------------------------------------------

@pytest.mark.parametrize("keep_components", [False, True])
def test_share_zero_changes_nothing(data, keep_components):
    left, right, matches, _ = data
    a = split_test_train(left, right, matches, 0.4, 0.2, random_state=3, keep_components=keep_components)
    b = split_test_train(left, right, matches, 0.4, 0.2, random_state=3, keep_components=keep_components,
                         drop_partner_share=0.0)
    assert all(x.equals(y) for x, y in zip(a, b))


@pytest.mark.parametrize("keep_components", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_partners_removed_from_test_only(data, keep_components, seed):
    left, right, matches, _ = data
    base = by_subset(split_test_train(left, right, matches, 0.4, 0.2, random_state=seed,
                                      keep_components=keep_components))
    cut = by_subset(split_test_train(left, right, matches, 0.4, 0.2, random_state=seed,
                                     keep_components=keep_components, drop_partner_share=0.5))
    for name in ("train", "validation"):
        assert all(x.equals(y) for x, y in zip(base[name], cut[name]))
    (lb, rb, mb), (lc, rc, mc) = base["test"], cut["test"]
    n_removed = int(round(0.5 * len(mb)))
    # left records unchanged, right records and pairs fewer
    assert original(lc) == original(lb)
    assert len(original(rb) - original(rc)) == n_removed
    assert len(mc) == len(mb) - n_removed
    kept = original_matches(lc, rc, mc)
    assert kept <= original_matches(lb, rb, mb)
    # the removed pairs' left records stay, without a match
    orphans = {l for l, r in original_matches(lb, rb, mb) - kept}
    assert len(orphans) == n_removed and orphans <= original(lc)
    assert not orphans & {l for l, _ in kept}


def test_right_records_of_non_matches_are_not_removed(data):
    left, right, matches, nm = data
    for seed in range(10):
        s = by_subset(split_test_train(left, right, matches, 0.5, 0.1, random_state=seed,
                                       non_matches=nm, drop_partner_share=0.6))
        l, r, _ = s["test"]
        for a, b in zip(nm["left"], nm["right"]):
            if a in original(l):
                assert b in original(r)


def test_warns_if_too_few_pairs_can_lose_their_partner(data):
    left, right, matches, nm = data
    with pytest.warns(UserWarning, match="drop_partner_share"):
        split_test_train(left, right, matches, 0.5, 0.1, random_state=0, non_matches=nm, drop_partner_share=1.0)


def test_share_must_be_a_share(data):
    left, right, matches, _ = data
    with pytest.raises(ValueError, match="drop_partner_share"):
        split_test_train(left, right, matches, drop_partner_share=1.5)


def test_no_warning_for_one_to_one_data(data):
    left, right, matches, _ = data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        split_test_train(left, right, matches, 0.5, 0.1, random_state=0, drop_partner_share=1.0)

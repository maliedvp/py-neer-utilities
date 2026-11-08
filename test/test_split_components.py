"""split_test_train(keep_components=True): pairs that share a record stay together.

The data are chains of one entity across volumes (A 1922 -> B 1925 -> C 1926),
completed by SetupData.data_preparation_cs, which adds the pairs a chain implies
(A-C) and the pair of a record in both Left and Right with itself (B-B). Drawn
one by one, the pairs of a chain land in different subsets, and inside a subset
a true match whose pair went elsewhere counts as a non-match.
"""
import itertools

import pandas as pd
import pytest

from neer_match_utilities.panel import SetupData
from neer_match_utilities.split import split_test_train

SUBSETS = ("train", "validation", "test")


@pytest.fixture
def chains():
    """Twelve entities seen in 2, 3 or 4 consecutive volumes; links between neighbours."""
    rows, links = [], []
    for e, length in enumerate([2, 3, 4, 2, 3, 2, 4, 3, 2, 2, 3, 2]):
        ids = [f"e{e}v{v}" for v in range(length)]
        rows += [{"id_unique": i, "entity": e, "name": f"name {e}"} for i in ids]
        links += list(zip(ids[:-1], ids[1:]))
    df = pd.DataFrame(rows)
    m = pd.DataFrame(links, columns=["left", "right"])
    left, right, m = SetupData(matches=m).data_preparation_cs(
        df_left=df[df["id_unique"].isin(m["left"])],
        df_right=df[df["id_unique"].isin(m["right"])],
        unique_id="id_unique",
    )
    pos_l = {v: i for i, v in enumerate(left["id_unique"])}
    pos_r = {v: i for i, v in enumerate(right["id_unique"])}
    matches = pd.DataFrame({"left": m["left"].map(pos_l), "right": m["right"].map(pos_r)})
    return left, right, matches


def subsets(result):
    lt, rt, mt, lv, rv, mv, ls, rs, ms = result
    return {"train": (lt, rt, mt), "validation": (lv, rv, mv), "test": (ls, rs, ms)}


def hidden_matches(left, right, matches):
    """Pairs of the subset's Left x Right of the same entity that are not listed as matches."""
    listed = set(zip(matches["left"], matches["right"]))
    return [
        (i, j)
        for i, j in itertools.product(range(len(left)), range(len(right)))
        if left["entity"].iat[i] == right["entity"].iat[j] and (i, j) not in listed
    ]


def test_example_chain():
    """The chain of the docstring: SetupData gives four pairs, all of one component."""
    df = pd.DataFrame({"id_unique": ["A", "B", "C"]})
    m = pd.DataFrame({"left": ["A", "B"], "right": ["B", "C"]})
    left, right, m = SetupData(matches=m).data_preparation_cs(
        df_left=df[df["id_unique"].isin(m["left"])],
        df_right=df[df["id_unique"].isin(m["right"])],
        unique_id="id_unique",
    )
    assert left["id_unique"].tolist() == ["A", "B"]
    assert right["id_unique"].tolist() == ["B", "C"]
    assert set(zip(m["left"], m["right"])) == {("A", "B"), ("A", "C"), ("B", "C"), ("B", "B")}


@pytest.mark.parametrize("seed", range(20))
def test_no_hidden_matches(chains, seed):
    left, right, matches = chains
    result = split_test_train(left, right, matches, test_ratio=0.4, validation_ratio=0.2,
                              random_state=seed, keep_components=True)
    for name, (l, r, m) in subsets(result).items():
        assert hidden_matches(l, r, m) == [], name


def test_pairs_drawn_one_by_one_hide_matches(chains):
    """The problem the option solves: without it, some split hides true matches."""
    left, right, matches = chains
    hidden = 0
    for seed in range(20):
        result = split_test_train(left, right, matches, test_ratio=0.4, validation_ratio=0.2,
                                  random_state=seed)
        hidden += sum(len(hidden_matches(*s)) for s in subsets(result).values())
    assert hidden > 0


@pytest.mark.parametrize("seed", range(10))
def test_records_in_one_subset_only(chains, seed):
    left, right, matches = chains
    result = split_test_train(left, right, matches, test_ratio=0.4, validation_ratio=0.2,
                              random_state=seed, keep_components=True)
    s = subsets(result)
    for side in (0, 1):
        for a, b in itertools.combinations(SUBSETS, 2):
            assert set(s[a][side]["id_unique"]).isdisjoint(s[b][side]["id_unique"]), (side, a, b)


@pytest.mark.parametrize("seed", range(10))
def test_sizes(chains, seed):
    left, right, matches = chains
    result = split_test_train(left, right, matches, test_ratio=0.4, validation_ratio=0.2,
                              random_state=seed, keep_components=True)
    s = subsets(result)
    largest = max(sum(left["entity"] == e) * sum(right["entity"] == e) for e in left["entity"].unique())
    assert sum(len(m) for _, _, m in s.values()) == len(matches)
    for name, ratio in (("test", 0.4), ("validation", 0.2)):
        target = int(round(ratio * len(matches)))
        assert target <= len(s[name][2]) < target + largest, name


def test_validation_ratio_zero(chains):
    left, right, matches = chains
    result = split_test_train(left, right, matches, test_ratio=0.5, validation_ratio=0.0,
                              random_state=1, keep_components=True)
    _, rv, mv = subsets(result)["validation"]
    assert len(mv) == 0 and len(rv) == 0


def test_reproducible_and_random(chains, split_signature):
    left, right, matches = chains
    kw = dict(test_ratio=0.4, validation_ratio=0.2, keep_components=True)
    assert split_signature(left, right, matches, random_state=3, **kw) == \
        split_signature(left, right, matches, random_state=3, **kw)
    assert len({split_signature(left, right, matches, random_state=s, **kw) for s in range(10)}) > 1


def test_one_to_one_sizes_exact():
    """With one pair per component the requested sizes are met exactly."""
    left = pd.DataFrame({"v": range(10)}, index=range(100, 110))
    right = pd.DataFrame({"w": range(10)}, index=range(200, 210))
    matches = pd.DataFrame({"left": range(100, 110), "right": range(200, 210)})
    result = split_test_train(left, right, matches, test_ratio=0.3, validation_ratio=0.2,
                              random_state=0, keep_components=True)
    s = subsets(result)
    assert (len(s["train"][2]), len(s["validation"][2]), len(s["test"][2])) == (5, 2, 3)

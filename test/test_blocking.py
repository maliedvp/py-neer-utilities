"""Blocking (1.2.12): candidate pairs, and their use in training, evaluation and prediction."""
import itertools

import numpy as np
import pandas as pd
import pytest
from neer_match.similarity_map import SimilarityMap
from rapidfuzz.distance import JaroWinkler

from neer_match_utilities.baseline_training import BaselineTrainingPipe
from neer_match_utilities.blocking import Blocking, blocked_metrics, candidate_pairs, tune_threshold_blocked
from neer_match_utilities.baseline_models import classification_metrics
from neer_match_utilities.panel import GenerateID
from neer_match_utilities.similarity_features import SimilarityFeatures
from neer_match_utilities.thresholds import tune_threshold

SMAP = {"name": ["levenshtein", "jaro_winkler"]}
METRICS = ["TP", "FP", "TN", "FN", "Accuracy", "Recall", "Precision", "F1", "MCC"]


def make_data(n=30, seed=0, year_shift=(0, 0, 1, 3)):
    """Right row j is an edited copy of left row order[j], same group, birth year shifted."""
    rng = np.random.default_rng(seed)
    names = ["".join(rng.choice(list("abcdefgh"), 6)) for _ in range(n)]
    group = rng.integers(0, 3, n)
    year = rng.integers(1840, 1850, n)
    order = rng.permutation(n)
    left = pd.DataFrame({"id": [f"L{i}" for i in range(n)], "name": names, "group": group,
                         "year": year, "const": 1})
    right = pd.DataFrame({
        "id": [f"R{i}" for i in range(n)],
        "name": [names[i][:-1] + "q" for i in order],
        "group": group[order],
        "year": year[order] + rng.choice(year_shift, n),
        "const": 1,
    })
    matches = pd.DataFrame({"left": order, "right": np.arange(n)})
    return left, right, matches


def brute_force(left, right, blocking):
    out = []
    for i, j in itertools.product(range(len(left)), range(len(right))):
        ok = True
        for spec in blocking.exact:
            lc, rc = spec.split("~") if "~" in spec else (spec, spec)
            a, b = left[lc].iat[i], right[rc].iat[j]
            ok &= not (pd.isna(a) or pd.isna(b)) and a == b
        for spec, tol in blocking.within.items():
            lc, rc = spec.split("~") if "~" in spec else (spec, spec)
            a, b = left[lc].iat[i], right[rc].iat[j]
            ok &= not (pd.isna(a) or pd.isna(b)) and abs(a - b) <= tol
        for spec, t in blocking.min_similarity.items():
            lc, rc = spec.split("~") if "~" in spec else (spec, spec)
            a, b = left[lc].iat[i], right[rc].iat[j]
            ok &= not (pd.isna(a) or pd.isna(b)) and JaroWinkler.similarity(str(a), str(b)) >= t
        if ok:
            out.append((i, j))
    return out


def random_frames(seed):
    rng = np.random.default_rng(seed)
    def frame(n):
        year = rng.integers(1840, 1846, n).astype(float)
        year[rng.random(n) < 0.1] = np.nan
        return pd.DataFrame({
            "sex": rng.choice(np.array(["m", "f", None], dtype=object), n, p=[0.45, 0.45, 0.1]),
            "parish": rng.integers(0, 3, n),
            "year": year,
            "height": rng.normal(170, 5, n),
            "name": ["".join(rng.choice(list("abc"), 4)) for _ in range(n)],
        })
    left, right = frame(25), frame(20)
    return left, right.rename(columns={"parish": "parish_r"})


RULES = [
    Blocking(exact=["sex"]),
    Blocking(exact=["sex", "parish~parish_r"]),
    Blocking(within={"year": 2}),
    Blocking(within={"year": 0}),
    Blocking(within={"height": 3.5}),
    Blocking(exact=["sex"], within={"year": 1, "height": 6}),
    Blocking(min_similarity={"name": 0.8}),
    Blocking(exact=["parish~parish_r"], within={"year": 2}, min_similarity={"name": 0.7}),
]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("rule", RULES, ids=range(len(RULES)))
def test_candidates_equal_brute_force(rule, seed):
    left, right = random_frames(seed)
    got = candidate_pairs(left, right, rule)
    assert list(got.columns) == ["left", "right"]
    assert list(zip(got["left"], got["right"])) == brute_force(left, right, rule)


def test_rule_checks():
    with pytest.raises(ValueError):
        Blocking()
    with pytest.raises(ValueError):
        Blocking(within={"year": -1})
    with pytest.raises(ValueError):
        Blocking(min_similarity={"name": 1.5})
    left, right = random_frames(0)
    with pytest.raises(KeyError):
        Blocking(exact=["parish"]).candidates(left, right)  # right names it parish_r
    assert Blocking(exact="sex").exact == ("sex",)
    assert Blocking(exact=["sex", "parish~parish_r"], within={"year": 2}).columns() == (
        ["sex", "parish", "year"], ["sex", "parish_r", "year"]
    )


def test_function_as_blocking():
    left, right = random_frames(0)
    pairs = candidate_pairs(left, right, lambda l, r: pd.DataFrame({"left": [3, 1, 1], "right": [0, 2, 2]}))
    assert list(zip(pairs["left"], pairs["right"])) == [(1, 2), (3, 0)]
    with pytest.raises(ValueError):
        candidate_pairs(left, right, lambda l, r: pd.DataFrame({"left": [len(l)], "right": [0]}))
    with pytest.raises(ValueError):
        candidate_pairs(left, right, lambda l, r: pd.DataFrame({"a": [0]}))
    assert candidate_pairs(left, right, None) is None


@pytest.mark.parametrize("matches_are_indices", [True, False])
@pytest.mark.parametrize("pairs_per_chunk", [1, 7, 10**6])
def test_pairs_with_candidates_are_rows_of_the_full_frame(matches_are_indices, pairs_per_chunk):
    left, right, matches = make_data()
    if not matches_are_indices:
        matches = pd.DataFrame({"left": left["id"].to_numpy()[matches["left"]],
                                "right": right["id"].to_numpy()[matches["right"]]})
    feats = SimilarityFeatures(SimilarityMap(SMAP))
    cand = candidate_pairs(left, right, Blocking(exact=["group"], within={"year": 2}))
    full = feats.pairs(left, right, matches, "id", "id", matches_are_indices=matches_are_indices)
    blocked = feats.pairs(left, right, matches, "id", "id", matches_are_indices=matches_are_indices,
                          candidates=cand)

    expected = full.frame().iloc[cand["left"].to_numpy() * len(right) + cand["right"].to_numpy()]
    frame = blocked.frame()
    pd.testing.assert_frame_equal(frame, expected.reset_index(drop=True))
    pd.testing.assert_frame_equal(pd.concat(blocked.chunks(pairs_per_chunk)), frame)
    np.testing.assert_array_equal(blocked.match_flags(pairs_per_chunk), frame["match"].to_numpy())
    positions = np.random.default_rng(0).permutation(len(blocked))[:9]
    pd.testing.assert_frame_equal(blocked.rows(positions), frame.iloc[positions])
    assert blocked.blocked and not full.blocked
    assert len(blocked) == len(cand) and blocked.n_all == full.n_all == len(left) * len(right)
    assert blocked.n_matches == full.n_matches == len(matches)
    # Shifts of 3 years fall outside the rule: some matches are not candidates
    assert 0 < frame["match"].sum() < len(matches)


def test_blocked_metrics_count_the_excluded_pairs():
    y_true = np.array([1, 1, 0, 0, 0, 1])
    y_hat = np.array([1, 0, 1, 0, 0, 1])
    m = blocked_metrics(y_true, y_hat, n_pairs=40, n_matches=5)
    # 2 true matches and 32 non-matches lie outside the 6 candidates
    assert (m["TP"], m["FP"], m["TN"], m["FN"]) == (2, 1, 2 + 32, 1 + 2)
    assert m["Pairs"] == 40 and m["Candidates"] == 6 and m["BlockingRecall"] == pytest.approx(3 / 5)
    # With every pair a candidate, the scores are classification_metrics'
    full = blocked_metrics(y_true, y_hat, n_pairs=6, n_matches=3)
    for k in METRICS:
        assert full[k] == pytest.approx(classification_metrics(y_true, y_hat)[k])


def test_tuning_without_excluded_pairs_is_tune_threshold():
    rng = np.random.default_rng(0)
    labels = rng.random(300) < 0.1
    proba = np.clip(labels * 0.5 + rng.random(300) * 0.6, 0, 1)
    for metric in ("mcc", "f1"):
        t, stats = tune_threshold_blocked(labels, proba, n_pairs=300, n_matches=int(labels.sum()), metric=metric)
        t0, stats0 = tune_threshold(labels, proba, metric=metric)
        assert t == t0
        for k in METRICS:
            assert stats[k] == pytest.approx(stats0[k])


def pipe(tmp_path, name, blocking, model_kind="logit"):
    p = BaselineTrainingPipe(
        model_name=name, similarity_map=SMAP,
        training_data=make_data(seed=0), validation_data=make_data(seed=1), testing_data=make_data(seed=2),
        model_kind=model_kind, base_dir=tmp_path, blocking=blocking,
    )
    p.execute()
    return p


def test_pipe_with_every_pair_a_candidate_equals_no_blocking(tmp_path):
    plain = pipe(tmp_path, "plain", None)
    every = pipe(tmp_path, "every", Blocking(exact=["const"]))
    for a, b in ((plain.metrics_train_, every.metrics_train_), (plain.metrics_test_, every.metrics_test_)):
        for k in METRICS:
            assert a[k] == pytest.approx(b[k])
        assert b["BlockingRecall"] == 1.0 and b["Candidates"] == b["Pairs"] == 30 * 30
    np.testing.assert_allclose(plain.model_.result.params, every.model_.result.params)


@pytest.mark.parametrize("model_kind", ["logit", "gb"])
def test_pipe_counts_excluded_matches_as_missed(tmp_path, model_kind):
    rule = Blocking(exact=["group"], within={"year": 2})
    p = pipe(tmp_path, f"blocked_{model_kind}", rule, model_kind)
    left, right, matches = make_data(seed=2)
    cand = candidate_pairs(left, right, rule)
    in_cand = set(zip(cand["left"], cand["right"])) & set(zip(matches["left"], matches["right"]))
    m = p.metrics_test_
    assert m["TP"] + m["FP"] + m["TN"] + m["FN"] == 30 * 30
    assert m["Candidates"] == len(cand) and m["BlockingRecall"] == pytest.approx(len(in_cand) / 30)
    assert m["FN"] >= 30 - len(in_cand)
    perf = pd.read_csv(tmp_path / f"blocked_{model_kind}" / "performance.csv")
    assert {"Pairs", "Candidates", "BlockingRecall"} <= set(perf.columns)
    if model_kind == "gb":
        assert p.metrics_val_["Pairs"] == 30 * 30


def test_pipe_without_candidates_raises(tmp_path):
    with pytest.raises(ValueError, match="no candidate"):
        pipe(tmp_path, "none", lambda l, r: pd.DataFrame({"left": [], "right": []}))


def test_suggest_with_blocking_takes_the_top_candidates(tmp_path):
    model = pipe(tmp_path, "s", None).model_
    left, right, _ = make_data(seed=3)
    rule = Blocking(exact=["group"])
    got = model.suggest(left, right, count=2, blocking=rule, pairs_per_chunk=7)

    cand = candidate_pairs(left, right, rule)
    frame = SimilarityFeatures(SimilarityMap(SMAP)).pairs(
        left.assign(_row_id=range(30)), right.assign(_row_id=range(30)), pd.DataFrame({"left": [], "right": []}),
        "_row_id", "_row_id", matches_are_indices=False, candidates=cand,
    ).frame()
    frame["prediction"] = model.predict_proba(frame)
    expected = (frame.rename(columns={"_row_id_left": "left", "_row_id_right": "right"})
                .sort_values(["left", "prediction"], ascending=[True, False])
                .groupby("left").head(2)[["left", "right", "prediction"]].reset_index(drop=True))
    pd.testing.assert_frame_equal(got, expected)
    assert set(zip(got["left"], got["right"])) <= set(zip(cand["left"], cand["right"]))
    # Without the rule, suggestions still come from all pairs
    assert len(model.suggest(left, right, count=2)) == 60
    empty = model.suggest(left, right, count=2, blocking=lambda l, r: pd.DataFrame({"left": [], "right": []}))
    assert len(empty) == 0


class LinkByPosition:
    """Stand-in model as in test_generate_id, recording the rule it is given."""

    def __init__(self):
        self.rules = []

    def suggest(self, left, right, count=1, verbose=0, blocking=None):
        self.rules.append(blocking)
        n = min(len(left), len(right))
        return pd.DataFrame({"left": range(n), "right": range(n), "prediction": [1.0] * n})


class NoBlocking:
    def suggest(self, left, right, count=1, verbose=0, **kwargs):
        raise AssertionError("not called")


def test_generate_id_passes_blocking_and_keeps_its_columns():
    panel = pd.DataFrame({"year": [1900, 1900, 1910, 1910], "name": ["A", "B", "A'", "B'"],
                          "parish": [1, 2, 1, 2], "other": 0})
    model, rule = LinkByPosition(), Blocking(exact=["parish"])
    gid = GenerateID(panel, panel_var="pid", time_var="year", model=model,
                     similarity_map={"name": ["levenshtein"]}, prediction_threshold=0.5, blocking=rule)
    assert "parish" in gid.df_panel.columns and "other" not in gid.df_panel.columns
    gid.execute()
    assert model.rules == [rule]
    with pytest.raises(TypeError, match="blocking"):
        GenerateID(panel, panel_var="pid", time_var="year", model=NoBlocking(),
                   similarity_map={"name": ["levenshtein"]}, blocking=rule)


def test_generate_id_blocking_equals_subgroups_when_no_period_is_missing(tmp_path):
    model = pipe(tmp_path, "g", None).model_
    left, right, _ = make_data(seed=4, year_shift=(0,))
    panel = pd.concat([left.assign(t=1), right.assign(t=2)], ignore_index=True)

    def groups(ids):
        return sorted(sorted(ids.loc[ids["pid"] == p, "index"]) for p in ids["pid"].unique())

    kw = dict(panel_var="pid", time_var="t", model=model, similarity_map=SMAP, prediction_threshold=0.5,
              relation="1:1")
    by_rule = GenerateID(panel, blocking=Blocking(exact=["group"]), **kw).execute()
    by_groups = GenerateID(panel, subgroups=["group"], **kw).execute()
    assert groups(by_rule) == groups(by_groups)
    assert sum(len(g) == 2 for g in groups(by_rule)) >= 10  # records were linked


def test_feature_selection_with_every_pair_a_candidate_equals_no_blocking():
    from neer_match_utilities.feature_selection import FeatureSelector
    smap = {"name": ["levenshtein", "jaro_winkler", "prefix"], "year": ["discrete"]}
    def select(blocking):
        return FeatureSelector(similarity_map=smap, training_data=make_data(n=40, seed=5), id_left_col="id",
                               id_right_col="id", cv=2, Cs=4, n_jobs=1, blocking=blocking).execute()
    a, b = select(None), select(Blocking(exact=["const"]))
    assert a.updated_similarity_map == b.updated_similarity_map
    assert a.selected_feature_columns == b.selected_feature_columns

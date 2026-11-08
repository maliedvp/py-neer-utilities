"""
Blocking: which record pairs a model looks at.

Without blocking, training, evaluation and prediction take every pair of
``left x right``. A :class:`Blocking` rule keeps only the candidate pairs that
agree on some columns (``exact``), lie close on others (``within``) or have
similar strings (``min_similarity``), as rule-based record linkage does, e.g.
the same sex, birth parish and birth year within two years.

The same rule serves everywhere, so that a model is trained on the kind of
pairs it later scores:

- :class:`~neer_match_utilities.baseline_training.BaselineTrainingPipe` takes
  its non-matches for fitting from the candidates only and scores only them.
  A true match that the rule excludes counts as missed (a false negative);
  the other excluded pairs count as correctly rejected. The metrics therefore
  describe blocking and model together, on all pairs, and add ``Pairs``,
  ``Candidates`` and ``BlockingRecall`` (the share of the true matches among
  the candidates).
- ``suggest`` of the baseline models and
  :class:`~neer_match_utilities.feature_selection.FeatureSelector` take
  ``blocking`` too, and :class:`~neer_match_utilities.panel.GenerateID` passes
  it on to ``suggest``.

The rule is passed, not a list of pairs: each split computes its own candidates.
Instead of a :class:`Blocking`, any function ``f(left, right)`` that returns the
candidate pairs as row positions (columns ``left`` and ``right``) can be passed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from neer_match_utilities.thresholds import THRESHOLD_GRID, TUNE_METRICS, _counts, _rates


def _columns(spec: str) -> tuple[str, str]:
    """``"col"`` names the column on both sides, ``"left_col~right_col"`` each side's."""
    if "~" in spec:
        left_col, right_col = spec.split("~", 1)
        return left_col, right_col
    return spec, spec


@dataclass(frozen=True)
class Blocking:
    """
    A rule for the candidate pairs of ``left x right``.

    A pair is a candidate if it meets every condition given. Columns are named
    as in a similarity map: ``"col"`` for a column of that name on both sides,
    ``"left_col~right_col"`` for differently named ones. A record with a missing
    value in a column the rule uses has no candidates. Values are compared as
    they are in the frames passed (e.g. after ``Prepare.format``).

    Parameters
    ----------
    exact :
        Columns whose values must be equal.
    within :
        Numeric columns and the largest absolute difference allowed, e.g.
        ``{"birth_year": 2}``. A difference of 0 is the same as ``exact``.
    min_similarity :
        String columns and the smallest Jaro-Winkler similarity allowed
        (0 to 1), e.g. ``{"last_name": 0.7}``, compared on ``str`` values.
        Checked only on the pairs that pass ``exact`` and ``within``; given
        alone, it compares every pair.

    Examples
    --------
    >>> Blocking(exact=["sex", "birth_parish"], within={"birth_year": 2})
    >>> Blocking(exact=["county"], within={"birth_year": 2}, min_similarity={"last_name": 0.7})
    """

    exact: Sequence[str] = ()
    within: Mapping[str, float] = field(default_factory=dict)
    min_similarity: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.exact, str):
            object.__setattr__(self, "exact", (self.exact,))
        object.__setattr__(self, "exact", tuple(self.exact))
        object.__setattr__(self, "within", dict(self.within))
        object.__setattr__(self, "min_similarity", dict(self.min_similarity))
        for col, tol in self.within.items():
            if not (np.isfinite(tol) and tol >= 0):
                raise ValueError(f"within[{col!r}] must be a number >= 0, got {tol!r}")
        for col, t in self.min_similarity.items():
            if not 0 <= t <= 1:
                raise ValueError(f"min_similarity[{col!r}] must lie in [0, 1], got {t!r}")
        if not (self.exact or self.within or self.min_similarity):
            raise ValueError("Blocking needs at least one of exact, within or min_similarity.")

    def columns(self) -> tuple[list[str], list[str]]:
        """The columns the rule reads in left and in right."""
        specs = [*self.exact, *self.within, *self.min_similarity]
        pairs = [_columns(s) for s in specs]
        return list(dict.fromkeys(p[0] for p in pairs)), list(dict.fromkeys(p[1] for p in pairs))

    def candidates(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        """The candidate pairs as row positions, columns ``left`` and ``right``, sorted."""
        n_left, n_right = len(left), len(right)
        lcols, rcols = self.columns()
        for side, frame, cols in (("left", left, lcols), ("right", right, rcols)):
            missing = [c for c in cols if c not in frame.columns]
            if missing:
                raise KeyError(f"Blocking reads {missing} from {side}, which has no such column.")

        L = pd.DataFrame({"left": np.arange(n_left)})
        R = pd.DataFrame({"right": np.arange(n_right)})
        keys = []

        for i, spec in enumerate(self.exact):
            lc, rc = _columns(spec)
            L[f"_e{i}"] = left[lc].to_numpy()
            R[f"_e{i}"] = right[rc].to_numpy()
            keys.append(f"_e{i}")

        bins, numeric = [], []
        for j, (spec, tol) in enumerate(self.within.items()):
            lc, rc = _columns(spec)
            lv = pd.to_numeric(left[lc], errors="coerce").to_numpy(dtype=float)
            rv = pd.to_numeric(right[rc], errors="coerce").to_numpy(dtype=float)
            if tol == 0:
                L[f"_e_w{j}"], R[f"_e_w{j}"] = lv, rv
                keys.append(f"_e_w{j}")
                continue
            # Values within tol of each other fall into the same or a
            # neighbouring bin of width tol; the exact test follows the merge.
            L[f"_v{j}"], R[f"_v{j}"] = lv, rv
            L[f"_b{j}"] = np.floor(lv / tol)
            R[f"_b{j}"] = np.floor(rv / tol)
            bins.append(f"_b{j}")
            numeric.append((f"_v{j}", tol))

        L = L.dropna()
        R = R.dropna()

        if keys or bins:
            if bins:
                shifted = []
                for offsets in itertools.product((-1, 0, 1), repeat=len(bins)):
                    Rs = R.copy()
                    for b, off in zip(bins, offsets):
                        Rs[b] = Rs[b] + off
                    shifted.append(Rs)
                R = pd.concat(shifted, ignore_index=True)
            pairs = L.merge(R, on=keys + bins, suffixes=("_l", "_r"))
            keep = np.ones(len(pairs), dtype=bool)
            for v, tol in numeric:
                keep &= np.abs(pairs[f"{v}_l"].to_numpy() - pairs[f"{v}_r"].to_numpy()) <= tol
            pairs = pairs.loc[keep, ["left", "right"]]
        else:
            pairs = L[["left"]].merge(R[["right"]], how="cross")

        if self.min_similarity and len(pairs):
            keep = np.ones(len(pairs), dtype=bool)
            lpos, rpos = pairs["left"].to_numpy(), pairs["right"].to_numpy()
            for spec, t in self.min_similarity.items():
                lc, rc = _columns(spec)
                lv, rv = left[lc].to_numpy(), right[rc].to_numpy()
                a, b = lv[lpos], rv[rpos]
                present = ~(pd.isna(a) | pd.isna(b))
                sim = np.zeros(len(pairs))
                if present.any():
                    sim[present] = cpdist(
                        [str(s) for s in a[present]], [str(s) for s in b[present]],
                        scorer=JaroWinkler.similarity, workers=1,
                    )
                keep &= present & (sim >= t)
            pairs = pairs.loc[keep]

        return (
            pairs.astype({"left": np.int64, "right": np.int64})
            .drop_duplicates()
            .sort_values(["left", "right"])
            .reset_index(drop=True)
        )


BlockingLike = Union[Blocking, Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame], None]


def candidate_pairs(left: pd.DataFrame, right: pd.DataFrame, blocking: BlockingLike) -> pd.DataFrame | None:
    """
    The candidate pairs of ``left x right`` under ``blocking``, as row positions.

    Returns None for ``blocking=None`` (every pair is a candidate). A function
    passed as ``blocking`` must return columns ``left`` and ``right`` with row
    positions 0..n-1 of the two frames; the pairs are deduplicated and sorted.
    """
    if blocking is None:
        return None
    if isinstance(blocking, Blocking):
        return blocking.candidates(left, right)
    if not callable(blocking):
        raise TypeError("blocking must be a Blocking, a function f(left, right) or None.")
    pairs = blocking(left, right)
    if not isinstance(pairs, pd.DataFrame) or not {"left", "right"}.issubset(pairs.columns):
        raise ValueError("A blocking function must return a DataFrame with columns 'left' and 'right'.")
    pairs = pairs[["left", "right"]]
    for col, n in (("left", len(left)), ("right", len(right))):
        values = pd.to_numeric(pairs[col], errors="coerce")
        if (~values.between(0, n - 1) | (values % 1 != 0)).any():
            raise ValueError(f"A blocking function must return row positions 0..{n - 1} in {col!r}.")
    return (
        pairs.astype(np.int64).drop_duplicates().sort_values(["left", "right"]).reset_index(drop=True)
    )


def blocking_columns(blocking: BlockingLike) -> tuple[list[str], list[str]] | None:
    """The columns a :class:`Blocking` reads; None for a function, which may read any."""
    if blocking is None:
        return [], []
    if isinstance(blocking, Blocking):
        return blocking.columns()
    return None


def _with_excluded(tp, fp, tn, fn, n_candidates, n_pairs, n_matches):
    """Counts with the pairs outside the candidates added as predicted non-matches."""
    matches_in = tp + fn
    excluded_matches = n_matches - matches_in
    excluded_non_matches = (n_pairs - n_candidates) - excluded_matches
    return tp, fp, tn + excluded_non_matches, fn + excluded_matches


def blocked_metrics(y_true: np.ndarray, y_hat: np.ndarray, n_pairs: int, n_matches: int) -> dict:
    """
    Metrics of blocking and model together, on all ``n_pairs`` pairs.

    ``y_true`` and ``y_hat`` hold the candidates' labels and predictions. The
    pairs outside the candidates count as predicted non-matches: the
    ``n_matches - y_true.sum()`` true matches among them as false negatives.
    Adds ``Pairs``, ``Candidates`` and ``BlockingRecall`` to the counts and
    rates of :func:`~neer_match_utilities.baseline_models.classification_metrics`.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_hat = np.asarray(y_hat).astype(bool)
    tp, fp = int(np.sum(y_true & y_hat)), int(np.sum(~y_true & y_hat))
    tn, fn = int(np.sum(~y_true & ~y_hat)), int(np.sum(y_true & ~y_hat))
    return _blocked_scores(tp, fp, tn, fn, len(y_true), n_pairs, n_matches)


def _blocked_scores(tp, fp, tn, fn, n_candidates, n_pairs, n_matches) -> dict:
    matches_in = tp + fn
    tp, fp, tn, fn = _with_excluded(tp, fp, tn, fn, n_candidates, n_pairs, n_matches)
    rates = {name: float(v) for name, v in _rates(tp, fp, tn, fn).items()}
    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn), **rates,
        "Pairs": int(n_pairs),
        "Candidates": int(n_candidates),
        # None, not NaN, where undefined: NaN != NaN would fail the reload check
        "BlockingRecall": float(matches_in / n_matches) if n_matches else None,
    }


def tune_threshold_blocked(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_pairs: int,
    n_matches: int,
    metric: str = "mcc",
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict]:
    """
    The threshold that maximises ``metric`` of blocking and model together.

    As :func:`~neer_match_utilities.thresholds.tune_threshold`, on the
    candidates' labels and probabilities, with the pairs outside the candidates
    counted as in :func:`blocked_metrics`. Of equally good thresholds the
    lowest wins.
    """
    if metric not in TUNE_METRICS:
        raise ValueError(f"metric must be one of {TUNE_METRICS}, not {metric!r}.")
    grid = THRESHOLD_GRID if thresholds is None else np.asarray(thresholds, dtype=float)
    tp, fp, tn, fn = _counts(labels, probabilities, grid)
    n_candidates = len(np.asarray(labels).ravel())
    values = _rates(*_with_excluded(tp, fp, tn, fn, n_candidates, n_pairs, n_matches))[metric.upper()]
    i = int(np.argmax(values))
    return float(grid[i]), _blocked_scores(tp[i], fp[i], tn[i], fn[i], n_candidates, n_pairs, n_matches)

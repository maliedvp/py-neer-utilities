"""
Classification thresholds for match probabilities.

Scores at a threshold and the choice of the threshold that maximises MCC or F1,
on all record pairs of a split at once. The deep learning pipeline uses them to
score its validation pairs after every epoch, to choose each stage's best epoch
and to tune the threshold of its final model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The thresholds tried, as GradientBoostingModel tunes its own
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)

TUNE_METRICS = ("mcc", "f1")


def pair_labels(matches: pd.DataFrame, n_left: int, n_right: int) -> np.ndarray:
    """
    Labels of all ``n_left * n_right`` record pairs, in the order
    ``DLMatchingModel.predict`` scores them: left record by left record.

    ``matches`` holds the row positions of the matched pairs in its first two
    columns, left and right, as neer-match reads them.
    """
    labels = np.zeros(n_left * n_right, dtype=np.int8)
    left = matches.iloc[:, 0].to_numpy(dtype=np.int64)
    right = matches.iloc[:, 1].to_numpy(dtype=np.int64)
    labels[left * n_right + right] = 1
    return labels


def _counts(labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray):
    """TP, FP, TN, FN at each threshold, a pair counting as a match at p >= t."""
    # In float32, as neer-match compares predictions with its threshold
    p = np.asarray(probabilities, dtype=np.float32).ravel()
    t = np.asarray(thresholds, dtype=np.float32)
    labels = np.asarray(labels).ravel().astype(bool)
    pos, neg = np.sort(p[labels]), np.sort(p[~labels])
    tp = len(pos) - np.searchsorted(pos, t, side="left")
    fp = len(neg) - np.searchsorted(neg, t, side="left")
    return tp, fp, len(neg) - fp, len(pos) - tp


def _rates(tp, fp, tn, fn) -> dict:
    """Rates of confusion counts; 0 where a ratio is undefined, as in scikit-learn."""
    tp, fp, tn, fn = (np.asarray(x, dtype=float) for x in (tp, fp, tn, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(denominator > 0, (tp * tn - fp * fn) / denominator, 0.0)
        accuracy = (tp + tn) / (tp + fp + tn + fn)
    return {"Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1": f1, "MCC": mcc}


def scores_at(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """
    TP, FP, TN, FN, Accuracy, Recall, Precision, F1 and MCC at one threshold.
    Precision, Recall, F1 and MCC are 0 where they are undefined.
    """
    tp, fp, tn, fn = (int(x[0]) for x in _counts(labels, probabilities, [threshold]))
    rates = {name: float(value) for name, value in _rates(tp, fp, tn, fn).items()}
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, **rates}


def tune_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "mcc",
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict]:
    """
    The threshold that maximises ``metric`` ('mcc' or 'f1'), with the scores
    at it.

    The thresholds tried are ``THRESHOLD_GRID``, 0.01 to 0.99 in steps of 0.01,
    unless given. Of equally good thresholds the lowest wins, as in
    ``GradientBoostingModel``'s tuning.
    """
    if metric not in TUNE_METRICS:
        raise ValueError(f"metric must be one of {TUNE_METRICS}, not {metric!r}.")
    grid = THRESHOLD_GRID if thresholds is None else np.asarray(thresholds, dtype=float)
    values = _rates(*_counts(labels, probabilities, grid))[metric.upper()]
    best = float(grid[int(np.argmax(values))])
    return best, scores_at(labels, probabilities, best)

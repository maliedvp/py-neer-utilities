"""
Feature selection utilities for similarity-based entity matching.

This module provides tools for selecting the most informative similarity features
from a potentially large set of candidates. It implements a two-stage feature
selection process:

1. **Correlation-based filtering**: Removes highly correlated features, keeping
   the one most correlated with the target variable.
2. **Elastic net regularization**: Uses penalized logistic regression to identify
   features that contribute unique predictive information.

The feature selector is designed to handle extreme class imbalance, which is
common in entity matching tasks where true matches are rare.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Iterable

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import get_scorer
from sklearn.utils.class_weight import compute_sample_weight
from scipy.special import expit
from joblib import Parallel, delayed

import warnings
from sklearn.exceptions import ConvergenceWarning

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from neer_match_utilities.similarity_features import SimilarityFeatures
from neer_match.similarity_map import SimilarityMap


SimilarityMapDict = Dict[str, List[str]]

from contextlib import contextmanager
import joblib

@contextmanager
def _fit_warnings_silenced():
    """
    Silence what the elastic-net fits routinely emit: ConvergenceWarning under
    the loose cross-validation tolerance, and RuntimeWarning from the matrix
    products.

    These filters used to be set at import, which silenced them in the caller's
    own code as well. They now apply only around the fits below, inside the
    function that joblib runs, so that they hold in its worker processes too.
    """
    with warnings.catch_warnings(), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        yield

@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to integrate tqdm progress bars with joblib parallel execution.

    Parameters
    ----------
    tqdm_object : tqdm.tqdm or None
        Progress bar object to update during parallel execution.
    """
    if tqdm_object is None:
        yield
        return

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def _optimality_violation(
    grad_intercept: float, grad: np.ndarray, beta: np.ndarray, lam1: float, lam2: float
) -> float:
    """
    Largest violation of the optimality conditions of the elastic-net problem,
    0 at its solution: the gradient of the intercept is zero, a coefficient
    that is not zero balances its penalty, and one at zero has a gradient no
    larger than the L1 weight.
    """
    nonzero = beta != 0
    violation = np.where(
        nonzero,
        np.abs(grad + lam2 * beta + lam1 * np.sign(beta)),
        np.maximum(np.abs(grad) - lam1, 0.0),
    )
    return max(abs(grad_intercept), float(violation.max(initial=0.0)))


def _solve_l1_quadratic(
    Q: np.ndarray, b: np.ndarray, lam1: float, penalized: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """
    Minimise ``0.5 * z'Qz - b'z + lam1 * sum(|z_j| for penalised j)`` exactly,
    for positive semidefinite Q, starting from z: feature-sign search (Lee,
    Battle, Raina and Ng, "Efficient sparse coding algorithms", 2007).

    It keeps the coordinates that are not zero (and the unpenalised ones) with
    their signs, solves the quadratic on them in closed form, and moves to the
    best point on the way there, stopping where a coordinate reaches zero; a
    zero coordinate whose gradient exceeds lam1 enters one at a time. It ends
    where the optimality conditions hold to rounding error. With a few dozen
    coordinates each step is a small linear system, where coordinate descent
    needs many sweeps over strongly correlated features.
    """
    m = len(b)
    z = np.array(z, dtype=float)
    penalized = np.asarray(penalized, dtype=bool)
    tol = 1e-11 * (np.abs(b).max(initial=0.0) + lam1) + 1e-300

    def value(v: np.ndarray) -> float:
        return float(0.5 * v @ (Q @ v) - b @ v + lam1 * np.abs(v[penalized]).sum())

    def coordinate_sweep(v: np.ndarray) -> np.ndarray:
        """One sweep of exact coordinate minimisation: never raises the value."""
        v = v.copy()
        for j in range(m):
            if Q[j, j] <= 0.0:
                continue
            u = b[j] - Q[j] @ v + Q[j, j] * v[j]
            if penalized[j]:
                v[j] = np.sign(u) * max(abs(u) - lam1, 0.0) / Q[j, j]
            else:
                v[j] = u / Q[j, j]
        return v

    sign = np.where(penalized, np.sign(z), 0.0)
    current = value(z)
    for _ in range(20 * m + 20):
        grad = Q @ z - b
        free = ~penalized | (sign != 0)
        if np.abs(grad[free] + lam1 * sign[free]).max(initial=0.0) <= tol:
            # optimal on the free coordinates: should a zero one enter?
            excess = np.where(penalized & (sign == 0), np.abs(grad) - lam1, -np.inf)
            j = int(np.argmax(excess))
            if not excess[j] > tol:
                return z
            sign[j] = -np.sign(grad[j])
            free[j] = True

        # The minimiser of the quadratic on the free coordinates, at their signs
        idx = np.flatnonzero(free)
        Q_ff, rhs = Q[np.ix_(idx, idx)], b[idx] - lam1 * sign[idx]
        try:
            solution = np.linalg.solve(Q_ff, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(Q_ff, rhs, rcond=None)[0]
        target = np.zeros(m)
        target[idx] = solution

        # The best of the target and the points where a coordinate reaches zero
        crossings = {1.0: []}
        for k in idx:
            if penalized[k] and z[k] != 0.0 and np.sign(target[k]) != np.sign(z[k]):
                crossings.setdefault(z[k] / (z[k] - target[k]), []).append(k)
        best, best_value = None, current
        for t, zeroed in crossings.items():
            candidate = z + t * (target - z)
            candidate[zeroed] = 0.0
            v = value(candidate)
            if v < best_value:
                best, best_value = candidate, v
        if best is None:
            # rounding left no descent along the segment: a coordinate sweep
            # still finds one, unless z is already as good as it gets
            best = coordinate_sweep(z)
            best_value = value(best)
            if not best_value < current:
                return z
        z, current = best, best_value
        sign = np.where(penalized, np.sign(z), 0.0)
    return z


def fit_elastic_net_logistic(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    C: float,
    l1_ratio: float,
    *,
    tol: float = 1e-9,
    max_iter: int = 100,
    intercept_init: float | None = None,
    coef_init: np.ndarray | None = None,
) -> tuple[float, np.ndarray, int, bool]:
    """
    Elastic-net logistic regression by proximal Newton steps, as glmnet fits it.

    It minimises the objective of scikit-learn's
    ``LogisticRegression(l1_ratio=l1_ratio, C=C)`` with an unpenalised intercept,

        C * sum_i w_i * logloss_i + (1 - l1_ratio) / 2 * ||beta||^2 + l1_ratio * ||beta||_1,

    divided by ``C * sum(w)``, which leaves the solution unchanged. Each step
    solves the penalised quadratic approximation of the loss exactly
    (``_solve_l1_quadratic``) and is shortened until the objective falls. It is
    meant for many record pairs and few features: every step costs one pass
    over the pairs per feature pair, ``n * p**2``, and it converges in tens of
    steps, where saga needs thousands of passes to converge when the class
    weights are as uneven as matches make them. Started from the solution for
    a stronger penalty (``intercept_init``, ``coef_init``), as along a path of
    C values, it needs only a few.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Features, ideally standardised.
    y : np.ndarray of shape (n,)
        Labels, 0 or 1.
    sample_weight : np.ndarray of shape (n,)
        Positive weight of each pair, such as class weights.
    C, l1_ratio : float
        As in scikit-learn's LogisticRegression.
    tol : float, default=1e-9
        The fit stops once ``_optimality_violation``, measured on the objective
        divided by ``C * sum(w)``, is below it.
    max_iter : int, default=100
        Most Newton steps. The fit also stops, unconverged, once a step no
        longer lowers the objective measurably.
    intercept_init, coef_init : optional
        Where to start; by default from the intercept-only model. The solution
        does not depend on it.

    Returns
    -------
    intercept, coef, n_iter, converged
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    s = np.asarray(sample_weight, dtype=float)
    a = s / s.sum()
    lam1 = l1_ratio / (C * s.sum())
    lam2 = (1.0 - l1_ratio) / (C * s.sum())

    def objective(intercept: float, beta: np.ndarray) -> float:
        eta = intercept + X @ beta
        return float(
            a @ (np.logaddexp(0.0, eta) - y * eta)
            + 0.5 * lam2 * beta @ beta
            + lam1 * np.abs(beta).sum()
        )

    def penalty(beta: np.ndarray) -> float:
        return 0.5 * lam2 * float(beta @ beta) + lam1 * float(np.abs(beta).sum())

    # Start from the given point, or from the intercept-only model
    theta = np.zeros(p + 1)  # [intercept, beta]
    if coef_init is not None:
        theta[1:] = np.asarray(coef_init, dtype=float)
    if intercept_init is not None:
        theta[0] = float(intercept_init)
    else:
        share = float(a @ y)
        theta[0] = np.log(share / (1.0 - share))
    f = objective(theta[0], theta[1:])

    # The quadratic model's L1 term covers the coefficients, not the intercept
    penalized = np.ones(p + 1, dtype=bool)
    penalized[0] = False
    ridge = np.full(p + 1, lam2)
    ridge[0] = 0.0

    converged, n_iter, stalled = False, 0, 0
    while True:
        mu = expit(theta[0] + X @ theta[1:])
        residual = a * (mu - y)
        grad = np.concatenate([[residual.sum()], X.T @ residual])
        if _optimality_violation(grad[0], grad[1:], theta[1:], lam1, lam2) < tol:
            converged = True
            break
        if n_iter >= max_iter or stalled >= 3:
            break
        n_iter += 1

        # Hessian of the loss, [intercept, beta] by [intercept, beta]
        w = a * mu * (1.0 - mu)
        Xw = X * w[:, None]
        H = np.empty((p + 1, p + 1))
        H[0, 0] = w.sum()
        H[0, 1:] = H[1:, 0] = Xw.sum(axis=0)
        H[1:, 1:] = Xw.T @ X
        del Xw

        # The penalised quadratic model around theta, in the point z it moves to:
        # grad'(z - theta) + (z - theta)'H(z - theta)/2 + penalty(z), which is
        # z'(H + ridge)z/2 - (H theta - grad)'z + lam1 * |z| up to a constant
        new = _solve_l1_quadratic(
            H + np.diag(ridge), H @ theta - grad, lam1, penalized, theta
        )

        # Shorten the step until the objective falls enough
        direction = new - theta
        decrease = grad @ direction + penalty(new[1:]) - penalty(theta[1:])
        if not decrease < 0.0:
            break  # no descent left at this precision
        length = 1.0
        while True:
            candidate = theta + length * direction
            f_candidate = objective(candidate[0], candidate[1:])
            if f_candidate <= f + 1e-4 * length * decrease:
                break
            length *= 0.5
            if length < 1e-10:
                candidate = None
                break
        if candidate is None:
            break
        # Count the steps that no longer lower the objective beyond rounding
        stalled = stalled + 1 if f - f_candidate <= 1e-13 * abs(f) else 0
        theta, f = candidate, f_candidate

    if not converged:
        warnings.warn(
            f"The elastic-net fit stopped after {n_iter} Newton steps before "
            f"reaching tol={tol}.",
            ConvergenceWarning,
        )
    return float(theta[0]), theta[1:].copy(), n_iter, converged


def _as_logistic_regression(intercept, coef, classes, **params) -> LogisticRegression:
    """
    A fitted scikit-learn LogisticRegression holding these coefficients, so that
    scorers and later code can use it like one fitted by scikit-learn.
    """
    model = LogisticRegression(**params)
    model.classes_ = np.asarray(classes)
    model.coef_ = np.asarray(coef, dtype=float).reshape(1, -1)
    model.intercept_ = np.array([intercept], dtype=float)
    model.n_features_in_ = model.coef_.shape[1]
    return model


@dataclass
class FeatureSelectionResult:
    """
    Result object returned by FeatureSelector.execute().

    Attributes
    ----------
    updated_similarity_map : Dict[str, List[str]]
        Reduced similarity map containing only selected features.
        Keys are variable names, values are lists of similarity concept names.
    selected_feature_columns : List[str]
        Names of selected feature columns in the format 'col_{var}_{var}_{similarity}'.
    selected_pairs : List[Tuple[str, str]]
        List of (variable, similarity_concept) tuples that were selected.
    coef_by_feature : pd.Series
        Coefficients from the final elastic net model, sorted by absolute value.
        Useful for understanding feature importance.
    meta : Dict[str, Any]
        Metadata about the selection process, including:
        - method: Feature selection method used
        - scoring: Scoring metric used for cross-validation
        - cv: Number of cross-validation folds
        - n_features_in: Number of input features
        - n_features_selected: Number of features selected
        - C_grid: The C values tried, by l1_ratio
        - solver: The solver of the elastic-net fits
        - did_fallback: Whether fallback to original map occurred
    """
    updated_similarity_map: SimilarityMapDict
    selected_feature_columns: List[str]
    selected_pairs: List[Tuple[str, str]]
    coef_by_feature: pd.Series
    meta: Dict[str, Any]


class FeatureSelector:
    """
    Supervised feature selector for similarity-based entity matching.

    This class implements a two-stage feature selection process optimized for
    entity matching tasks with extreme class imbalance:

    **Stage 1: Correlation-based filtering** (optional)
        Removes redundant features by identifying groups of highly correlated
        features and keeping only the one most correlated with the target.

    **Stage 2: Elastic net regularization**
        Uses penalized logistic regression with L1/L2 penalties to identify
        features that contribute unique predictive information. Cross-validation
        is used to select optimal regularization parameters.

    The selector returns an updated similarity map containing only the features
    that passed both selection stages.

    Parameters
    ----------
    similarity_map : Dict[str, List[str]] or SimilarityMap
        Mapping from variable names to lists of similarity concepts.
        Example: {'name': ['jaro_winkler', 'levenshtein'], 'address': ['cosine']}
    training_data : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Three-tuple of (left_df, right_df, matches_df) used for feature selection.
    id_left_col, id_right_col : str, default='id'
        Column names containing entity IDs in left and right dataframes.
    matches_id_left, matches_id_right : str, default='left', 'right'
        Column names in matches_df identifying left/right entity IDs.
    match_col : str, default='match'
        Name of the binary match indicator column (1=match, 0=non-match).
    matches_are_indices : bool, default=True
        If True, treat match IDs as row positions (0..n-1), whatever the frames'
        index labels are. If False, treat as entity IDs.
    method : str, default='elastic_net'
        Feature selection method. Currently only 'elastic_net' is supported.
    scoring : str, default='average_precision'
        Scoring metric for cross-validation. Options: 'f1', 'roc_auc',
        'average_precision', 'neg_log_loss'. For imbalanced data, 'average_precision'
        is recommended.
    cv : int, default=5
        Number of cross-validation folds.
    l1_ratios : tuple, default=(0.8, 0.9, 1.0)
        L1 penalty ratios to try. 1.0 = pure Lasso, 0.0 = pure Ridge.
    Cs : int or Iterable[float], default=20
        Regularization strengths to try. If int, that many values per
        l1_ratio, logarithmically spaced from the C at which the penalty sets
        every coefficient to zero, computed from the training pairs, to 100,000
        times it (see ``_make_Cs_grid``); up to version 1.2.4, from 0.01 to
        1000 whatever the data. Pass ``np.logspace(-2, 3, 20)`` for that range.
        Higher C = less regularization. If the best C is the smallest or the
        largest tried, a warning says so: the best penalty may then lie outside
        the range.
    class_weight : str or None, default='balanced'
        Class weighting strategy. 'balanced' adjusts weights inversely proportional
        to class frequencies, recommended for imbalanced data.
    solver : {'newton-cd', 'saga'}, default='newton-cd'
        How the elastic-net fits are computed. Both minimise the objective of
        scikit-learn's LogisticRegression. 'newton-cd' takes proximal Newton
        steps (``fit_elastic_net_logistic``) and converges in about ten of
        them, a second or so on a hundred thousand pairs of tens of features.
        'saga', used up to version 1.2.3, is scikit-learn's stochastic solver:
        with the uneven class weights of matches it typically stops at
        ``max_iter`` far from the solution, after many minutes, so that its
        selection depends on where it stopped. Kept to reproduce earlier
        results.
    max_iter : int, default=5000
        Most iterations of each cross-validation fit: Newton steps for
        'newton-cd', which needs far fewer, epochs for 'saga'. The final fit
        of 'saga' takes up to 20,000.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available processors.
    min_coef_threshold : float, default=0.0
        Minimum absolute coefficient value for feature retention. Features with
        abs(coef) < threshold are dropped. Set to 0.0 to keep all non-zero features.
    max_correlation : float or None, default=None
        Correlation threshold for Stage 1 filtering. Features with pairwise
        correlation > threshold are candidates for removal. Example: 0.95.
        If None, Stage 1 is skipped.
    always_keep : Dict[str, List[str]] or None, default=None
        Features to always retain regardless of selection results.
        Example: {'surname': ['jaro_winkler']} always keeps surname jaro_winkler.
    preferred_separators : tuple, default=('__', '|', ':', '-', '_')
        Separators to try when parsing feature names (internal use).
    pairs_per_chunk : int, default=250_000
        Number of pairs whose similarity features are computed at a time. Only the
        feature columns are kept, so memory grows with 8 bytes per feature and
        pair rather than with the full pairwise frame. Does not affect the result.

    Attributes
    ----------
    similarity_map : Dict[str, List[str]]
        The input similarity map.
    left_train, right_train, matches_train : pd.DataFrame
        Training datasets for feature selection.

    Examples
    --------
    >>> from neer_match_utilities import FeatureSelector
    >>> selector = FeatureSelector(
    ...     similarity_map={'name': ['jaro_winkler', 'levenshtein', 'cosine'],
    ...                     'address': ['jaro_winkler', 'levenshtein']},
    ...     training_data=(left_df, right_df, matches_df),
    ...     max_correlation=0.95,
    ...     min_coef_threshold=0.01
    ... )
    >>> result = selector.execute()
    >>> print(result.updated_similarity_map)
    {'name': ['jaro_winkler'], 'address': ['levenshtein']}
    """

    # An int Cs spans this many powers of ten of C, up from the C at which the
    # penalty sets every coefficient to zero
    C_GRID_DECADES = 5

    def __init__(
        self,
        similarity_map: SimilarityMapDict,
        training_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        *,
        # these must match how you build df_train in your pipeline
        id_left_col: str = "id",
        id_right_col: str = "id",
        matches_id_left: str = "left",
        matches_id_right: str = "right",
        match_col: str = "match",
        matches_are_indices: bool = True,
        # selection configuration
        method: str = "elastic_net",  # currently only elastic_net
        scoring: str = "average_precision",  # "f1" | "roc_auc" | "average_precision" | "neg_log_loss"
        cv: int = 5,
        l1_ratios = (0.8, 0.9, 1.0),
        Cs: int | Iterable[float] = 20,
        class_weight: Optional[str] = "balanced",  # consider "balanced" if match class is rare
        solver: str = "newton-cd",  # or "saga", the solver up to version 1.2.3
        max_iter: int = 5000,
        random_state: int = 42,
        n_jobs: int = -1,
        # thresholding for feature selection
        min_coef_threshold: float = 0.0,  # drop features with abs(coef) < threshold
        # correlation-based pre-filtering
        max_correlation: Optional[float] = None,  # if set (e.g., 0.95), drop correlated features first
        # if you want to always keep certain similarities
        always_keep: Optional[Dict[str, List[str]]] = None,
        # if feature name parsing fails, you can tell the parser a separator preference
        preferred_separators: Tuple[str, ...] = ("__", "|", ":", "-", "_"),
        # pairs whose similarity features are computed at a time
        pairs_per_chunk: int = 250_000,
    ):
        self.similarity_map: SimilarityMapDict = dict(similarity_map)
        self.left_train, self.right_train, self.matches_train = training_data

        self.id_left_col = id_left_col
        self.id_right_col = id_right_col
        self.matches_id_left = matches_id_left
        self.matches_id_right = matches_id_right
        self.match_col = match_col
        self.matches_are_indices = matches_are_indices

        self.method = method
        self.scoring = scoring
        self.cv = cv
        self.l1_ratios = list(l1_ratios)
        self.Cs = Cs

        self.class_weight = class_weight
        if solver not in ("newton-cd", "saga"):
            raise ValueError(f"solver must be 'newton-cd' or 'saga', not {solver!r}.")
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.min_coef_threshold = min_coef_threshold
        self.max_correlation = max_correlation
        self.always_keep = always_keep or {}
        self.preferred_separators = preferred_separators
        self.pairs_per_chunk = pairs_per_chunk

    def _make_Cs_grid(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Regularization strengths to try in cross-validation, by l1_ratio.

        With ``Cs`` an int, a path of that many C values per l1_ratio, as glmnet
        builds one: logarithmically spaced from the C at which the L1 penalty
        sets every coefficient to zero, computed from the data, to
        ``10**C_GRID_DECADES`` times it. How strongly a given C penalises
        depends on the number and weights of the pairs, as scikit-learn's
        objective multiplies C with the sum of their losses, so that a fixed
        range of C suits one data size only. Otherwise the C values given, for
        every l1_ratio.

        Parameters
        ----------
        X : np.ndarray
            The standardised features of all training pairs.
        y : np.ndarray
            Their labels.

        Returns
        -------
        dict
            The C values by l1_ratio, ascending. Higher C means less
            regularization, so that more features are kept.
        """
        if not isinstance(self.Cs, int):
            given = np.sort(np.asarray(list(self.Cs), dtype=float))
            return {l1: given for l1 in self.l1_ratios}

        # The gradient of the weighted log loss at the intercept-only model:
        # every coefficient stays zero as long as the L1 weight
        # l1_ratio / (C * sum(w)) is at least its largest entry
        weights = compute_sample_weight(self.class_weight, y)
        a = weights / weights.sum()
        largest = float(np.abs(X.T @ (a * (float(a @ y) - y))).max(initial=0.0))
        if not largest > 0.0:
            # no feature varies with the labels: the range of version 1.2.4
            return {l1: np.logspace(-2, 3, self.Cs) for l1 in self.l1_ratios}

        grids = {}
        for l1 in self.l1_ratios:
            # without an L1 part nothing is ever zero; take a small one, as glmnet does
            C_zero = max(l1, 1e-3) / (weights.sum() * largest)
            grids[l1] = C_zero * np.logspace(0, self.C_GRID_DECADES, self.Cs)
        return grids

    def _drop_correlated_features(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """
        Remove redundant features based on pairwise correlations (Stage 1).

        For each group of features with pairwise correlation exceeding
        `self.max_correlation`, this method keeps only the feature most
        correlated with the target variable and drops the rest.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (rows=samples, columns=features).
        y : np.ndarray
            Target variable (binary: 0=non-match, 1=match).

        Returns
        -------
        pd.DataFrame
            Feature matrix with correlated features removed.

        Notes
        -----
        If `self.max_correlation` is None, no filtering is performed and X
        is returned unchanged.
        """
        if self.max_correlation is None:
            return X

        print(f"[FeatureSelector] Checking for correlations > {self.max_correlation}")

        # Compute pairwise feature correlations
        corr_matrix = X.corr().abs()

        # Compute correlation of each feature with target variable
        target_corr = X.corrwith(pd.Series(y, index=X.index)).abs()

        # Extract upper triangle to avoid duplicate pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = set()
        dropped_details = []

        for col1 in upper_tri.columns:
            if col1 in to_drop:
                continue

            # Identify features highly correlated with col1
            correlated_features = upper_tri.index[upper_tri[col1] > self.max_correlation].tolist()

            if correlated_features:
                # From the correlated group, select the feature most predictive of target
                candidates = [col1] + [f for f in correlated_features if f not in to_drop]

                if len(candidates) > 1:
                    # Keep feature with highest target correlation
                    best_feature = max(candidates, key=lambda f: target_corr.get(f, 0.0))

                    # Mark others for removal
                    for candidate in candidates:
                        if candidate != best_feature:
                            to_drop.add(candidate)
                            dropped_details.append(
                                f"{candidate} (target_corr={target_corr.get(candidate, 0.0):.3f}) "
                                f"-> kept {best_feature} (target_corr={target_corr.get(best_feature, 0.0):.3f})"
                            )

        if to_drop:
            print(f"[FeatureSelector] Stage 1 - Dropping {len(to_drop)} correlated features: {sorted(to_drop)}")
            print(f"[FeatureSelector] Stage 1 - Details (kept most predictive):")
            for detail in dropped_details[:5]:
                print(f"  - {detail}")
            if len(dropped_details) > 5:
                print(f"  ... and {len(dropped_details) - 5} more")
            X = X.drop(columns=list(to_drop))
        else:
            print(f"[FeatureSelector] Stage 1 - No highly correlated features found")

        return X

    def _print_correlation_summary(self, X: pd.DataFrame, top_n: int = 10):
        """
        Print diagnostic information about feature correlations.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        top_n : int, default=10
            Number of top correlations to display.
        """
        corr_matrix = X.corr().abs()

        # Extract upper triangle to avoid duplicate pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Collect all pairwise correlations
        correlations = []
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                val = upper_tri.loc[idx, col]
                if pd.notna(val):
                    correlations.append((val, idx, col))

        if correlations:
            correlations.sort(reverse=True)
            print(f"\n[FeatureSelector] Top {top_n} feature correlations:")
            for i, (corr_val, feat1, feat2) in enumerate(correlations[:top_n], 1):
                print(f"  {i}. {feat1} <-> {feat2}: {corr_val:.3f}")
            print()

    def _selected_cols_to_similarity_map(self, selected_cols: list[str]) -> dict[str, list[str]]:
        """
        Convert selected feature column names back to similarity map format.

        Parameters
        ----------
        selected_cols : list[str]
            List of selected feature column names (format: 'col_{var}_{var}_{similarity}').

        Returns
        -------
        dict[str, list[str]]
            Updated similarity map containing only selected features.
            Variables with no selected features are excluded.
        """
        selected_set = set(selected_cols)
        selected: dict[str, set[str]] = {}

        # Parse feature names to extract (variable, similarity) pairs
        for lcol, rcol, sim in SimilarityMap(self.similarity_map):
            col_name = f"col_{lcol}_{rcol}_{sim}"
            if col_name in selected_set:
                selected.setdefault(lcol, set()).add(sim)

        # Merge in features marked as always_keep
        for var, sims in (self.always_keep or {}).items():
            selected.setdefault(var, set()).update(sims)

        # Preserve original ordering of similarities within each variable
        updated: dict[str, list[str]] = {}
        for var, sims in self.similarity_map.items():
            keep = [s for s in sims if s in selected.get(var, set())]
            if keep:
                updated[var] = keep

        return updated

    def execute(self) -> FeatureSelectionResult:
        """
        Execute the two-stage feature selection process.

        This method performs:
        1. Builds pairwise similarity features from training data
        2. Cleans data (removes constant columns, handles missing values)
        3. **Stage 1** (optional): Correlation-based filtering
        4. Scales features for regularization
        5. **Stage 2**: Elastic net cross-validation and feature selection
        6. Applies coefficient thresholding (if configured)
        7. Converts selected features back to similarity map format

        Returns
        -------
        FeatureSelectionResult
            Object containing the updated similarity map, selected features,
            coefficients, and metadata about the selection process.

        Raises
        ------
        ValueError
            If the method is not 'elastic_net', or if no usable features remain
            after cleaning or correlation filtering, or if there are too few
            positive examples for cross-validation.

        Notes
        -----
        The method prints detailed diagnostic information during execution:
        - Top feature correlations
        - Features dropped in Stage 1 (correlation filtering)
        - Cross-validation progress and best parameters
        - Features dropped in Stage 2 (elastic net)
        - Top coefficients by absolute value
        - Final summary statistics

        Examples
        --------
        >>> result = selector.execute()
        >>> print(f"Selected {len(result.selected_feature_columns)} features")
        >>> print(f"Updated similarity map: {result.updated_similarity_map}")
        """
        if self.method != "elastic_net":
            raise ValueError(f"Unsupported method={self.method}. Only 'elastic_net' is implemented.")

        # Step 1: Build similarity features from training data, block by block.
        # Only the feature columns and the match labels are kept, so the full
        # pairwise frame, with its two ID columns, is never held in memory.
        smap_obj = SimilarityMap(self.similarity_map)
        feats = SimilarityFeatures(similarity_map=smap_obj)

        pairs = feats.pairs(
            left=self.left_train,
            right=self.right_train,
            matches=self.matches_train,
            left_id_col=self.id_left_col,
            right_id_col=self.id_right_col,
            match_col=self.match_col,
            matches_id_left=self.matches_id_left,
            matches_id_right=self.matches_id_right,
            matches_are_indices=self.matches_are_indices,
        )

        y_blocks, X_blocks = [], []
        for df_train in pairs.chunks(self.pairs_per_chunk):
            if self.match_col not in df_train.columns:
                raise ValueError(
                    f"Expected match label column '{self.match_col}' in df_train, "
                    f"but columns are: {list(df_train.columns)[:30]}..."
                )

            # Step 2: Extract target variable and feature matrix
            y_blocks.append(df_train[self.match_col].astype(int).to_numpy())

            # Filter to similarity feature columns only
            X_blocks.append(df_train[[c for c in df_train.columns if c.startswith("col_") and c != self.match_col]])

        y = np.concatenate(y_blocks)
        X = pd.concat(X_blocks)
        del df_train, X_blocks

        # Verify sufficient positive examples for cross-validation
        pos = int(y.sum())
        if pos < self.cv:
            raise ValueError(f"Not enough positives for cv={self.cv}: pos={pos}. Reduce cv or use more labeled matches.")

        # Ensure all features are numeric
        X = X.select_dtypes(include=[np.number]).copy()

        # Handle infinity and missing values
        # In place: the same values, without two more full copies of X
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0.0, inplace=True)

        # Remove constant columns (no information)
        nunique = X.nunique(dropna=False)
        if not (nunique > 1).all():
            X = X.loc[:, nunique > 1]

        if X.shape[1] == 0:
            raise ValueError("After cleaning, no usable similarity feature columns remain.")

        # Diagnostic: print top feature correlations
        self._print_correlation_summary(X)

        # Step 3 (Stage 1): Correlation-based pre-filtering (optional)
        X = self._drop_correlated_features(X, y)

        if X.shape[1] == 0:
            raise ValueError("After correlation filtering, no usable similarity feature columns remain.")

        # Step 4: Standardize features (required for elastic net)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Only the column names are needed from here on, so free the unscaled
        # copy before the fits rather than holding both through them
        feature_columns = X.columns
        del X

        # Step 5 (Stage 2): Elastic net cross-validation
        # Build parameter grid for exhaustive search
        Cs_grid = self._make_Cs_grid(X_scaled, y)
        l1_grid = list(self.l1_ratios)
        for l1, grid in Cs_grid.items():
            print(f"[FeatureSelector] l1_ratio={l1}: {len(grid)} C values from {grid[0]:.3g} to {grid[-1]:.3g}")

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scorer = get_scorer(self.scoring)

        # Pre-compute CV folds for reproducibility. Each pair's fold number takes
        # one byte, where the folds' index arrays take eight per pair and fold,
        # and it is what the worker processes receive.
        fold_of = np.empty(len(y), dtype=np.min_scalar_type(self.cv))
        for k, (_, val_idx) in enumerate(skf.split(X_scaled, y)):
            fold_of[val_idx] = k

        classes = np.unique(y)

        def saga_fit(X_fit, y_fit, l1_ratio: float, C: float, final: bool) -> LogisticRegression:
            """One fit by scikit-learn's saga, the solver up to version 1.2.3."""
            # l1_ratio alone selects the elastic-net penalty; passing
            # penalty= is deprecated in scikit-learn 1.8 and removed in 1.10
            clf = LogisticRegression(
                solver="saga",
                l1_ratio=l1_ratio,
                C=C,
                max_iter=20000 if final else self.max_iter,
                tol=1e-3 if final else 1e-2,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
            return clf.fit(X_fit, y_fit)

        def newton_path(X_fit, y_fit, l1_ratio: float, Cs) -> dict:
            """
            Fits at the given C values, as fitted LogisticRegressions by C: the
            strongest penalty first, each fit started from the one before, as
            glmnet computes a path. The solutions do not depend on the order.
            """
            # The weights LogisticRegression derives from class_weight
            weights = compute_sample_weight(self.class_weight, y_fit)
            models, intercept, coef = {}, None, None
            for C in sorted(Cs):
                intercept, coef, _, _ = fit_elastic_net_logistic(
                    X_fit, y_fit, weights, C, l1_ratio, max_iter=self.max_iter,
                    intercept_init=intercept, coef_init=coef,
                )
                models[C] = _as_logistic_regression(
                    intercept, coef, classes, l1_ratio=l1_ratio, C=C, class_weight=self.class_weight
                )
            return models

        def fold(fold_idx: int):
            """Training and validation part of one CV fold."""
            # The indices skf.split yields: both sorted, train the complement
            in_fold = fold_of == fold_idx
            train_idx, val_idx = np.flatnonzero(~in_fold), np.flatnonzero(in_fold)
            return X_scaled[train_idx], X_scaled[val_idx], y[train_idx], y[val_idx]

        def fit_and_score(l1_ratio: float, C: float, fold_idx: int) -> list:
            """saga: train and evaluate one configuration on one CV fold."""
            X_tr, X_va, y_tr, y_va = fold(fold_idx)
            with _fit_warnings_silenced():
                clf = saga_fit(X_tr, y_tr, l1_ratio, C, final=False)
                score = scorer(clf, X_va, y_va)
            return [(l1_ratio, C, fold_idx, float(score))]

        def path_and_score(l1_ratio: float, fold_idx: int) -> list:
            """newton-cd: train and evaluate all C values of one l1_ratio on one CV fold."""
            X_tr, X_va, y_tr, y_va = fold(fold_idx)
            with _fit_warnings_silenced():
                models = newton_path(X_tr, y_tr, l1_ratio, Cs_grid[l1_ratio])
                return [
                    (l1_ratio, C, fold_idx, float(scorer(clf, X_va, y_va)))
                    for C, clf in models.items()
                ]

        # One task per fit for saga; for newton-cd one per path of C values
        if self.solver == "saga":
            tasks = [
                delayed(fit_and_score)(l1, C, k)
                for l1 in l1_grid for C in Cs_grid[l1] for k in range(self.cv)
            ]
            unit = "fit"
        else:
            tasks = [delayed(path_and_score)(l1, k) for l1 in l1_grid for k in range(self.cv)]
            unit = "path"

        # Execute parallel CV with progress bar
        pbar = tqdm(total=len(tasks), desc=f"[FeatureSelector] CV {unit}s", unit=unit) if tqdm is not None else None

        with tqdm_joblib(pbar):
            results = [
                result
                for task_results in Parallel(n_jobs=self.n_jobs, prefer="processes")(tasks)
                for result in task_results
            ]

        # Aggregate cross-validation scores
        scores = {}
        counts = {}
        for l1, C, _, sc in results:
            key = (l1, C)
            scores[key] = scores.get(key, 0.0) + sc
            counts[key] = counts.get(key, 0) + 1

        mean_scores = {k: scores[k] / counts[k] for k in scores}
        best_params = max(mean_scores.items(), key=lambda kv: kv[1])[0]
        best_l1, best_C = best_params
        best_score = mean_scores[best_params]

        print(f"[FeatureSelector] Stage 2 - Best params: l1_ratio={best_l1}, C={best_C}, score={best_score:.6f}")
        if len(Cs_grid[best_l1]) > 1 and best_C in (Cs_grid[best_l1][0], Cs_grid[best_l1][-1]):
            end = "smallest" if best_C == Cs_grid[best_l1][0] else "largest"
            message = (
                f"[FeatureSelector] The best C, {best_C:.3g}, is the {end} tried for "
                f"l1_ratio={best_l1}: the best penalty may lie outside the range of Cs."
            )
            print(message)
            warnings.warn(message, UserWarning, stacklevel=2)

        # Step 6: Refit final model on full training set with best parameters;
        # newton-cd along the path of C values up to the best one
        with _fit_warnings_silenced():
            if self.solver == "saga":
                final_model = saga_fit(X_scaled, y, best_l1, best_C, final=True)
            else:
                path = [C for C in Cs_grid[best_l1] if C <= best_C]
                final_model = newton_path(X_scaled, y, best_l1, path)[best_C]

        coef = pd.Series(final_model.coef_.ravel(), index=feature_columns)

        # Step 7: Apply coefficient threshold to select final features
        if self.min_coef_threshold > 0:
            mask = coef.abs() >= self.min_coef_threshold
            selected_feature_columns = coef.index[mask].tolist()
            dropped_by_threshold = coef.index[~mask].tolist()
            print(f"[FeatureSelector] Stage 2 - Applied min_coef_threshold={self.min_coef_threshold}")
            if dropped_by_threshold:
                print(f"[FeatureSelector] Stage 2 - Dropping {len(dropped_by_threshold)} features with weak coefficients: {dropped_by_threshold}")
        else:
            selected_feature_columns = coef.index[coef != 0].tolist()
            dropped_by_elasticnet = coef.index[coef == 0].tolist()
            if dropped_by_elasticnet:
                print(f"[FeatureSelector] Stage 2 - Elastic net zeroed {len(dropped_by_elasticnet)} features: {dropped_by_elasticnet}")

        print(f"\n[FeatureSelector] Summary: features_in={len(feature_columns)} selected_cols={len(selected_feature_columns)}")

        # Diagnostic: print top coefficients
        top_coefs = coef.abs().sort_values(ascending=False).head(15)
        print(f"\n[FeatureSelector] Top 15 coefficients by absolute value:")
        for feat in top_coefs.index:
            marker = "✓" if feat in selected_feature_columns else "✗"
            print(f"  {marker} {feat}: {coef[feat]:+.4f}")
        print()

        print(f"[FeatureSelector] pos={int(y.sum())} neg={int((1-y).sum())} "
            f"features_in={len(feature_columns)} selected_cols={len(selected_feature_columns)}")

        # Step 8: Build metadata dictionary
        meta = {
            "method": self.method,
            "scoring": self.scoring,
            "cv": self.cv,
            "l1_ratios": self.l1_ratios,
            "Cs": self.Cs,
            "C_grid": {l1: grid.tolist() for l1, grid in Cs_grid.items()},
            "class_weight": self.class_weight,
            "solver": self.solver,
            "n_features_in": int(len(feature_columns)),
            "n_features_selected": int(len(selected_feature_columns)),
        }

        meta["did_fallback"] = False

        # Step 9: Convert selected features back to similarity map format
        if len(selected_feature_columns) == 0:
            # Fallback: no features selected, keep original map
            updated = dict(self.similarity_map)
            meta["did_fallback"] = True
        else:
            updated = self._selected_cols_to_similarity_map(selected_feature_columns)
            if not updated:
                # Fallback: conversion produced empty map
                updated = dict(self.similarity_map)
                meta["did_fallback"] = True

        selected_pairs = [(var, sim) for var, sims in updated.items() for sim in sims]

        print(f"[FeatureSelector] did_fallback={meta['did_fallback']}")

        return FeatureSelectionResult(
            updated_similarity_map=updated,
            selected_feature_columns=selected_feature_columns,
            selected_pairs=selected_pairs,
            coef_by_feature=coef.sort_values(key=lambda s: s.abs(), ascending=False),
            meta=meta,
        )

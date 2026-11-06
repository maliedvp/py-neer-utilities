import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from neer_match_utilities.feature_selection import (
    FeatureSelector,
    _optimality_violation,
    _solve_l1_quadratic,
    fit_elastic_net_logistic,
)


def problem(n=400, p=5, seed=0):
    """Imbalanced labels, two informative features and three noise features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (rng.random(n) < expit(-2.5 + 1.5 * X[:, 0] - 1.0 * X[:, 1])).astype(int)
    return X, y, compute_sample_weight("balanced", y)


def sklearn_fit(X, y, C, l1_ratio, solver, tol, max_iter):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LogisticRegression(
            solver=solver, C=C, l1_ratio=l1_ratio, tol=tol, max_iter=max_iter,
            class_weight="balanced",
        ).fit(X, y)
    return clf.intercept_[0], clf.coef_.ravel()


@pytest.mark.parametrize("C, l1_ratio", [(0.05, 0.5), (1.0, 0.8), (0.02, 1.0)])
def test_matches_saga_where_saga_converges(C, l1_ratio):
    X, y, w = problem()
    intercept, coef, _, converged = fit_elastic_net_logistic(X, y, w, C, l1_ratio)
    saga_intercept, saga_coef = sklearn_fit(X, y, C, l1_ratio, "saga", 1e-10, 200_000)

    assert converged
    np.testing.assert_allclose(coef, saga_coef, atol=1e-4)
    assert intercept == pytest.approx(saga_intercept, abs=1e-4)
    # the L1 part sets the same coefficients exactly to zero
    assert ((coef == 0) == (np.abs(saga_coef) < 1e-8)).all()


def test_matches_lbfgs_without_l1_penalty():
    X, y, w = problem()
    intercept, coef, _, _ = fit_elastic_net_logistic(X, y, w, 0.5, 0.0)
    expected_intercept, expected_coef = sklearn_fit(X, y, 0.5, 0.0, "lbfgs", 1e-12, 10_000)

    np.testing.assert_allclose(coef, expected_coef, atol=1e-6)
    assert intercept == pytest.approx(expected_intercept, abs=1e-6)


def test_strong_penalty_zeroes_the_noise_features():
    X, y, w = problem(n=2000)
    _, coef, _, _ = fit_elastic_net_logistic(X, y, w, 0.01, 1.0)

    assert coef[0] > 0 > coef[1]
    assert (coef[2:] == 0).all()


def test_returns_the_optimum():
    X, y, w = problem()
    intercept, coef, n_iter, converged = fit_elastic_net_logistic(X, y, w, 0.3, 0.9)

    a = w / w.sum()
    lam1, lam2 = 0.9 / (0.3 * w.sum()), 0.1 / (0.3 * w.sum())
    residual = a * (expit(intercept + X @ coef) - y)
    assert converged and n_iter < 30
    assert _optimality_violation(residual.sum(), X.T @ residual, coef, lam1, lam2) < 1e-9


def test_a_warm_start_reaches_the_same_solution():
    X, y, w = problem()
    weak_intercept, weak_coef, _, _ = fit_elastic_net_logistic(X, y, w, 0.05, 0.9)

    cold = fit_elastic_net_logistic(X, y, w, 2.0, 0.9)
    warm = fit_elastic_net_logistic(X, y, w, 2.0, 0.9, intercept_init=weak_intercept, coef_init=weak_coef)

    np.testing.assert_allclose(warm[1], cold[1], atol=1e-6)
    assert warm[0] == pytest.approx(cold[0], abs=1e-6)
    assert warm[2] <= cold[2]


@pytest.mark.parametrize("seed", range(40))
def test_the_l1_quadratic_is_solved_exactly(seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(2, 25))
    A = rng.normal(size=(3 * m, m))
    A[:, 1:] += 2 * rng.random() * A[:, :1]  # correlated columns, as similarities are
    Q = A.T @ A / (3 * m) + (0.0 if seed % 3 == 0 else 0.01) * np.eye(m)
    b, lam1 = rng.normal(size=m), float(rng.uniform(0.01, 1.0))
    penalized = np.ones(m, dtype=bool)
    penalized[0] = seed % 2 == 0  # like the unpenalised intercept
    start = rng.normal(size=m) if seed % 4 == 0 else np.zeros(m)

    z = _solve_l1_quadratic(Q, b, lam1, penalized, start)

    grad = Q @ z - b
    violation = np.where(
        ~penalized, np.abs(grad),
        np.where(z != 0, np.abs(grad + lam1 * np.sign(z)), np.maximum(np.abs(grad) - lam1, 0.0)),
    )
    assert violation.max() < 1e-10 * (np.abs(b).max() + lam1)


def test_warns_when_stopped_early():
    X, y, w = problem()
    with pytest.warns(ConvergenceWarning):
        *_, converged = fit_elastic_net_logistic(X, y, w, 1.0, 0.8, max_iter=1)
    assert not converged


def make_selector(**kwargs):
    rng = np.random.default_rng(0)
    names = ["".join(rng.choice(list("abcdefgh"), 6)) for _ in range(30)]
    left = pd.DataFrame({"id": range(30), "name": names, "city": rng.choice(["x", "y"], 30)})
    right = pd.DataFrame({"id": range(30), "name": [n[:-1] + "q" for n in names], "city": left["city"]})
    matches = pd.DataFrame({"left": range(30), "right": range(30)})
    return FeatureSelector(
        {"name": ["levenshtein", "jaro_winkler"], "city": ["levenshtein"]},
        (left, right, matches),
        cv=2, Cs=3, l1_ratios=(0.9,), n_jobs=1,
        **kwargs,
    )


@pytest.mark.parametrize("solver", ["newton-cd", "saga"])
def test_feature_selector_runs_with_either_solver(solver):
    result = make_selector(solver=solver).execute()

    assert result.meta["solver"] == solver
    assert result.selected_feature_columns


def test_newton_cd_is_the_default():
    assert make_selector().solver == "newton-cd"


def test_unknown_solver_is_rejected():
    with pytest.raises(ValueError, match="solver"):
        make_selector(solver="lbfgs")

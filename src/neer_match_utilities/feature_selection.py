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
from joblib import Parallel, delayed

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress sklearn warnings about deprecated parameters and convergence issues
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'penalty' was deprecated.*"
)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
    matches_are_indices : bool, default=False
        If True, treat match IDs as integer row indices. If False, treat as entity IDs.
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
        Regularization strengths to try. If int, generates `Cs` values
        logarithmically spaced from 0.01 to 1000. Higher C = less regularization.
    class_weight : str or None, default='balanced'
        Class weighting strategy. 'balanced' adjusts weights inversely proportional
        to class frequencies, recommended for imbalanced data.
    max_iter : int, default=5000
        Maximum iterations for elastic net solver.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available processors.
    min_coef_threshold : float, default=0.0
        Minimum absolute coefficient value for feature retention. Features with
        |coef| < threshold are dropped. Set to 0.0 to keep all non-zero features.
    max_correlation : float or None, default=None
        Correlation threshold for Stage 1 filtering. Features with pairwise
        correlation > threshold are candidates for removal. Example: 0.95.
        If None, Stage 1 is skipped.
    always_keep : Dict[str, List[str]] or None, default=None
        Features to always retain regardless of selection results.
        Example: {'surname': ['jaro_winkler']} always keeps surname jaro_winkler.
    preferred_separators : tuple, default=('__', '|', ':', '-', '_')
        Separators to try when parsing feature names (internal use).

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
        matches_are_indices: bool = False,
        # selection configuration
        method: str = "elastic_net",  # currently only elastic_net
        scoring: str = "average_precision",  # "f1" | "roc_auc" | "average_precision" | "neg_log_loss"
        cv: int = 5,
        l1_ratios = (0.8, 0.9, 1.0),
        Cs: int | Iterable[float] = 20,
        class_weight: Optional[str] = "balanced",  # consider "balanced" if match class is rare
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
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.min_coef_threshold = min_coef_threshold
        self.max_correlation = max_correlation
        self.always_keep = always_keep or {}
        self.preferred_separators = preferred_separators

    def _make_Cs_grid(self) -> np.ndarray:
        """
        Generate grid of regularization parameter values for cross-validation.

        Returns
        -------
        np.ndarray
            Array of C values to try during cross-validation.
            If Cs was specified as an int, returns logarithmically spaced values
            from 0.01 to 1000. Otherwise returns the provided values.

        Notes
        -----
        Higher C values mean less regularization (more features kept).
        Lower C values mean stronger regularization (more aggressive dropping).
        """
        if isinstance(self.Cs, int):
            return np.logspace(-2, 3, self.Cs)
        return np.asarray(list(self.Cs), dtype=float)

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

        # Step 1: Build similarity feature dataframe from training data
        smap_obj = SimilarityMap(self.similarity_map)
        feats = SimilarityFeatures(similarity_map=smap_obj)

        df_train = feats.pairwise_similarity_dataframe(
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

        if self.match_col not in df_train.columns:
            raise ValueError(
                f"Expected match label column '{self.match_col}' in df_train, "
                f"but columns are: {list(df_train.columns)[:30]}..."
            )

        # Step 2: Extract target variable and feature matrix
        y = df_train[self.match_col].astype(int).to_numpy()

        # Verify sufficient positive examples for cross-validation
        pos = int(y.sum())
        if pos < self.cv:
            raise ValueError(f"Not enough positives for cv={self.cv}: pos={pos}. Reduce cv or use more labeled matches.")

        X = df_train.drop(columns=[self.match_col])

        # Filter to similarity feature columns only
        X = X[[c for c in X.columns if c.startswith("col_")]]

        # Ensure all features are numeric
        X = X.select_dtypes(include=[np.number]).copy()

        # Handle infinity and missing values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Remove constant columns (no information)
        nunique = X.nunique(dropna=False)
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

        # Step 5 (Stage 2): Elastic net cross-validation
        # Build parameter grid for exhaustive search
        Cs_grid = self._make_Cs_grid()
        l1_grid = list(self.l1_ratios)

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scorer = get_scorer(self.scoring)

        # Pre-compute CV folds for reproducibility
        folds = list(skf.split(X_scaled, y))

        def fit_and_score(l1_ratio: float, C: float, fold_idx: int):
            """Train and evaluate one configuration on one CV fold."""
            train_idx, val_idx = folds[fold_idx]
            X_tr, X_va = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            clf = LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                l1_ratio=l1_ratio,
                C=C,
                max_iter=self.max_iter,
                tol=1e-2,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
            clf.fit(X_tr, y_tr)
            score = scorer(clf, X_va, y_va)
            return (l1_ratio, C, fold_idx, float(score))

        # Generate all parameter × fold combinations
        tasks = [(l1, C, k) for l1 in l1_grid for C in Cs_grid for k in range(len(folds))]
        total_fits = len(tasks)

        # Execute parallel CV with progress bar
        pbar = tqdm(total=total_fits, desc="[FeatureSelector] CV fits", unit="fit") if tqdm is not None else None

        with tqdm_joblib(pbar):
            results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(fit_and_score)(l1, C, k) for (l1, C, k) in tasks
            )

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

        # Step 6: Refit final model on full training set with best parameters
        final_model = LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=best_l1,
            C=best_C,
            max_iter=20000,
            tol=1e-3,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        final_model.fit(X_scaled, y)

        coef = pd.Series(final_model.coef_.ravel(), index=X.columns)

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

        print(f"\n[FeatureSelector] Summary: features_in={X.shape[1]} selected_cols={len(selected_feature_columns)}")

        # Diagnostic: print top coefficients
        top_coefs = coef.abs().sort_values(ascending=False).head(15)
        print(f"\n[FeatureSelector] Top 15 coefficients by absolute value:")
        for feat in top_coefs.index:
            marker = "✓" if feat in selected_feature_columns else "✗"
            print(f"  {marker} {feat}: {coef[feat]:+.4f}")
        print()

        print(f"[FeatureSelector] pos={int(y.sum())} neg={int((1-y).sum())} "
            f"features_in={X.shape[1]} selected_cols={len(selected_feature_columns)}")

        # Step 8: Build metadata dictionary
        meta = {
            "method": self.method,
            "scoring": self.scoring,
            "cv": self.cv,
            "l1_ratios": self.l1_ratios,
            "Cs": self.Cs,
            "class_weight": self.class_weight,
            "n_features_in": int(X.shape[1]),
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

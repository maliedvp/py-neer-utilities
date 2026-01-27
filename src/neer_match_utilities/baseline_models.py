from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from neer_match import similarity_map as _sim
from neer_match.similarity_map import SimilarityMap
from neer_match_utilities.custom_similarities import CustomSimilarities
CustomSimilarities()  # monkey-patch once, globally
from neer_match_utilities.similarity_features import SimilarityFeatures
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# Surpress certain warnings globally in this module
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.seterr(over='ignore', divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning)

class SuggestMixin:
    """
    Adds a NeerMatch-like .suggest(left, right, count, verbose) API to baseline models.

    Requires:

    - self.predict_proba(df_pairs) implemented
    - self.similarity_map set to a SimilarityMap (or dict) describing features to compute
    """

    similarity_map: SimilarityMap | dict | None = None  # dynamically attached by loader

    def suggest(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        *,
        count: int = 10,
        verbose: int = 0,
        left_id_col: str | None = None,
        right_id_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Return top-k candidate matches per left record (like neer_match DL models).

        Output columns:
          - left: integer row index into `left` (0..len(left)-1)
          - right: integer row index into `right` (0..len(right)-1)
          - prediction: match probability
        """
        if self.similarity_map is None:
            raise ValueError(
                "Baseline model has no similarity_map attached. "
                "Load it via ModelBaseline.load(...) or pass similarity_map when saving."
            )

        # normalize similarity map to SimilarityMap object
        smap = self.similarity_map
        if isinstance(smap, dict):
            smap = SimilarityMap(smap)
        elif not isinstance(smap, SimilarityMap):
            raise TypeError("similarity_map must be a dict or SimilarityMap")

        # We want neer_match-like indices: 0..n-1
        left_tmp = left.reset_index(drop=True).copy()
        right_tmp = right.reset_index(drop=True).copy()

        # pick ID cols (internal)
        # If user provides real IDs, keep them; otherwise use row index ids.
        if left_id_col is None:
            left_tmp["_row_id"] = np.arange(len(left_tmp), dtype=int)
            left_id_col = "_row_id"
        if right_id_col is None:
            right_tmp["_row_id"] = np.arange(len(right_tmp), dtype=int)
            right_id_col = "_row_id"

        feats = SimilarityFeatures(similarity_map=smap)

        empty_matches = pd.DataFrame({"left": [], "right": []})

        df_pairs = feats.pairwise_similarity_dataframe(
            left=left_tmp,
            right=right_tmp,
            matches=empty_matches,
            left_id_col=left_id_col,
            right_id_col=right_id_col,
            match_col="match",
            matches_id_left="left",
            matches_id_right="right",
            matches_are_indices=False,
        )

        proba = self.predict_proba(df_pairs)
        df_pairs["prediction"] = proba

        # Identify the output ID columns produced by SimilarityFeatures
        # It may suffix _left/_right if names collide.
        if left_id_col == right_id_col:
            out_left = f"{left_id_col}_left"
            out_right = f"{right_id_col}_right"
        else:
            out_left = left_id_col
            out_right = right_id_col

        out = df_pairs[[out_left, out_right, "prediction"]].rename(
            columns={out_left: "left", out_right: "right"}
        )

        # Top-k per left
        out = out.sort_values(["left", "prediction"], ascending=[True, False])
        out = out.groupby("left", as_index=False).head(count).reset_index(drop=True)

        if verbose:
            print(f"[baseline.suggest] left={len(left_tmp)} right={len(right_tmp)} pairs={len(df_pairs)}")

        return out

@dataclass
class LogitMatchingModel(SuggestMixin):
    """
    Logistic regression baseline on similarity features using statsmodels.

    This class is designed as an alternative to the DL/NS models in `neer_match`,
    using statsmodels' Logit on top of the similarity features produced by
    `AlternativeModels`.

    It supports:
    - evaluation with TP, FP, TN, FN, Accuracy, Recall, Precision, F1, MCC,
    - full inference via `summary()`.
    """

    result: sm.discrete.discrete_model.BinaryResultsWrapper | None = field(
        default=None, init=False
    )
    feature_cols: list[str] = field(default_factory=list, init=False)

    def fit(
        self,
        df: pd.DataFrame,
        match_col: str = "match",
        feature_prefix: str = "col_",
    ) -> "LogitMatchingModel":
        """
        Fit logistic regression on a pairwise similarity DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            (Possibly subsampled) DataFrame produced by AlternativeModels.
        match_col : str, default "match"
            Name of the binary target column.
        feature_prefix : str, default "col\_"
            Prefix of feature columns (similarity features).
        """
        if match_col not in df.columns:
            raise KeyError(f"Match column '{match_col}' not found in df.columns")

        # 1. Select feature columns
        feature_cols = sorted([c for c in df.columns if c.startswith(feature_prefix)])
        X = df[feature_cols]
        y = df[match_col].to_numpy().astype(int)

        # 2. Drop constant (zero-variance) columns – they cause singularities and add no info
        std = X.std(ddof=0)
        nonconstant_cols = std[std > 0].index.tolist()
        if len(nonconstant_cols) < len(feature_cols):
            # Optional: you could log / print which columns were dropped
            # dropped = sorted(set(feature_cols) - set(nonconstant_cols))
            # print(f"Dropping constant features: {dropped}")
            X = X[nonconstant_cols]
            feature_cols = nonconstant_cols

        self.feature_cols = feature_cols

        # 3. Add constant for intercept term
        X_sm = sm.add_constant(X, has_constant="add")

        # 4. Fit classical MLE logit, with regularized fallback
        model = sm.Logit(y, X_sm)

        try:
            # Try unpenalized MLE first
            self.result = model.fit(disp=0)
        except np.linalg.LinAlgError:
            # If Hessian is singular (separation / collinearity), try progressively stronger regularization
            for alpha in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                try:
                    self.result = model.fit_regularized(
                        alpha=alpha,
                        L1_wt=0.0,    # 0 → pure L2 (ridge)
                        maxiter=1000,
                        disp=0
                    )
                    if self.result is not None:
                        print(f"[LogitMatchingModel] Used regularization alpha={alpha} due to singular matrix")
                        break
                except (np.linalg.LinAlgError, ValueError):
                    continue
            else:
                raise RuntimeError(
                    "Could not fit logit model even with strong regularization. Possible causes:\n"
                    "  - Perfect separation: One or more features perfectly predict matches\n"
                    "  - Multicollinearity: Features are linear combinations of each other\n"
                    "Try: (1) More aggressive feature selection, (2) Remove problematic features, "
                    "(3) Increase training data size"
                )

        return self

    def _check_fitted(self):
        if self.result is None:
            raise RuntimeError("LogitMatchingModel is not fitted yet. Call `fit()` first.")

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_prefix: str = "col_",
    ) -> np.ndarray:
        self._check_fitted()

        if self.feature_cols:
            feature_cols = self.feature_cols
        else:
            feature_cols = sorted([c for c in df.columns if c.startswith(feature_prefix)])

        X = df[feature_cols]
        X_sm = sm.add_constant(X, has_constant="add")

        proba = self.result.predict(X_sm)
        return np.asarray(proba)

    def evaluate(
        self,
        df: pd.DataFrame,
        match_col: str = "match",
        feature_prefix: str = "col_",
        threshold: float = 0.5,
    ) -> dict:
        if match_col not in df.columns:
            raise KeyError(f"Match column '{match_col}' not found in DataFrame.")

        self._check_fitted()

        y_true = df[match_col].to_numpy().astype(int)
        proba = self.predict_proba(df, feature_prefix=feature_prefix)
        y_hat = (proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

        acc = accuracy_score(y_true, y_hat)
        prec = precision_score(y_true, y_hat, zero_division=0)
        rec = recall_score(y_true, y_hat, zero_division=0)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_hat) if tp + fp + tn + fn > 0 else 0.0

        return {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "Accuracy": float(acc),
            "Recall": float(rec),
            "Precision": float(prec),
            "F1": float(f1),
            "MCC": float(mcc),
        }

    def summary(self):
        self._check_fitted()
        return self.result.summary()
    

@dataclass
class ProbitMatchingModel(SuggestMixin):
    """
    Probit regression baseline on similarity features using statsmodels.

    Same interface as LogitMatchingModel, but using a normal CDF link.
    """

    result: sm.discrete.discrete_model.BinaryResultsWrapper | None = field(
        default=None, init=False
    )
    feature_cols: list[str] = field(default_factory=list, init=False)

    def fit(
        self,
        df: pd.DataFrame,
        match_col: str = "match",
        feature_prefix: str = "col_",
    ) -> "ProbitMatchingModel":
        """
        Fit probit regression on a pairwise similarity DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            (Possibly subsampled) DataFrame produced by AlternativeModels.
        match_col : str, default "match"
            Name of the binary target column.
        feature_prefix : str, default "col\_"
            Prefix of feature columns (similarity features).
        """
        if match_col not in df.columns:
            raise KeyError(f"Match column '{match_col}' not found in df.columns")

        # 1. Select feature columns
        feature_cols = sorted([c for c in df.columns if c.startswith(feature_prefix)])
        X = df[feature_cols]
        y = df[match_col].to_numpy().astype(int)

        # 2. Drop constant (zero-variance) columns
        std = X.std(ddof=0)
        nonconstant_cols = std[std > 0].index.tolist()
        if len(nonconstant_cols) < len(feature_cols):
            X = X[nonconstant_cols]
            feature_cols = nonconstant_cols

        self.feature_cols = feature_cols

        # 3. Add constant for intercept term
        X_sm = sm.add_constant(X, has_constant="add")

        # 4. Probit with ridge regularization (helps with separation)
        model = sm.Probit(y, X_sm)

        # Try progressively stronger regularization until fitting succeeds
        for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            try:
                self.result = model.fit_regularized(
                    alpha=alpha,
                    L1_wt=0.0,    # 0 → pure L2 (ridge)
                    maxiter=1000,
                    disp=0
                )
                if self.result is not None:
                    if alpha > 1e-4:  # Only print if we needed stronger regularization
                        print(f"[ProbitMatchingModel] Used regularization alpha={alpha} due to singular matrix")
                    break
            except (np.linalg.LinAlgError, ValueError):
                continue
        else:
            raise RuntimeError(
                "Could not fit probit model even with strong regularization. Possible causes:\n"
                "  - Perfect separation: One or more features perfectly predict matches\n"
                "  - Multicollinearity: Features are linear combinations of each other\n"
                "Try: (1) More aggressive feature selection, (2) Remove problematic features, "
                "(3) Increase training data size"
            )

        return self

    def _check_fitted(self):
        if self.result is None:
            raise RuntimeError("ProbitMatchingModel is not fitted yet. Call `fit()` first.")

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_prefix: str = "col_",
    ) -> np.ndarray:
        self._check_fitted()

        if self.feature_cols:
            feature_cols = self.feature_cols
        else:
            feature_cols = sorted([c for c in df.columns if c.startswith(feature_prefix)])

        X = df[feature_cols]
        X_sm = sm.add_constant(X, has_constant="add")

        proba = self.result.predict(X_sm)
        return np.asarray(proba)

    def evaluate(
        self,
        df: pd.DataFrame,
        match_col: str = "match",
        feature_prefix: str = "col_",
        threshold: float = 0.5,
    ) -> dict:
        if match_col not in df.columns:
            raise KeyError(f"Match column '{match_col}' not found in DataFrame.")

        self._check_fitted()

        y_true = df[match_col].to_numpy().astype(int)
        proba = self.predict_proba(df, feature_prefix=feature_prefix)
        y_hat = (proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

        acc = accuracy_score(y_true, y_hat)
        prec = precision_score(y_true, y_hat, zero_division=0)
        rec = recall_score(y_true, y_hat, zero_division=0)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_hat) if tp + fp + tn + fn > 0 else 0.0

        return {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "Accuracy": float(acc),
            "Recall": float(rec),
            "Precision": float(prec),
            "F1": float(f1),
            "MCC": float(mcc),
        }

    def summary(self):
        self._check_fitted()
        return self.result.summary()

@dataclass
class GradientBoostingModel(SuggestMixin):
    """
    Gradient boosting baseline on similarity features using scikit-learn.

    Designed as an alternative to the DL/NS models in `neer_match`, using a
    tree-based GradientBoostingClassifier on top of similarity features
    produced by `AlternativeModels`.

    It supports:
    - evaluation with TP, FP, TN, FN, Accuracy, Recall, Precision, F1, MCC,
    - a simple `summary()` reporting feature importances.

    Notes
    -----
    - Unlike Logit/Probit, this model has no statistical inference (SE/p-values).
    - Works well with nonlinearities and interactions in similarity features.
    """

    model: GradientBoostingClassifier = field(
        default_factory=lambda: GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=1.0,
            random_state=42,
        )
    )

    feature_cols: list[str] = field(default_factory=list, init=False)
    best_threshold_: float | None = field(default=None, init=False)

    def fit(
        self,
        df: pd.DataFrame,
        match_col: str = "match",
        feature_prefix: str = "col_",
        use_class_weight: bool = False,
    ) -> "GradientBoostingModel":
        """
        Fit gradient boosting on a pairwise similarity DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            (Possibly subsampled) DataFrame produced by AlternativeModels.
        match_col : str, default "match"
            Name of the binary target column.
        feature_prefix : str, default "col\_"
            Prefix of feature columns (similarity features).
        use_class_weight : bool, default False
            If True, uses inverse-frequency sample weights to upweight matches.
            Useful if you fit on a very imbalanced dataset.
        """
        if match_col not in df.columns:
            raise KeyError(f"Match column '{match_col}' not found in df.columns")

        # 1. Select feature columns
        feature_cols = sorted([c for c in df.columns if c.startswith(feature_prefix)])
        if not feature_cols:
            raise ValueError(f"No feature columns starting with {feature_prefix!r} found.")

        X = df[feature_cols]
        y = df[match_col].to_numpy().astype(int)

        # 2. Drop constant (zero-variance) columns
        std = X.std(ddof=0)
        nonconstant_cols = std[std > 0].index.tolist()
        if len(nonconstant_cols) < len(feature_cols):
            X = X[nonconstant_cols]
            feature_cols = nonconstant_cols

        self.feature_cols = feature_cols

        # 3. Optional class weighting via sample weights
        sample_weight = None
        if use_class_weight:
            # inverse frequency weights (balanced)
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            if n_pos == 0 or n_neg == 0:
                sample_weight = None
            else:
                w_pos = n_neg / (n_pos + n_neg)
                w_neg = n_pos / (n_pos + n_neg)
                sample_weight = np.where(y == 1, w_pos, w_neg)

        # 4. Fit
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def _check_fitted(self):
        if not hasattr(self.model, "estimators_"):
            raise RuntimeError("GradientBoostingModel is not fitted yet. Call `fit()` first.")

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_prefix: str = "col_",
    ) -> np.ndarray:
        """
        Predict match probabilities for a pairwise similarity DataFrame.

        Returns the probability for the positive class (match = 1).
        """
        self._check_fitted()

        feature_cols = self.feature_cols or sorted([c for c in df.columns if c.startswith(feature_prefix)])
        X = df[feature_cols]
        proba = self.model.predict_proba(X)[:, 1]
        return np.asarray(proba)

    def evaluate(
        self,
        df: pd.DataFrame,
        match_col: str = "match",
        feature_prefix: str = "col_",
        threshold: float = 0.5,
    ) -> dict:
        """
        Evaluate the model on a pairwise similarity DataFrame.

        Returns a dict:

        - TP, FP, TN, FN (integers)
        - Accuracy, Recall, Precision, F1, MCC (floats)
        """
        if match_col not in df.columns:
            raise KeyError(f"Match column '{match_col}' not found in DataFrame.")

        self._check_fitted()

        y_true = df[match_col].to_numpy().astype(int)
        proba = self.predict_proba(df, feature_prefix=feature_prefix)
        y_hat = (proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

        acc = accuracy_score(y_true, y_hat)
        prec = precision_score(y_true, y_hat, zero_division=0)
        rec = recall_score(y_true, y_hat, zero_division=0)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_hat) if (tp + fp + tn + fn) > 0 else 0.0

        return {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "Accuracy": float(acc),
            "Recall": float(rec),
            "Precision": float(prec),
            "F1": float(f1),
            "MCC": float(mcc),
        }

    def summary(self, top_k: int = 20) -> pd.DataFrame:
        """
        Return a simple "summary" as a DataFrame of feature importances.

        Parameters
        ----------
        top_k : int, default 20
            Number of most important features to return.

        Returns
        -------
        pd.DataFrame
            Columns: feature, importance
        """
        self._check_fitted()

        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            raise RuntimeError("Model does not expose feature_importances_.")

        df_imp = pd.DataFrame(
            {"feature": self.feature_cols, "importance": importances}
        ).sort_values("importance", ascending=False)

        return df_imp.head(top_k).reset_index(drop=True)

    def best_threshold(
        self,
        df_val: pd.DataFrame,
        match_col: str = "match",
        feature_prefix: str = "col_",
        metric: str = "mcc",
        thresholds: np.ndarray | None = None,
        store_treshold: bool = True,
    ) -> tuple[float, dict]:
        """
        Find the classification threshold that maximizes a metric on validation data.

        Parameters
        ----------
        df_val : pd.DataFrame
            Validation DataFrame produced by AlternativeModels.pairwise_similarity_dataframe().
        match_col : str, default "match"
            Target column.
        feature_prefix : str, default "col\_"
            Feature column prefix.
        metric : {"mcc","f1"}, default "mcc"
            Metric to maximize.
        thresholds : np.ndarray or None
            Threshold grid. If None, uses np.linspace(0.01, 0.99, 99).

        Returns
        -------
        best_t : float
            Threshold achieving the best metric on df_val.
        best_stats : dict
            Evaluation dict (TP/FP/TN/FN/Accuracy/Recall/Precision/F1/MCC) at best_t.
        """
        self._check_fitted()

        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)

        y_true = df_val[match_col].to_numpy().astype(int)
        proba = self.predict_proba(df_val, feature_prefix=feature_prefix)

        best_t = 0.5
        best_score = -np.inf
        best_stats = None

        for t in thresholds:
            y_hat = (proba >= t).astype(int)

            # choose metric
            if metric.lower() == "f1":
                score = f1_score(y_true, y_hat, zero_division=0)
            elif metric.lower() == "mcc":
                score = matthews_corrcoef(y_true, y_hat)
            else:
                raise ValueError("metric must be 'mcc' or 'f1'")

            if score > best_score:
                best_score = score
                best_t = float(t)
                best_stats = self.evaluate(
                    df_val,
                    match_col=match_col,
                    feature_prefix=feature_prefix,
                    threshold=float(t),
                )
            if store_treshold:
                self.best_threshold_ = best_t

        return best_t, best_stats
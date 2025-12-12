from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from neer_match.similarity_map import SimilarityMap

from neer_match_utilities.similarity_features import (
    SimilarityFeatures,
    subsample_non_matches,
)
from neer_match_utilities.baseline_models import (
    LogitMatchingModel,
    ProbitMatchingModel,
    GradientBoostingModel,
)
from neer_match_utilities.baseline_io import ModelBaseline
from neer_match_utilities.training import Training  # for performance_statistics_export


BaselineKind = Literal["logit", "probit", "gb"]


@dataclass
class BaselineTrainingPipe:
    """
    Orchestrates training + evaluation + export for baseline (non-DL) models:
    - LogitMatchingModel (statsmodels)
    - ProbitMatchingModel (statsmodels)
    - GradientBoostingModel (sklearn)

    Pipeline steps
    --------------
    1) Build full pairwise similarity DataFrames for train/val/test
    2) Subsample non-matches for fitting (optional)
    3) Fit selected baseline model
    4) Choose threshold (optional; recommended for GB)
    5) Evaluate on full train + test
    6) Save model via ModelBaseline.save(...)
    7) Export performance.csv + similarity_map.dill via Training.performance_statistics_export(...)
    """

    model_name: str
    similarity_map: dict | SimilarityMap

    # data
    training_data: Any  # (left_train, right_train, matches_train) or {"left":..,"right":..,"matches":..}
    testing_data: Any
    validation_data: Any | None = None

    # id columns
    id_left_col: str = "id"
    id_right_col: str = "id"

    # how to interpret matches df
    matches_id_left: str = "left"
    matches_id_right: str = "right"
    matches_are_indices: bool = True

    # model config
    model_kind: BaselineKind = "gb"

    # sampling
    mismatch_share_fit: float = 1.0
    random_state: int = 42
    shuffle_fit: bool = True

    # thresholding
    threshold: float = 0.5
    tune_threshold: bool = True
    tune_metric: Literal["mcc", "f1"] = "mcc"

    # export
    base_dir: Path | None = None
    export_model: bool = True
    export_stats: bool = True
    reload_sanity_check: bool = True

    # internals (filled during execute)
    model_: Any = field(default=None, init=False)
    best_threshold_: float | None = field(default=None, init=False)
    metrics_train_: dict | None = field(default=None, init=False)
    metrics_test_: dict | None = field(default=None, init=False)
    metrics_val_: dict | None = field(default=None, init=False)

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _unpack_split(obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj["left"], obj["right"], obj["matches"]
        left, right, matches = obj
        return left, right, matches

    def _smap(self) -> SimilarityMap:
        if isinstance(self.similarity_map, SimilarityMap):
            return self.similarity_map
        if isinstance(self.similarity_map, dict):
            return SimilarityMap(self.similarity_map)
        raise TypeError("similarity_map must be a dict or a SimilarityMap instance")

    def _make_model(self):
        if self.model_kind == "logit":
            return LogitMatchingModel()
        if self.model_kind == "probit":
            return ProbitMatchingModel()
        if self.model_kind == "gb":
            return GradientBoostingModel()
        raise ValueError(f"Unknown model_kind: {self.model_kind!r}")

    def _resolve_base_dir(self) -> Path:
        return Path.cwd() if self.base_dir is None else Path(self.base_dir)

    # ---------------------------
    # Main entry point
    # ---------------------------
    def execute(self) -> Any:
        base_dir = self._resolve_base_dir()
        smap = self._smap()

        left_train, right_train, matches_train = self._unpack_split(self.training_data)
        left_test, right_test, matches_test = self._unpack_split(self.testing_data)
        unpacked_val = self._unpack_split(self.validation_data)
        left_val = right_val = matches_val = None
        if unpacked_val is not None:
            left_val, right_val, matches_val = unpacked_val

        # 1) Build similarity features
        feats = SimilarityFeatures(similarity_map=smap)

        df_train = feats.pairwise_similarity_dataframe(
            left=left_train,
            right=right_train,
            matches=matches_train,
            left_id_col=self.id_left_col,
            right_id_col=self.id_right_col,
            match_col="match",
            matches_id_left=self.matches_id_left,
            matches_id_right=self.matches_id_right,
            matches_are_indices=self.matches_are_indices,
        )

        df_test = feats.pairwise_similarity_dataframe(
            left=left_test,
            right=right_test,
            matches=matches_test,
            left_id_col=self.id_left_col,
            right_id_col=self.id_right_col,
            match_col="match",
            matches_id_left=self.matches_id_left,
            matches_id_right=self.matches_id_right,
            matches_are_indices=self.matches_are_indices,
        )

        df_val = None
        if left_val is not None:
            df_val = feats.pairwise_similarity_dataframe(
                left=left_val,
                right=right_val,
                matches=matches_val,
                left_id_col=self.id_left_col,
                right_id_col=self.id_right_col,
                match_col="match",
                matches_id_left=self.matches_id_left,
                matches_id_right=self.matches_id_right,
                matches_are_indices=self.matches_are_indices,
            )

        # 2) Subsample for fitting
        df_fit = subsample_non_matches(
            df_train,
            match_col="match",
            mismatch_share=self.mismatch_share_fit,
            random_state=self.random_state,
            shuffle=self.shuffle_fit,
        )

        # 3) Fit model
        model = self._make_model()
        model.fit(df_fit, match_col="match")

        # 4) Threshold selection
        chosen_t = float(self.threshold)

        if self.model_kind == "gb" and self.tune_threshold:
            if df_val is None:
                raise ValueError("tune_threshold=True requires validation_data for GradientBoostingModel.")
            best_t, val_stats = model.best_threshold(df_val, metric=self.tune_metric)
            # store in-model + in-pipe
            self.best_threshold_ = float(best_t)
            chosen_t = float(best_t)
            self.metrics_val_ = val_stats
        else:
            self.best_threshold_ = chosen_t

        # 5) Evaluate
        metrics_train = model.evaluate(df_train, match_col="match", threshold=chosen_t)
        metrics_test = model.evaluate(df_test, match_col="match", threshold=chosen_t)

        self.model_ = model
        self.metrics_train_ = metrics_train
        self.metrics_test_ = metrics_test

        # If model supports storing the threshold (GB does in your code), keep it:
        if hasattr(model, "best_threshold_"):
            model.best_threshold_ = chosen_t

        # 6) Save model
        if self.export_model:
            ModelBaseline.save(
                model=model,
                target_directory=base_dir,
                name=self.model_name,
                similarity_map=smap,  # store instructions
            )

        # 7) Export performance + similarity map
        if self.export_stats:
            training_util = Training(
                similarity_map=smap.instructions,
                df_left=left_train,
                df_right=right_train,
                id_left=self.id_left_col,
                id_right=self.id_right_col,
            )
            training_util.performance_statistics_export(
                model=model,
                model_name=self.model_name,
                target_directory=base_dir,
                evaluation_train=metrics_train,
                evaluation_test=metrics_test,
                export_model=self.export_model,  # keep your signature if you added it
            )

        # 8) Optional reload sanity check
        if self.reload_sanity_check and self.export_model:
            loaded = ModelBaseline.load(base_dir / self.model_name)

            # Ensure we use the same threshold
            t_reload = chosen_t
            if getattr(loaded, "best_threshold_", None) is not None:
                t_reload = float(loaded.best_threshold_)

            mtr = loaded.evaluate(df_train, match_col="match", threshold=t_reload)
            mts = loaded.evaluate(df_test, match_col="match", threshold=t_reload)

            if mtr != metrics_train:
                raise AssertionError("Train metrics changed after reload!")
            if mts != metrics_test:
                raise AssertionError("Test metrics changed after reload!")

        return model
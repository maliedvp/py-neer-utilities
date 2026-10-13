from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from neer_match.similarity_map import SimilarityMap

from neer_match_utilities.similarity_features import (
    SimilarityFeatures,
    SimilarityPairs,
    subsample_non_match_positions,
)
from neer_match_utilities.baseline_models import (
    LogitMatchingModel,
    ProbitMatchingModel,
    GradientBoostingModel,
    classification_metrics,
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
    1) Set up the pairwise similarity features for train/val/test
    2) Subsample non-matches for fitting (optional), choosing the pairs before
       computing their features
    3) Fit selected baseline model
    4) Choose threshold (optional; recommended for GB)
    5) Save model via ModelBaseline.save(...)
    6) Evaluate on full train + test, together with the reloaded model for the
       sanity check
    7) Export performance.csv + similarity_map.dill via Training.performance_statistics_export(...)

    Similarity features are computed in blocks of at most ``pairs_per_chunk``
    pairs and discarded once scored, so memory does not grow with the number of
    pairs. Only the labels and predictions of all pairs are kept, a few bytes
    each. The fitted model does not depend on ``pairs_per_chunk``. Logit and
    probit probabilities come from matrix products, whose rounding can differ in
    the last bits between block sizes, so a pair scored within about 1e-15 of
    the threshold could be classified differently.
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
    # last, so that existing positional arguments keep their meaning
    overwrite: bool = False  # replace a model already saved under model_name
    pairs_per_chunk: int = 250_000  # pairs whose features are held in memory at once

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

        # Fail now rather than at step 6, after the features have been built
        model_dir = base_dir / self.model_name / "model"
        if self.export_model and not self.overwrite and model_dir.exists():
            raise FileExistsError(
                f"Directory '{model_dir}' already exists. Pass overwrite=True to "
                "replace the saved model, or choose a different model_name."
            )

        if self.model_kind == "gb" and self.tune_threshold and self.validation_data is None:
            raise ValueError("tune_threshold=True requires validation_data for GradientBoostingModel.")

        smap = self._smap()

        left_train, right_train, matches_train = self._unpack_split(self.training_data)
        left_test, right_test, matches_test = self._unpack_split(self.testing_data)
        unpacked_val = self._unpack_split(self.validation_data)
        left_val = right_val = matches_val = None
        if unpacked_val is not None:
            left_val, right_val, matches_val = unpacked_val

        # 1) Set up similarity features. Nothing is computed per pair yet.
        feats = SimilarityFeatures(similarity_map=smap)
        pair_args = dict(
            left_id_col=self.id_left_col,
            right_id_col=self.id_right_col,
            match_col="match",
            matches_id_left=self.matches_id_left,
            matches_id_right=self.matches_id_right,
            matches_are_indices=self.matches_are_indices,
        )

        train_pairs = feats.pairs(left_train, right_train, matches_train, **pair_args)
        test_pairs = feats.pairs(left_test, right_test, matches_test, **pair_args)

        val_pairs = None
        if left_val is not None:
            val_pairs = feats.pairs(left_val, right_val, matches_val, **pair_args)

        # 2) Subsample for fitting. The pairs are chosen from the match column
        #    alone, exactly as subsample_non_matches would choose them from the
        #    full frame, and only the chosen ones get their features computed.
        positions = subsample_non_match_positions(
            train_pairs.match_flags(self.pairs_per_chunk),
            mismatch_share=self.mismatch_share_fit,
            random_state=self.random_state,
            shuffle=self.shuffle_fit,
        )
        df_fit = train_pairs.rows(positions)

        # 3) Fit model
        model = self._make_model()
        model.fit(df_fit, match_col="match")
        del df_fit

        # suggest() needs the similarity map. ModelBaseline.load attaches it to a
        # reloaded model; attach it here too, so the returned model can suggest.
        model.similarity_map = smap

        # 4) Threshold selection
        chosen_t = float(self.threshold)

        if self.model_kind == "gb" and self.tune_threshold:
            y_val, proba_val = self._labels_and_probabilities(model, val_pairs)
            best_t, val_stats = model._best_threshold_from_proba(
                y_val, proba_val, metric=self.tune_metric
            )
            # store in-model + in-pipe
            self.best_threshold_ = float(best_t)
            chosen_t = float(best_t)
            self.metrics_val_ = val_stats
        else:
            self.best_threshold_ = chosen_t

        # If model supports storing the threshold (GB does in your code), keep it:
        if hasattr(model, "best_threshold_"):
            model.best_threshold_ = chosen_t

        # 5) Save model
        if self.export_model:
            ModelBaseline.save(
                model=model,
                target_directory=base_dir,
                name=self.model_name,
                similarity_map=smap,  # store instructions
                overwrite=self.overwrite,
            )

        # 6) Evaluate. For the reload sanity check, the saved model is loaded
        #    first and scored in the same pass, so that the features of every
        #    pair are computed only once.
        models, thresholds = [model], [chosen_t]
        if self.reload_sanity_check and self.export_model:
            loaded = ModelBaseline.load(base_dir / self.model_name)

            # Ensure we use the same threshold
            t_reload = chosen_t
            if getattr(loaded, "best_threshold_", None) is not None:
                t_reload = float(loaded.best_threshold_)

            models.append(loaded)
            thresholds.append(t_reload)

        metrics_train, *reloaded_train = self._evaluate(models, thresholds, train_pairs)
        metrics_test, *reloaded_test = self._evaluate(models, thresholds, test_pairs)

        self.model_ = model
        self.metrics_train_ = metrics_train
        self.metrics_test_ = metrics_test

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
                # The model was already written in step 5. Saving it again here would
                # overwrite that meta.pkl with one built from a different call.
                export_model=False,
            )

        # 8) Optional reload sanity check
        if reloaded_train:
            if reloaded_train[0] != metrics_train:
                raise AssertionError("Train metrics changed after reload!")
            if reloaded_test[0] != metrics_test:
                raise AssertionError("Test metrics changed after reload!")

        return model

    def _labels_and_probabilities(self, model, pairs: SimilarityPairs) -> tuple[np.ndarray, np.ndarray]:
        """Match labels and predicted probabilities of all pairs, scored block by block."""
        labels, probabilities = [], []
        for chunk in pairs.chunks(self.pairs_per_chunk):
            labels.append(chunk["match"].to_numpy().astype(int))
            probabilities.append(model.predict_proba(chunk))
        return np.concatenate(labels), np.concatenate(probabilities)

    def _evaluate(self, models: list, thresholds: list[float], pairs: SimilarityPairs) -> list[dict]:
        """
        `evaluate` of each model on all pairs, scored block by block.

        Per pair, only the label and each model's 0/1 prediction are kept.
        """
        labels = []
        predictions = [[] for _ in models]
        for chunk in pairs.chunks(self.pairs_per_chunk):
            labels.append(chunk["match"].to_numpy().astype(np.int8))
            for model, threshold, kept in zip(models, thresholds, predictions):
                kept.append((model.predict_proba(chunk) >= threshold).astype(np.int8))

        y_true = np.concatenate(labels)
        return [classification_metrics(y_true, np.concatenate(kept)) for kept in predictions]

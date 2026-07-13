from __future__ import annotations

from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from neer_match.similarity_map import SimilarityMap
import dill

from neer_match_utilities.baseline_models import (
    LogitMatchingModel,
    ProbitMatchingModel,
    GradientBoostingModel,
)


def _coef_table_to_payload(coef_table: pd.DataFrame | None) -> dict | None:
    """
    Reduce the regression table to plain Python types.

    Storing lists rather than a pickled DataFrame keeps ``meta.pkl`` readable across
    pandas versions, which a pickled pandas object is not.
    """
    if coef_table is None:
        return None
    return {
        "names": [str(i) for i in coef_table.index],
        "columns": [str(c) for c in coef_table.columns],
        "values": coef_table.to_numpy(dtype=float).tolist(),
    }


def _coef_table_from_payload(payload: dict | None) -> pd.DataFrame | None:
    """Rebuild the regression table written by :func:`_coef_table_to_payload`."""
    if not payload:
        return None
    return pd.DataFrame(
        payload["values"],
        index=payload["names"],
        columns=payload["columns"],
    )


class ModelBaseline:
    """
    Save/load utilities for non-DL baseline models:
    - LogitMatchingModel
    - ProbitMatchingModel
    - GradientBoostingModel
    """

    @staticmethod
    def save(
        model,
        target_directory: Path,
        name: str,
        similarity_map: SimilarityMap | dict | None = None,
        remove_data: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        similarity_map:
            Pass either a SimilarityMap instance OR the underlying dict (instructions).
            This is stored so that `ModelBaseline.load(...)` can return a model with
            `loaded_model.similarity_map` just like the DL models.
        remove_data:
            Only relevant for the statsmodels baselines (Logit/Probit). A statsmodels
            ``Results`` object keeps a reference to its ``Model``, and hence to the full
            ``endog``/``exog`` arrays. Since the design matrix here is the entire
            left x right cross join, pickling it verbatim costs roughly 1.9 KB per training
            pair -- tens of gigabytes on realistic data -- in order to persist a coefficient
            vector of a few hundred bytes.

            When True (default), the regression table is rendered and stored in ``meta.pkl``
            first, and the result is then pickled without its data. The artifact becomes a
            few hundred KB regardless of dataset size; ``predict_proba``, ``evaluate``,
            ``suggest`` and ``summary`` all keep working on the reloaded model. What is lost
            is the ability to recompute data-dependent quantities (residuals, influence
            measures) from the archive.

            Set to False to reproduce the old behavior and pickle the data along with
            the model.

        Notes
        -----
        With ``remove_data=True`` statsmodels strips the arrays from the *in-memory* result
        as well, so the passed-in ``model`` loses its residuals and influence measures too.
        Its coefficients, predictions and summary are unaffected.
        """
        target_directory = Path(target_directory) / name / "model"
        target_directory.mkdir(parents=True, exist_ok=True)

        # --- normalize similarity_map to a plain dict (like SimilarityMap.instructions) ---
        sim_map_dict = None
        if similarity_map is not None:
            if isinstance(similarity_map, SimilarityMap):
                sim_map_dict = similarity_map.instructions
            elif isinstance(similarity_map, dict):
                sim_map_dict = similarity_map
            else:
                raise TypeError("similarity_map must be SimilarityMap, dict, or None")

        # --- statsmodels models (Logit/Probit) ---
        if hasattr(model, "result") and model.result is not None:
            # Extract coefficients and the regression table into plain types while the data is
            # still attached. These make meta.pkl self-sufficient: a reloaded model can predict
            # and report its summary from them alone, even if the statsmodels pickle below
            # turns out to be unreadable under a different pandas.
            if hasattr(model, "capture_inference"):
                model.capture_inference()

            model.result.save(
                str(target_directory / "statsmodels_result.pkl"),
                remove_data=remove_data,
            )

            params = getattr(model, "params_", None)
            meta = {
                "model_type": type(model).__name__,
                "feature_cols": getattr(model, "feature_cols", []),
                "similarity_map": sim_map_dict,
                "data_removed": bool(remove_data),
                "summary_text": getattr(model, "summary_text_", None),
                "coef_table": _coef_table_to_payload(getattr(model, "coef_table_", None)),
                # Plain floats and strings -- deliberately not pandas, so that the archive
                # stays readable across pandas versions.
                "params": None if params is None else [float(v) for v in params],
                "param_names": list(getattr(model, "param_names_", None) or []) or None,
            }
            with open(target_directory / "meta.pkl", "wb") as f:
                pickle.dump(meta, f)

            # Human-readable copy, so the regression table is legible without Python.
            summary_text = getattr(model, "summary_text_", None)
            if summary_text:
                (target_directory / "summary.txt").write_text(summary_text)

            return

        # --- sklearn model (GradientBoostingModel) ---
        if isinstance(model, GradientBoostingModel):
            with open(target_directory / "sklearn_model.pkl", "wb") as f:
                pickle.dump(model.model, f)

            meta = {
                "model_type": "GradientBoostingModel",
                "feature_cols": getattr(model, "feature_cols", []),
                "similarity_map": sim_map_dict,
                # optional but recommended if you use threshold tuning:
                "best_threshold": getattr(model, "best_threshold_", None),
            }
            with open(target_directory / "meta.pkl", "wb") as f:
                pickle.dump(meta, f)

            return

        raise ValueError(f"Unsupported baseline model type: {type(model)}")

    @staticmethod
    def load(model_directory: Path):
        """
        Load a baseline model.

        Reads both layouts: models saved with the training data embedded in the statsmodels
        result (the original behavior), and models saved without it. In the latter case the
        regression table is restored from ``meta.pkl`` so that ``summary()`` and
        ``coefficients()`` keep working.
        """
        base_dir = Path(model_directory)          # .../default_model
        model_dir = base_dir / "model"            # .../default_model/model

        with open(model_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        model_type = meta["model_type"]
        feature_cols = meta.get("feature_cols", [])

        sim_map_dict = meta.get("similarity_map", None)

        # fallback: ../similarity_map.dill
        if sim_map_dict is None:
            dill_path = base_dir / "similarity_map.dill"
            if dill_path.exists():
                with open(dill_path, "rb") as f:
                    sim_map_dict = dill.load(f)

        sim_map_obj = SimilarityMap(sim_map_dict) if isinstance(sim_map_dict, dict) else None

        if model_type in {"LogitMatchingModel", "ProbitMatchingModel"}:
            model = LogitMatchingModel() if model_type == "LogitMatchingModel" else ProbitMatchingModel()
            model.feature_cols = feature_cols
            model.similarity_map = sim_map_obj

            # Absent for archives written before these were stored; there the statsmodels
            # result is the only source and `summary()` renders from it directly.
            model.summary_text_ = meta.get("summary_text", None)
            model.coef_table_ = _coef_table_from_payload(meta.get("coef_table", None))

            params = meta.get("params", None)
            names = meta.get("param_names", None)
            if params is not None and names:
                model.params_ = np.asarray(params, dtype=float)
                model.param_names_ = [str(n) for n in names]

            # Prefer the statsmodels result when it can be read, so behavior is identical to
            # before. A pandas pickled under one version is not guaranteed to load under
            # another, so failure here is expected rather than exceptional.
            result_path = model_dir / "statsmodels_result.pkl"
            try:
                model.result = sm.load(str(result_path))
            except Exception as exc:
                if model.params_ is None:
                    raise RuntimeError(
                        f"Could not read '{result_path}' ({type(exc).__name__}: {exc}), and "
                        f"'{model_dir / 'meta.pkl'}' holds no coefficients to fall back on.\n\n"
                        "This archive was written before the coefficients were stored "
                        "separately, so the pickled statsmodels result -- which embeds pandas "
                        "objects -- is the only copy of them. It can only be opened by a pandas "
                        "version compatible with the one that wrote it."
                    ) from exc

                warnings.warn(
                    f"Could not read '{result_path}' ({type(exc).__name__}: {exc}). "
                    "Falling back to the coefficients stored in meta.pkl; predictions and "
                    "summary are unaffected.",
                    RuntimeWarning,
                )
                model.result = None

            return model

        if model_type == "GradientBoostingModel":
            with open(model_dir / "sklearn_model.pkl", "rb") as f:      # <-- FIX
                gb = pickle.load(f)
            model = GradientBoostingModel()
            model.model = gb
            model.feature_cols = feature_cols
            model.similarity_map = sim_map_obj
            model.best_threshold_ = meta.get("best_threshold", None)
            return model

        raise ValueError(f"Unknown baseline model type: {model_type}")
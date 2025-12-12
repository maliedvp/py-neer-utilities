from __future__ import annotations

from pathlib import Path
import pickle
import statsmodels.api as sm
from neer_match.similarity_map import SimilarityMap
import dill

from neer_match_utilities.baseline_models import (
    LogitMatchingModel,
    ProbitMatchingModel,
    GradientBoostingModel,
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
    ) -> None:
        """
        Parameters
        ----------
        similarity_map:
            Pass either a SimilarityMap instance OR the underlying dict (instructions).
            This is stored so that `ModelBaseline.load(...)` can return a model with
            `loaded_model.similarity_map` just like the DL models.
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
            model.result.save(str(target_directory / "statsmodels_result.pkl"))

            meta = {
                "model_type": type(model).__name__,
                "feature_cols": getattr(model, "feature_cols", []),
                "similarity_map": sim_map_dict,
            }
            with open(target_directory / "meta.pkl", "wb") as f:
                pickle.dump(meta, f)

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
            res = sm.load(str(model_dir / "statsmodels_result.pkl"))   # <-- FIX
            model = LogitMatchingModel() if model_type == "LogitMatchingModel" else ProbitMatchingModel()
            model.result = res
            model.feature_cols = feature_cols
            model.similarity_map = sim_map_obj
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
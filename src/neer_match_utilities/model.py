from pathlib import Path
import pickle
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
from neer_match.similarity_map import SimilarityMap
import tensorflow as tf
import typing
import shutil
import sys


class Model:
    """
    A class for saving and loading matching models.

    Methods
    -------
    save(model, target_directory, name):
        Save the specified model to a target directory.
    load(model_directory):
        Load a model from a given directory.
    """

    @staticmethod
    def save(
        model: typing.Union["DLMatchingModel", "NSMatchingModel"],
        target_directory: Path,
        name: str,
    ) -> None:
        """
        Save the model to a specified directory.

        Parameters
        ----------
        model : DLMatchingModel or NSMatchingModel
            The model to be saved.
        target_directory : Path
            The directory where the model should be saved.
        name : str
            Name of the model directory.
        """

        target_directory = Path(target_directory) / name / "model"

        if target_directory.exists():
            replace = input(
                f"Directory '{target_directory}' already exists. Replace the old model? (y/n): "
            ).strip().lower()
            if replace == "y":
                shutil.rmtree(target_directory)
                print(f"Old model at '{target_directory}' has been replaced.")
            elif replace == "n":
                print("Execution halted as per user request.")
                sys.exit(0)
            else:
                print("Invalid input. Please type 'y' or 'n'. Aborting operation.")
                return

        target_directory.mkdir(parents=True, exist_ok=True)

        # --- Build composite similarity info ---
        # Use the original instructions stored in the SimilarityMap.
        # We assume model.similarity_map.instructions is a dict: { field: [metric1, metric2, ...], ... }
        instructions = model.similarity_map.instructions
        fields = list(instructions.keys())
        association_sizes = model.similarity_map.association_sizes()  # aggregated sizes per field
        composite_similarity_info = {}
        for i, field in enumerate(fields):
            agg_size = association_sizes[i]
            metrics = instructions[field]  # list of metric names as originally provided
            composite_similarity_info[field] = {
                "metrics": metrics,
                "aggregated_size": agg_size,
                "per_metric_size": agg_size // len(metrics)
            }

        # --- Save model initialization parameters from the record pair network ---
        model_params = {
            "initial_feature_width_scales": model.record_pair_network.initial_feature_width_scales,
            "feature_depths": model.record_pair_network.feature_depths,
            "initial_record_width_scale": model.record_pair_network.initial_record_width_scale,
            "record_depth": model.record_pair_network.record_depth,
        }

        # Save a composite dictionary containing both similarity info and model parameters.
        composite_save = {"similarity_info": composite_similarity_info, "model_params": model_params}
        with open(target_directory / "model_info.pkl", "wb") as f:
            pickle.dump(composite_save, f)
        # --- End composite info saving ---

        if isinstance(model, DLMatchingModel):
            model.save_weights(target_directory / "model.weights.h5")
            if hasattr(model, "optimizer") and model.optimizer:
                optimizer_config = {
                    "class_name": model.optimizer.__class__.__name__,
                    "config": model.optimizer.get_config(),
                }
                with open(target_directory / "optimizer.pkl", "wb") as f:
                    pickle.dump(optimizer_config, f)
        elif isinstance(model, NSMatchingModel):
            model.record_pair_network.save_weights(target_directory / "record_pair_network.weights.h5")
            if hasattr(model, "optimizer") and model.optimizer:
                optimizer_config = {
                    "class_name": model.optimizer.__class__.__name__,
                    "config": model.optimizer.get_config(),
                }
                with open(target_directory / "optimizer.pkl", "wb") as f:
                    pickle.dump(optimizer_config, f)
        else:
            raise ValueError("The model must be an instance of DLMatchingModel or NSMatchingModel")

        print(f"Model successfully saved to {target_directory}")

    @staticmethod
    def load(model_directory: Path) -> typing.Union[DLMatchingModel, NSMatchingModel]:
        """
        Load a model from a specified directory.

        Parameters
        ----------
        model_directory : Path
            The directory containing the saved model.

        Returns
        -------
        DLMatchingModel or NSMatchingModel
            The loaded model.
        """

        model_directory = Path(model_directory) / "model"
        if not model_directory.exists():
            raise FileNotFoundError(f"Model directory '{model_directory}' does not exist.")

        # --- Load composite model info (similarity info and model parameters) ---
        with open(model_directory / "model_info.pkl", "rb") as f:
            composite_save = pickle.load(f)
        composite_similarity_info = composite_save["similarity_info"]
        model_params = composite_save["model_params"]

        # Reconstruct the original similarity_map as expected by DLMatchingModel:
        # (a plain dict mapping each field to its list of metric names)
        original_similarity_map = {field: info["metrics"] for field, info in composite_similarity_info.items()}

        # IMPORTANT: Reconstruct a SimilarityMap instance from the plain dict.
        similarity_map_instance = SimilarityMap(original_similarity_map)

        # Compute aggregated sizes in the order of fields.
        fields = list(composite_similarity_info.keys())
        aggregated_sizes = [composite_similarity_info[field]["aggregated_size"] for field in fields]
        # --- End loading composite info ---

        if (model_directory / "model.weights.h5").exists():
            # Initialize the model using the reconstructed SimilarityMap instance and stored parameters.
            model = DLMatchingModel(
                similarity_map=similarity_map_instance,
                initial_feature_width_scales=model_params["initial_feature_width_scales"],
                feature_depths=model_params["feature_depths"],
                initial_record_width_scale=model_params["initial_record_width_scale"],
                record_depth=model_params["record_depth"],
            )
            input_shapes = [tf.TensorShape([None, s]) for s in aggregated_sizes]
            model.build(input_shapes=input_shapes)

            # --- Build dummy inputs as a list of tensors (one per field) ---
            # Each dummy tensor has shape (1, aggregated_size) for that field.
            dummy_tensors = [
                tf.zeros((1, composite_similarity_info[field]["aggregated_size"]))
                for field in fields
            ]
            # --- End dummy inputs ---

            _ = model(dummy_tensors)  # Forward pass to instantiate all sublayers.
            model.load_weights(model_directory / "model.weights.h5")

            if (model_directory / "optimizer.pkl").exists():
                with open(model_directory / "optimizer.pkl", "rb") as f:
                    optimizer_config = pickle.load(f)
                optimizer_class = getattr(tf.keras.optimizers, optimizer_config["class_name"])
                model.optimizer = optimizer_class.from_config(optimizer_config["config"])
        elif (model_directory / "record_pair_network.weights.h5").exists():
            model = NSMatchingModel(original_similarity_map)
            model.compile()
            model.record_pair_network.load_weights(model_directory / "record_pair_network.weights.h5")

            if (model_directory / "optimizer.pkl").exists():
                with open(model_directory / "optimizer.pkl", "rb") as f:
                    optimizer_config = pickle.load(f)
                optimizer_class = getattr(tf.keras.optimizers, optimizer_config["class_name"])
                model.optimizer = optimizer_class.from_config(optimizer_config["config"])
        else:
            raise ValueError("Invalid model directory: neither DLMatchingModel nor NSMatchingModel was detected.")

        return model

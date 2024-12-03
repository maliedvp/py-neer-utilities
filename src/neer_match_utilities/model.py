from pathlib import Path
import pickle
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
from neer_match.similarity_map import SimilarityMap
import tensorflow as tf
import typing


class Model:
    @staticmethod
    def save(
        model: typing.Union[DLMatchingModel, NSMatchingModel],
        target_directory: Path,
        name: str,
    ) -> None:
        target_directory = Path(target_directory) / name / "model"
        target_directory.mkdir(parents=True, exist_ok=True)

        # Save the similarity map
        with open(target_directory / "similarity_map.pkl", "wb") as f:
            pickle.dump(model.similarity_map, f)

        if isinstance(model, DLMatchingModel):
            model.save_weights(target_directory / "model.weights.h5")

            if hasattr(model, "optimizer") and model.optimizer is not None:
                optimizer_config = {
                    "class_name": model.optimizer.__class__.__name__,
                    "config": model.optimizer.get_config(),
                }
                with open(target_directory / "optimizer.pkl", "wb") as f:
                    pickle.dump(optimizer_config, f)

        elif isinstance(model, NSMatchingModel):
            model.record_pair_network.save_weights(
                target_directory / "record_pair_network.weights.h5"
            )
            if hasattr(model, "optimizer") and model.optimizer is not None:
                optimizer_config = {
                    "class_name": model.optimizer.__class__.__name__,
                    "config": model.optimizer.get_config(),
                }
                with open(target_directory / "optimizer.pkl", "wb") as f:
                    pickle.dump(optimizer_config, f)
        else:
            raise ValueError(
                "The model must be an instance of DLMatchingModel or NSMatchingModel"
            )

        print(f"Model successfully saved to {target_directory}")

    @staticmethod
    def load(model_directory: Path) -> typing.Union[DLMatchingModel, NSMatchingModel]:
        model_directory = Path(model_directory) / "model"
        if not model_directory.exists():
            raise FileNotFoundError(f"Model directory '{model_directory}' does not exist.")

        # Load the similarity map
        with open(model_directory / "similarity_map.pkl", "rb") as f:
            similarity_map = pickle.load(f)

        if (model_directory / "model.weights.h5").exists():
            model = DLMatchingModel(similarity_map)
            input_shapes = [
                tf.TensorShape([None, feature_size])
                for feature_size in similarity_map.association_sizes()
            ]
            model.build(input_shapes=input_shapes)
            model.load_weights(model_directory / "model.weights.h5")

            if (model_directory / "optimizer.pkl").exists():
                with open(model_directory / "optimizer.pkl", "rb") as f:
                    optimizer_config = pickle.load(f)
                optimizer_class = getattr(tf.keras.optimizers, optimizer_config["class_name"])
                model.optimizer = optimizer_class.from_config(optimizer_config["config"])

        elif (model_directory / "record_pair_network.weights.h5").exists():
            model = NSMatchingModel(similarity_map)
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

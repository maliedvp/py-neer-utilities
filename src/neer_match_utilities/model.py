from pathlib import Path
import pickle
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
from neer_match.similarity_map import SimilarityMap
import tensorflow as tf
from typing import Dict, List, Tuple
import typing
import uuid
import pandas as pd
import numpy as np
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

        Raises
        ------
        ValueError
            If the model is not an instance of DLMatchingModel or NSMatchingModel.
        """

        target_directory = Path(target_directory) / name / "model"

        if target_directory.exists():
            replace = input(f"Directory '{target_directory}' already exists. Replace the old model? (y/n): ").strip().lower()
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

        with open(target_directory / "similarity_map.pkl", "wb") as f:
            pickle.dump(model.similarity_map, f)

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
            model.record_pair_network.save_weights(
                target_directory / "record_pair_network.weights.h5"
            )
            if hasattr(model, "optimizer") and model.optimizer:
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

        Raises
        ------
        FileNotFoundError
            If the model directory does not exist.
        ValueError
            If the model type cannot be determined.
        """

        model_directory = Path(model_directory) / "model"
        if not model_directory.exists():
            raise FileNotFoundError(f"Model directory '{model_directory}' does not exist.")

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
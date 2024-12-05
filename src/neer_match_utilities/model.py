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
    @staticmethod
    def save(
        model: typing.Union["DLMatchingModel", "NSMatchingModel"],
        target_directory: Path,
        name: str,
    ) -> None:
        target_directory = Path(target_directory) / name / "model"
        
        # Check if the directory already exists
        if target_directory.exists():
            replace = input(f"Directory '{target_directory}' already exists. Replace the old model? (y/n): ").strip().lower()
            if replace == "y":
                # Remove the existing directory
                shutil.rmtree(target_directory)
                print(f"Old model at '{target_directory}' has been replaced.")
            elif replace == "n":
                print("Execution halted as per user request.")
                sys.exit(0)
            else:
                print("Invalid input. Please type 'y' or 'n'. Aborting operation.")
                return

        # Create the directory and save the model
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


class GenerateID:
    """
    A class to generate and harmonize unique IDs across time periods for panel data.

    Attributes:
        df_panel (pd.DataFrame): The panel dataset.
        panel_var (str): The panel identifier variable.
        time_var (str): The time period variable.
        subgroups (list): List of subgroup variables for slicing.
        model: A model object with a `suggest` method for generating ID suggestions.
        similarity_map (dict): A dictionary mapping column names to their similarity functions.
        prediction_threshold (float): Threshold for prediction acceptance. Default is 0.9.
    """

    def __init__(
        self, 
        df_panel: pd.DataFrame, 
        panel_var: str, 
        time_var: str, 
        model, 
        similarity_map: Dict, 
        prediction_threshold: float = 0.9, 
        subgroups: List[str] = None
    ):
        """
        Initialize the GenerateID class.

        Args:
            df_panel (pd.DataFrame): The panel dataset.
            panel_var (str): The panel identifier variable that is supposed to be created.
            time_var (str): The time period variable.
            subgroups (list, optional): List of subgroup variables for slicing.
            model: A model object with a `suggest` method.
            similarity_map (dict): A dictionary of similarity functions for columns.
            prediction_threshold (float, optional): Threshold for predictions. Defaults to 0.9.
        """
        subgroups = subgroups or []

        if df_panel.index.duplicated().any():
            raise ValueError("The index of df_panel contains duplicate entries.")

        self.panel_var = panel_var
        self.time_var = time_var
        self.subgroups = subgroups
        self.model = model
        self.similarity_map = similarity_map
        self.prediction_threshold = prediction_threshold

        # Ensure df_panel only includes relevant columns
        self.df_panel = df_panel[
            [col for col in df_panel.columns if col in similarity_map.keys() or col in subgroups or col == time_var]
        ]

    def group_by_subgroups(self):
        """
        Group the panel data into subgroups.

        Returns:
            pd.core.groupby.generic.DataFrameGroupBy: Grouped dataframe by subgroups.
        """
        return self.df_panel.groupby(self.subgroups)

    def generate_suggestions(self, df_slice: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
        """
        Generate ID suggestions for consecutive time periods.

        Args:
            df_slice (pd.DataFrame): A dataframe slice.

        Returns:
            Tuple[pd.DataFrame, List[int]]: A concatenated dataframe of suggestions and a list of periods.
        """
        periods = sorted(df_slice[self.time_var].unique())
        suggestions_dict = {}

        for idx, period in enumerate(periods[:-1]):
            print(f"Processing periods {period}-{periods[idx + 1]} at {pd.Timestamp.now()}")

            left = df_slice[df_slice[self.time_var] == period].reset_index(drop=False)
            right = df_slice[df_slice[self.time_var] == periods[idx + 1]].reset_index(drop=False)

            suggestions = self.model.suggest(left, right, count=1)
            suggestions = suggestions[suggestions['prediction'] >= self.prediction_threshold]

            suggestions = pd.merge(
                left[['index']], suggestions, left_index=True, right_on='left'
            )
            suggestions = pd.merge(
                suggestions, right[['index']], left_on='right', right_index=True, suffixes=('_left', '_right')
            )

            suggestions['period_left'] = period
            suggestions['period_right'] = periods[idx + 1]
            suggestions['periods_compared'] = f"{period}-{periods[idx + 1]}"

            suggestions.drop(columns=['left', 'right'], inplace=True)
            suggestions_dict[idx] = suggestions

        if suggestions_dict:
            suggestions_df = pd.concat(suggestions_dict.values(), ignore_index=True)
        else:
            suggestions_df = pd.DataFrame(columns=[
                'index_left',
                'prediction',
                'index_right',
                'period_left',
                'period_right',
                'periods_compared'
            ])

        return suggestions_df, periods

    def harmonize_ids(self, suggestions: pd.DataFrame, periods: List[int], original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonize IDs across time periods.

        Args:
            suggestions (pd.DataFrame): The dataframe with suggestions.
            periods (List[int]): List of periods.
            original_df (pd.DataFrame): The original dataframe.

        Returns:
            pd.DataFrame: Harmonized ID mapping.
        """
        unique_ids = list(original_df.index)

        id_mapping = pd.DataFrame({
            'index': unique_ids,
            'index_harm': unique_ids
        })

        for period in sorted(periods, reverse=True):
            replacement_map = dict(zip(id_mapping['index'], id_mapping['index_harm']))
            temp_df = suggestions[suggestions['period_right'] == period].copy()

            temp_df['id_left_harm'] = temp_df['index_left'].map(replacement_map)
            temp_df['id_right_harm'] = temp_df['index_right'].map(replacement_map)

            update_map = dict(zip(temp_df['id_right_harm'], temp_df['id_left_harm']))
            id_mapping['index_harm'] = id_mapping['index_harm'].map(update_map).fillna(id_mapping['index_harm']).astype(int)

        return id_mapping

    def assign_ids(self, id_mapping: pd.DataFrame) -> pd.DataFrame:
        """
        Assign unique IDs to the harmonized IDs.

        Args:
            id_mapping (pd.DataFrame): The harmonized ID mapping dataframe.

        Returns:
            pd.DataFrame: Dataframe with assigned unique IDs.
        """
        unique_indices = id_mapping['index_harm'].unique()
        id_map = {idx: uuid.uuid4() for idx in unique_indices}

        id_mapping[self.panel_var] = id_mapping['index_harm'].map(id_map)
        id_mapping.drop(columns=['index_harm'], inplace=True)
        return id_mapping.reset_index(drop=True)

    def execute(self) -> pd.DataFrame:
        """
        Execute the full ID generation and harmonization process.

        Returns:
            pd.DataFrame: The final ID mapping.
        """
        if self.subgroups:
            harmonized_dict = {}
            for subgroup, group_df in self.group_by_subgroups():
                suggestions, periods = self.generate_suggestions(group_df)
                harmonized_dict[subgroup] = self.harmonize_ids(suggestions, periods, group_df)

            id_mapping = pd.concat(harmonized_dict.values(), ignore_index=False)
        else:
            suggestions, periods = self.generate_suggestions(self.df_panel)
            id_mapping = self.harmonize_ids(suggestions, periods, self.df_panel)

        return self.assign_ids(id_mapping)

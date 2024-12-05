from .base import SuperClass
import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil
import dill
import os
import numpy as np

class Training(SuperClass):
    """
    A class for managing and evaluating training processes, including 
    reordering matches, evaluating performance metrics, and exporting models.

    Inherits:
    ---------
    SuperClass : Base class providing shared attributes and methods.
    """

    def matches_reorder(self, matches: pd.DataFrame, matches_id_left: str, matches_id_right: str):
        """
        Reorders a matches DataFrame to include indices from the left and 
        right DataFrames instead of their original IDs.

        Parameters:
        -----------
        matches : pd.DataFrame
            DataFrame containing matching pairs.
        matches_id_left : str
            Column name in the `matches` DataFrame corresponding to the left IDs.
        matches_id_right : str
            Column name in the `matches` DataFrame corresponding to the right IDs.

        Returns:
        --------
        pd.DataFrame
            A DataFrame with columns `left` and `right`, representing the indices
            of matching pairs in the left and right DataFrames.
        """
        # Add custom indices
        self.df_left['index_left'] = self.df_left.index
        self.df_right['index_right'] = self.df_right.index

        # Combine the datasets into one
        df = pd.merge(
            self.df_left, 
            matches, 
            left_on=self.id_left, 
            right_on=matches_id_left,
            how='right',
            validate='1:m',
            suffixes=('_l', '_r')
        )

        df = pd.merge(
            df,
            self.df_right,
            left_on=matches_id_right,
            right_on=self.id_right,
            how='left',
            validate='m:1',
            suffixes=('_l', '_r')
        )

        # Extract and rename index columns
        matches = df[['index_left', 'index_right']].rename(
            columns={
                'index_left': 'left', 
                'index_right': 'right'
            }
        ).reset_index(drop=True)

        matches = matches.sort_values(by='left', ascending=True).reset_index(drop=True)

        return matches

    def evaluate_dataframe(self, evaluation_test: dict, evaluation_train: dict):
        """
        Combines and evaluates test and training performance metrics.

        Parameters:
        -----------
        evaluation_test : dict
            Dictionary containing performance metrics for the test dataset.
        evaluation_train : dict
            Dictionary containing performance metrics for the training dataset.

        Returns:
        --------
        pd.DataFrame
            A DataFrame with accuracy, precision, recall, F-score, and a timestamp
            for both test and training datasets.
        """
        # Create DataFrames for test and training metrics
        df_test = pd.DataFrame([evaluation_test])
        df_test.insert(0, 'data', ['test'])

        df_train = pd.DataFrame([evaluation_train])
        df_train.insert(0, 'data', ['train'])

        # Concatenate and calculate metrics
        df = pd.concat([df_test, df_train], axis=0, ignore_index=True)

        df['timestamp'] = datetime.now()

        return df

    def performance_statistics_export(self, model, model_name: str, target_directory: Path, evaluation_train: dict = {}, evaluation_test: dict = {}):
        """
        Exports the trained model, similarity map, and evaluation metrics to the specified directory.

        Parameters:
        -----------
        model : Model object
            The trained model to export.
        model_name : str
            Name of the model to use as the export directory name.
        target_directory : Path
            The target directory where the model will be exported.
        evaluation_train : dict, optional
            Performance metrics for the training dataset (default is {}).
        evaluation_test : dict, optional
            Performance metrics for the test dataset (default is {}).

        Returns:
        --------
        None

        Notes:
        ------
        - The method creates a subdirectory named after `model_name` inside `target_directory`.
        - If `evaluation_train` and `evaluation_test` are provided, their metrics are saved as a CSV file.
        - Similarity maps are serialized using `dill` and saved in the export directory.
        """
        # Construct the full path for the model directory
        model_dir = target_directory / model_name

        # Ensure the directory exists
        if not model_dir.exists():
            os.mkdir(model_dir)
            print(f"Directory {model_dir} created for model export.")
        else:
            print(f"Directory {model_dir} already exists. Files will be written into it.")

        # Generate performance metrics and save
        if evaluation_test and evaluation_train:
            df_evaluate = self.evaluate_dataframe(evaluation_test, evaluation_train)
            df_evaluate.to_csv(model_dir / 'performance.csv', index=False)
            print(f"Performance metrics saved to {model_dir / 'performance.csv'}")


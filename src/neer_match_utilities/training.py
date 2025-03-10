from .base import SuperClass
import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil
import dill
import os
import numpy as np
import tensorflow.keras.backend as K


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

        Parameters
        ----------
        matches : pd.DataFrame
            DataFrame containing matching pairs.
        matches_id_left : str
            Column name in the `matches` DataFrame corresponding to the left IDs.
        matches_id_right : str
            Column name in the `matches` DataFrame corresponding to the right IDs.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns `left` and `right`, representing the indices
            of matching pairs in the left and right DataFrames.
        """
        
        # Create local copies of the original dataframes
        df_left = self.df_left.copy()
        df_right = self.df_right.copy()


        # Add custom indices
        df_left['index_left'] = self.df_left.index
        df_right['index_right'] = self.df_right.index

        # Combine the datasets into one
        df = pd.merge(
            df_left, 
            matches, 
            left_on=self.id_left, 
            right_on=matches_id_left,
            how='right',
            validate='1:m',
            suffixes=('_l', '_r')
        )

        df = pd.merge(
            df,
            df_right,
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

        Parameters
        ----------
        evaluation_test : dict
            Dictionary containing performance metrics for the test dataset.
        evaluation_train : dict
            Dictionary containing performance metrics for the training dataset.

        Returns
        -------
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


def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal Loss function for binary classification tasks.

    Focal Loss is designed to address class imbalance by assigning higher weights
    to the minority class and focusing the model's learning on hard-to-classify examples.
    It reduces the loss contribution from well-classified examples, making it
    particularly effective for imbalanced datasets.

    Parameters
    ----------
    alpha : float, optional, default=0.25
        Weighting factor for the positive class (minority class).

        - Must be in the range [0, 1].
        - A higher value increases the loss contribution from the positive class
          (underrepresented class) relative to the negative class (overrepresented class).

    gamma : float, optional, default=2.0
        Focusing parameter that reduces the loss contribution from easy examples.

        - ``gamma = 0``: No focusing, equivalent to Weighted Binary Cross-Entropy Loss.
        - ``gamma > 0``: Focuses more on hard-to-classify examples.
        - Larger values emphasize harder examples more strongly.

    Returns
    -------
    loss : callable
        A loss function that computes the focal loss given the true labels (`y_true`)
        and predicted probabilities (`y_pred`).

    Raises
    ------
    ValueError
        If `alpha` is not in the range [0, 1].

    Notes
    -----
    - The positive class (minority or underrepresented class) is weighted by `alpha`.
    - The negative class (majority or overrepresented class) is automatically weighted
      by ``1 - alpha``.
    - Ensure `alpha` is set appropriately to reflect the level of imbalance in the dataset.

    References
    ----------
    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection. In ICCV.

    Explanation of Key Terms
    -------------------------
    - **Positive Class (Underrepresented):**

      - Refers to the class with fewer examples in the dataset.
      - Typically weighted by `alpha`, which should be greater than 0.5 in highly imbalanced datasets.

    - **Negative Class (Overrepresented):**

      - Refers to the class with more examples in the dataset.
      - Its weight is automatically ``1 - alpha``.
    """

    if not (0 <= alpha <= 1):
        raise ValueError("Parameter `alpha` must be in the range [0, 1].")

    def loss(y_true, y_pred):
        # Compute the binary cross-entropy
        bce = K.binary_crossentropy(y_true, y_pred)

        # Compute p_t, the probability of the true class
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        # Apply focal loss scaling
        return K.mean(alpha * K.pow(1 - p_t, gamma) * bce)

    return loss


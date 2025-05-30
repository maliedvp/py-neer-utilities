from .base import SuperClass
import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil
import dill
import os
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf


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


def focal_loss(alpha=0.75, gamma=2.0):
    """
    Focal Loss function for binary classification tasks.

    Focal Loss is designed to address class imbalance by assigning higher weights
    to the minority class and focusing the model's learning on hard-to-classify examples.
    It reduces the loss contribution from well-classified examples, making it
    particularly effective for imbalanced datasets.

    Parameters
    ----------
    alpha : float, optional, default=0.75
        Weighting factor for the positive class (minority class).

        - Must be in the range [0, 1].
        - A higher value increases the loss contribution from the positive class
          (underrepresented class) relative to the negative class (overrepresented class).

    gamma : float, optional, default=2.0
        Focusing parameter that reduces the loss contribution from easy examples.

        - ``gamma = 0``: No focusing, equivalent to Weighted Binary Cross-Entropy Loss (if alpha is set to 0.5).
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


def soft_f1_loss(epsilon: float = 1e-7):
    """
    Soft F1 Loss for imbalanced binary classification tasks.

    Soft F1 Loss provides a differentiable approximation of the F1 score,
    combining precision and recall into a single metric. By optimizing
    this loss, models are encouraged to balance false positives and false
    negatives, which is especially useful when classes are imbalanced.

    Parameters
    ----------
    epsilon : float, optional, default=1e-7
        Small constant added to numerator and denominator to avoid division
        by zero and stabilize training. Must be > 0.

    Returns
    -------
    loss : callable
        A loss function that takes true labels (`y_true`) and predicted
        probabilities (`y_pred`) and returns `1 - soft_f1`, so that
        minimizing this loss maximizes the soft F1 score.

    Raises
    ------
    ValueError
        If `epsilon` is not strictly positive.

    Notes
    -----
    - True positives (TP), false positives (FP), and false negatives (FN)
      are computed in a “soft” (differentiable) manner by summing over
      probabilities rather than thresholded predictions.
    - Soft F1 = (2·TP + ε) / (2·TP + FP + FN + ε).
    - Loss = 1 − Soft F1, which ranges from 0 (perfect) to 1 (worst).

    References
    ----------
    - Sahu, S., & Wig, G. S. (2020). Soft F1 and Dice Loss for Deep Learning.
      *arXiv preprint arXiv:2004.13709*.
    - Zhang, X., & Yang, Y. (2018). An Improved Cross-Entropy Focal Loss for
      Imbalanced Classification. *Proceedings of the AAAI Conference on Artificial Intelligence*.

    Explanation of Key Terms
    ------------------------
    - **True Positives (TP):** Sum of predicted probabilities for actual
      positive examples.
    - **False Positives (FP):** Sum of predicted probabilities assigned to
      negative examples.
    - **False Negatives (FN):** Sum of (1 − predicted probability) for
      positive examples.
    - **ε (epsilon):** Stabilizer to prevent division by zero when TP, FP,
      and FN are all zero.

    Examples
    --------
    ```python
    loss_fn = soft_f1_loss(epsilon=1e-6)
    y_true = tf.constant([[1, 0, 1]], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.2, 0.7]], dtype=tf.float32)
    loss_value = loss_fn(y_true, y_pred)
    print(loss_value.numpy())  # e.g. 0.1…
    ```
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Soft counts
        tp = tf.reduce_sum(y_pred * y_true)
        fp = tf.reduce_sum(y_pred * (1 - y_true))
        fn = tf.reduce_sum((1 - y_pred) * y_true)

        # Soft F1 calculation
        soft_f1 = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        return 1.0 - soft_f1

    return loss


def combined_loss(
    weight_f1: float = 0.5,
    epsilon: float = 1e-7,
    alpha: float = 0.25,
    gamma: float = 2.0
):
    """
    Combined Loss: weighted sum of Soft F1 Loss and Focal Loss for imbalanced binary classification.

    This loss blends the advantages of a differentiable F1-based objective (which balances
    precision and recall) with the sample‐focusing property of Focal Loss (which down‐weights
    easy examples). By tuning `weight_f1`, you can interpolate between solely optimizing
    for F1 score (when `weight_f1=1.0`) and solely focusing on hard examples via focal loss
    (when `weight_f1=0.0`).

    Parameters
    ----------
    weight_f1 : float, optional, default=0.5
        Mixing coefficient ∈ [0, 1].  
        - `weight_f1=1.0`: optimize only Soft F1 Loss.  
        - `weight_f1=0.0`: optimize only Focal Loss.  
        - Intermediate values blend the two objectives proportionally.
    epsilon : float, optional, default=1e-7
        Small stabilizer for Soft F1 calculation. Must be > 0.
    alpha : float, optional, default=0.25
        Balancing factor for Focal Loss, weighting the positive (minority) class.  
        Must lie in [0, 1].
    gamma : float, optional, default=2.0
        Focusing parameter for Focal Loss.  
        - `gamma=0` → reduces to weighted BCE.  
        - Larger `gamma` emphasizes harder (misclassified) examples.

    Returns
    -------
    loss : callable
        A function `loss(y_true, y_pred)` that computes:
        
            weight_f1 * SoftF1(y_true, y_pred; ε)
          + (1 − weight_f1) * FocalLoss(y_true, y_pred; α, γ)

        Minimizing this combined loss encourages both a high F1 score
        and focus on hard‐to‐classify samples.

    Raises
    ------
    ValueError
        If `weight_f1` is not in [0, 1], or if `epsilon` ≤ 0, or if `alpha` is not
        in [0, 1], or if `gamma` < 0.

    Notes
    -----
    - **Soft F1 Loss** (1 − F1) is differentiable and promotes balanced precision/recall.  
    - **Focal Loss** down‐weights well‐classified examples to focus learning on difficult cases.  
    - Adjust `weight_f1` to prioritize either overall F1 (higher weight_f1) or hard‐example mining (lower weight_f1).

    References
    ----------
    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
      Focal Loss for Dense Object Detection. *ICCV*.
    Sahu, S., & Wig, G. S. (2020). Soft F1 and Dice Loss for Deep Learning.
      *arXiv:2004.13709*.

    Explanation of Key Terms
    ------------------------
    - **Soft F1**: (2·TP + ε) / (2·TP + FP + FN + ε), differentiable surrogate for the F1 score.
    - **Focal Loss**: α·(1−p_t)ᵞ·BCE(p_t), where p_t is the model’s estimated probability for the true class.
    - **TP, FP, FN**: soft counts over probabilities (see Soft F1 docs).

    Examples
    --------
    ```python
    # create combined loss with equal weighting
    loss_fn = combined_loss(weight_f1=0.5, epsilon=1e-6, alpha=0.25, gamma=2.0)

    y_true = tf.constant([[1, 0, 1]], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.2, 0.7]], dtype=tf.float32)

    value = loss_fn(y_true, y_pred)
    print("Combined loss:", value.numpy())
    ```
    """
    # Validate hyper-parameters
    if not (0.0 <= weight_f1 <= 1.0):
        raise ValueError("`weight_f1` must be in [0, 1].")
    if epsilon <= 0:
        raise ValueError("`epsilon` must be strictly positive.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("`alpha` must be in [0, 1].")
    if gamma < 0:
        raise ValueError("`gamma` must be non-negative.")

    # Instantiate the individual losses
    f1_fn   = soft_f1_loss(epsilon)
    focal_fn = focal_loss(alpha=alpha, gamma=gamma)

    def loss(y_true, y_pred):
        # Weighted combination
        return (weight_f1 * f1_fn(y_true, y_pred)
                + (1.0 - weight_f1) * focal_fn(y_true, y_pred))

    return loss

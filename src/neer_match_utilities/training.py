from __future__ import annotations
from .base import SuperClass

from neer_match.similarity_map import SimilarityMap
from neer_match.matching_model import DLMatchingModel
from neer_match_utilities.model import Model, EpochEndSaver

import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil
import dill
import os
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
import math


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


def focal_loss(alpha=0.99, gamma=1.5):
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
        # numerical safety
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1.0 - eps)

        # per-example alpha: alpha for positive, (1-alpha) for negative
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

        # p_t is the prob of the true class
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # BCE equals -log(p_t) when reduced per-example
        bce = -K.log(p_t)

        # focal modulation and weighting
        fl = alpha_t * K.pow(1.0 - p_t, gamma) * bce
        return K.mean(fl)

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
    - Bénédict, G., Koops, V., Odijk D., & de Rijke M. (2021). SigmoidF1: A 
      Smooth F1 Score Surrogate Loss for Multilabel Classification. *arXiv 2108.10566*.

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
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon)

        # Soft counts
        tp = tf.reduce_sum(y_pred * y_true)
        fp = tf.reduce_sum(y_pred * (1 - y_true))
        fn = tf.reduce_sum((1 - y_pred) * y_true)

        # Denominator
        denom = 2 * tp + fp + fn + epsilon

        # Avoid NaNs from 0/0
        soft_f1 = tf.where(denom > 0, (2 * tp + epsilon) / denom, tf.constant(0.0))

        loss_val = 1.0 - soft_f1
        return tf.where(tf.math.is_finite(loss_val), loss_val, tf.constant(1.0))

    return loss


def combined_loss(
    weight_f1: float = 0.5,
    epsilon: float = 1e-7,
    alpha: float = 0.99,
    gamma: float = 1.5
):
    """
    Combined loss: weighted sum of Soft F1 loss and Focal Loss for imbalanced binary classification.

    This loss blends the advantages of a differentiable F1-based objective (which balances
    precision and recall) with the sample-focusing property of Focal Loss (which down-weights
    easy examples). By tuning ``weight_f1``, you can interpolate between solely optimizing
    for F1 score (when ``weight_f1 = 1.0``) and solely focusing on hard examples via focal loss
    (when ``weight_f1 = 0.0``).

    Parameters
    ----------
    weight_f1 : float, default=0.5
        Mixing coefficient in ``[0, 1]``.
        - ``weight_f1 = 1.0``: optimize only Soft F1 loss.
        - ``weight_f1 = 0.0``: optimize only Focal Loss.
        - Intermediate values blend the two objectives proportionally.
    epsilon : float, default=1e-7
        Small stabilizer for Soft F1 calculation. Must be ``> 0``.
    alpha : float, default=0.25
        Balancing factor for Focal Loss, weighting the positive (minority) class.
        Must lie in ``[0, 1]``.
    gamma : float, default=2.0
        Focusing parameter for Focal Loss.
        - ``gamma = 0`` reduces to weighted BCE.
        - Larger ``gamma`` emphasizes harder (misclassified) examples.

    Returns
    -------
    callable
        A function ``loss(y_true, y_pred)`` that computes

        .. math::

           \\text{CombinedLoss}
           = \\text{weight\\_f1} \\cdot \\text{SoftF1}(y, \\hat{y};\\,\\varepsilon)
             + (1 - \\text{weight\\_f1}) \\cdot \\text{FocalLoss}(y, \\hat{y};\\,\\alpha, \\gamma).

        Minimizing this combined loss encourages both a high F1 score
        and focus on hard-to-classify samples.

    Raises
    ------
    ValueError
        If ``weight_f1`` is not in ``[0, 1]``, or if ``epsilon <= 0``, or if ``alpha`` is not
        in ``[0, 1]``, or if ``gamma < 0``.

    Notes
    -----
    - **Soft F1 loss**: ``1 - \\text{SoftF1}``, where

      .. math::

         \\text{SoftF1} = \\frac{2\\,TP + \\varepsilon}{2\\,TP + FP + FN + \\varepsilon}.

      Here ``TP``, ``FP``, and ``FN`` are *soft* counts computed from probabilities.
    - **Focal Loss** down-weights well-classified examples to focus learning on difficult cases.

    References
    ----------
    - Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
      Focal Loss for Dense Object Detection. *ICCV*.
    - Bénédict, G., Koops, V., Odijk, D., & de Rijke, M. (2021).
      SigmoidF1: A Smooth F1 Score Surrogate Loss for Multilabel Classification. *arXiv:2108.10566*.

    Examples
    --------
    .. code-block:: python

       import tensorflow as tf
       loss_fn = combined_loss(weight_f1=0.5, epsilon=1e-6, alpha=0.25, gamma=2.0)

       y_true = tf.constant([[1, 0, 1]], dtype=tf.float32)
       y_pred = tf.constant([[0.9, 0.2, 0.7]], dtype=tf.float32)

       value = loss_fn(y_true, y_pred)
       print("Combined loss:", float(value.numpy()))
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

def alpha_balanced(left, right, matches, mismatch_share:float=1.0, max_alpha:float=.95) -> float:
    """
    Compute α so that α*N_pos = (1-α)*N_neg.

    Parameters
    ----------
    left, right : pandas.DataFrame
    matches     : pandas.DataFrame

    Returns
    -------
    float
        α in [0,1] for focal loss (positive-class weight).
    """
    N_pos   = len(matches)
    N_neg   = len(left)*len(right)-len(matches)

    alpha   = (mismatch_share * N_neg) / (mismatch_share * N_neg + N_pos)

    alpha = min(alpha, max_alpha)

    N_total = len(left) * len(right)
    if N_total <= 0:
        raise ValueError("Total number of pairs is zero.")

    return alpha


class TrainingPipe:
    """
    Orchestrates the full training and evaluation process of a deep learning
    record-linkage model using a user-supplied similarity map and preprocessed data.

    The class handles both training phases (soft-F1 pretraining and focal-loss fine-tuning),
    dynamic learning-rate scheduling, and automatic weight-decay adaptation. It also exports
    checkpoints, final models, and evaluation statistics for reproducibility.

    Parameters:
    -----------
    model_name : str
        Name assigned to the trained model. A corresponding subdirectory is created
        under the project directory to store checkpoints and exports.

    training_data : tuple or dict
        Preprocessed training data in one of the following formats:
        - Tuple: (left_train, right_train, matches_train)
        - Dict: {"left": left_train, "right": right_train, "matches": matches_train}

    testing_data : tuple or dict
        Preprocessed testing data in one of the following formats:
        - Tuple: (left_test, right_test, matches_test)
        - Dict: {"left": left_test, "right": right_test, "matches": matches_test}

    similarity_map : dict
        User-defined similarity configuration mapping variable names to similarity measures.
        Must follow the format accepted by `SimilarityMap`.

    id_left_col : str, optional
        Name of the unique identifier column in the left DataFrames
        (`left_train` and `left_test`).
        The ID column is used internally to index entities and to align
        training labels. Defaults to `"id_unique"`.

    id_right_col : str, optional
        Name of the unique identifier column in the right DataFrames
        (`right_train` and `right_test`).
        The ID column is used internally to index entities and to align
        training labels. Defaults to `"id_unique"`.

    no_tm_pbatch : int
        Target number of positive (matching) pairs per batch. Used to adapt batch size
        dynamically via the `required_batch_size` heuristic.

    save_architecture: bool, optional
        Whether to save the model architecture in an image alongside weights when exporting.
        Requires binaries of of graphviz to be installed. Otherwise, the code breaks.
        Defaults to `False`.

    stage_1 : bool, optional
        Whether to run the first training stage (soft-F1 pretraining).  
        If False, the arguments `epochs_1` and `mismatch_share_1` are not required
        and will be ignored.

    stage_2 : bool, optional
        Whether to run the second training stage (focal-loss fine-tuning).  
        If False, the arguments `epochs_2`, `mismatch_share_2`, `no_tm_pbatch`,
        `gamma`, and `max_alpha` are not required and will be ignored.

    epochs_1 : int, optional
        Number of training epochs during the first phase (soft-F1 pretraining).
        Required only when `stage_1=True`.

    mismatch_share_1 : float, optional
        Fraction of all possible negative (non-matching) pairs used during Round 1.
        Required only when `stage_1=True`.

    stage1_loss : str or callable, optional
        Loss function used during Stage 1 (pretraining).  
        By default, this is ``soft_f1_loss()``, reproducing the original NeerMatch
        behavior.  

        The argument accepts either:

        - A **string** specifying a built-in or predefined loss:
            * ``"soft_f1"`` — use the standard soft-F1 loss (default)
            * ``"binary_crossentropy"`` — use wrapped binary crossentropy
            (internally adapted for NeerMatch’s evaluation loop)

        - A **callable loss function**, allowing full customization:
            * ``soft_f1_loss()`` — explicit soft-F1 loss
            * ``focal_loss(alpha=0.25, gamma=2.0)`` — focal loss with parameters
            * Any user-defined loss function of signature ``loss(y_true, y_pred)``

    epochs_2 : int, optional
        Number of training epochs during the second phase (focal-loss fine-tuning).
        Required only when `stage_2=True`.

    mismatch_share_2 : float, optional
        Fraction of sampled negative pairs used during Round 2.
        Required only when `stage_2=True`.

    gamma : float, optional
        Focusing parameter of the focal loss (Round 2).  
        Required only when `stage_2=True`.

    max_alpha : float, optional
        Maximum weighting factor of the positive class for focal loss (Round 2).  
        Required only when `stage_2=True`.

    Returns:
    --------
    None

    Notes:
    ------
    - The pipeline assumes that the data have already been preprocessed, formatted, and tokenized.
    - Round 1 (soft-F1 phase) initializes the model and emphasizes balanced learning across classes.
    - Round 2 (focal-loss phase) refines the model to focus on hard-to-classify examples.
    - Dynamic heuristics are used to automatically infer:
        * Batch size (via expected positive density)
        * Peak learning rate (scaled with batch size, positives per batch, and parameter count)
        * Weight decay (adjusted based on model size and learning rate)
    - Model checkpoints, histories, and evaluation reports are stored in subdirectories named
      after the provided `model_name`.
    - The final model, similarity map, and performance metrics are exported to disk using the
      `Training.performance_statistics_export` method for reproducibility.
    - Each training stage can be enabled or disabled independently through
    the `stage_1` and `stage_2` flags.
    - If a stage is disabled, its hyperparameters are not required and will
    be ignored.
    - When only one stage is active, the warm-up pass automatically adapts
    to the active stage's mismatch sampling configuration.
    """

    # ---------- Built-in helpers ----------
    @staticmethod
    def required_batch_size(num_matches: int,
                            num_left: int,
                            num_right: int,
                            mismatch_share: float,
                            desired_pos_per_batch: int = 8,
                            eps: float = 1e-12) -> tuple[int, float]:
        total_mismatches = max(num_left * num_right - num_matches, 0)
        sampled_negatives = mismatch_share * total_mismatches
        denom = num_matches + sampled_negatives
        if denom <= eps or num_matches == 0:
            return max(desired_pos_per_batch, 1), 0.0
        q = num_matches / denom
        q = max(min(q, 1.0), eps)
        batch_size = math.ceil(desired_pos_per_batch / q)
        return batch_size, batch_size * q

    @staticmethod
    def count_trainable_params(keras_model) -> int:
        return int(sum(v.shape.num_elements() for v in keras_model.trainable_variables))

    @staticmethod
    def suggest_peak_lr_adamw(batch_size: int,
                              positives_per_batch: float | None = None,
                              param_count: int | None = None,
                              base_batch: int = 256,
                              base_lr: float = 1e-3,
                              min_lr: float = 3e-4,
                              max_lr: float = 3e-3) -> float:
        lr = base_lr * (batch_size / float(base_batch))
        if positives_per_batch is not None:
            x = max(min(positives_per_batch, 16.0), 1.0)
            shrink = 0.6 + 0.4 * ((x - 1.0) / 15.0)  # in [0.6, 1.0]
            lr *= shrink
        if param_count is not None:
            scale = math.log10(max(param_count, 1)) - math.log10(10_000_000)
            lr *= (0.8 ** max(scale, 0))
        return float(max(min(lr, max_lr), min_lr))

    @staticmethod
    def _suggest_weight_decay_adamw(batch_size: int,
                                    param_count: int,
                                    learning_rate: float,
                                    base_batch: int = 256,
                                    base_params: int = 10_000_000,
                                    base_lr: float = 1e-3,
                                    base_wd: float = 5e-4,
                                    wd_min: float = 1e-5,
                                    wd_max: float = 5e-3) -> float:
        batch_scale = base_batch / float(batch_size)
        batch_scale = min(max(batch_scale, 0.5), 2.0)
        param_scale = (base_params / float(param_count)) ** 0.2
        param_scale = min(max(param_scale, 0.5), 2.0)
        lr_scale = (learning_rate / base_lr) ** 0.5
        lr_scale = min(max(lr_scale, 0.5), 2.0)
        wd = base_wd * batch_scale * param_scale * lr_scale
        return float(max(min(wd, wd_max), wd_min))

    # ---------- Inner: WarmupCosine schedule ----------
    @tf.keras.utils.register_keras_serializable(package="custom")
    class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, peak_lr, warmup_steps, total_steps, min_lr_ratio=0.1, name=None):
            super().__init__()
            self.peak_lr = float(peak_lr)
            self.warmup_steps = int(warmup_steps)
            self.total_steps = int(total_steps)
            self.min_lr_ratio = float(min_lr_ratio)
            self.name = name

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            peak = tf.cast(self.peak_lr, tf.float32)
            warm_steps = tf.cast(self.warmup_steps, tf.float32)
            tot_steps = tf.cast(self.total_steps, tf.float32)
            min_ratio = tf.cast(self.min_lr_ratio, tf.float32)

            warm = peak * (step / tf.maximum(warm_steps, 1.0))
            cos = peak * (
                min_ratio
                + 0.5 * (1.0 - min_ratio)
                * (1.0 + tf.cos(math.pi * (step - warm_steps) /
                                tf.maximum(tot_steps - warm_steps, 1.0)))
            )
            return tf.where(step < warm_steps, warm, cos)

        def get_config(self):
            return {
                "peak_lr": self.peak_lr,
                "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps,
                "min_lr_ratio": self.min_lr_ratio,
                "name": self.name,
            }

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    # ---------- Pipeline ----------
    def __init__(
        self,
        model_name: str,
        training_data,   # (left_train, right_train, matches_train) OR dict with keys
        testing_data,    # (left_test, right_test, matches_test)   OR dict with keys
        similarity_map: dict,
        *,
        initial_feature_width_scales: int = 10,
        feature_depths: int = 2,
        initial_record_width_scale: int = 10,
        record_depth: int = 4,
        id_left_col: str = "id_unique",
        id_right_col: str = "id_unique",
        no_tm_pbatch: int | None = None,
        save_architecture: bool = False,
        stage_1: bool = True,
        stage_2: bool = True,
        epochs_1: int | None = None,
        mismatch_share_1: float | None = None,
        stage1_loss: str | callable = soft_f1_loss(),
        epochs_2: int | None = None,
        mismatch_share_2: float | None = None,
        gamma: float | None = None,
        max_alpha: float | None = None,
    ):
        if similarity_map is None:
            raise ValueError("similarity_map is required and must not be None.")
        self.similarity_map = similarity_map

        self.model_name = model_name
        self.initial_feature_width_scales = initial_feature_width_scales
        self.feature_depths = feature_depths
        self.initial_record_width_scale = initial_record_width_scale
        self.record_depth = record_depth

        self.stage_1 = bool(stage_1)
        self.stage_2 = bool(stage_2)

        if not self.stage_1 and not self.stage_2:
            raise ValueError("At least one of stage_1 or stage_2 must be True.")

        # ---- Stage 1 hyperparameters ----
        if self.stage_1:
            missing_stage1 = {
                name: val
                for name, val in [
                    ("epochs_1", epochs_1),
                    ("mismatch_share_1", mismatch_share_1),
                    ("no_tm_pbatch", no_tm_pbatch),  # NEW
                ]
                if val is None
            }
            if missing_stage1:
                raise ValueError(
                    "The following arguments must be provided when stage_1=True: "
                    + ", ".join(missing_stage1.keys())
                )

            self.epochs_1 = int(epochs_1)
            self.mismatch_share_1 = float(mismatch_share_1)
            self.no_tm_pbatch = int(no_tm_pbatch)

        # ---- Stage 1 loss configuration ----
        if not (isinstance(stage1_loss, str) or callable(stage1_loss)):
            raise ValueError("stage1_loss must be a loss name (str) or a callable.")
        self.stage1_loss = stage1_loss

        # ---- Stage 2 hyperparameters ----
        if self.stage_2:
            missing = {
                name: val
                for name, val in [
                    ("epochs_2", epochs_2),
                    ("mismatch_share_2", mismatch_share_2),
                    ("no_tm_pbatch", no_tm_pbatch),
                    ("gamma", gamma),
                    ("max_alpha", max_alpha),
                ]
                if val is None
            }
            if missing:
                missing_keys = ", ".join(missing.keys())
                raise ValueError(
                    f"The following arguments must be provided when stage_2=True: {missing_keys}."
                )

            self.epochs_2 = int(epochs_2)
            self.mismatch_share_2 = float(mismatch_share_2)
            self.no_tm_pbatch = int(no_tm_pbatch)
            self.gamma = float(gamma)
            self.max_alpha = float(max_alpha)
        else:
            # dummy values; not used when stage_2=False
            self.epochs_2 = 0
            self.mismatch_share_2 = 0.0
            self.no_tm_pbatch = 0
            self.gamma = 0.0
            self.max_alpha = 0.0

        self.id_left_col = id_left_col
        self.id_right_col = id_right_col

        self.save_architecture = save_architecture

        # Unpack user-supplied data
        self.left_train, self.right_train, self.matches_train = self._unpack_split(training_data)
        self.left_test, self.right_test, self.matches_test = self._unpack_split(testing_data)

        # Basic sanity checks
        for name, df, col in [
            ("left_train", self.left_train, self.id_left_col),
            ("right_train", self.right_train, self.id_right_col),
            ("left_test", self.left_test, self.id_left_col),
            ("right_test", self.right_test, self.id_right_col),
        ]:
            if col not in df.columns:
                raise ValueError(f"{name} must include column '{col}'.")

        self.base_dir = Path.cwd()
        self.model: DLMatchingModel | None = None

        # Minimal Training helper for exporting stats at the end
        self.training_util = Training(
            similarity_map=self.similarity_map,
            df_left=self.left_train,
            df_right=self.right_train,
            id_left=self.id_left_col,
            id_right=self.id_right_col,
        )

    @staticmethod
    def _unpack_split(obj):
        if isinstance(obj, dict):
            return obj["left"], obj["right"], obj["matches"]
        left, right, matches = obj
        return left, right, matches

    # ---------- Public entry point ----------
    def execute(self):
        # Build model with provided similarity map
        smap = SimilarityMap(self.similarity_map)
        self.model = DLMatchingModel(
            similarity_map=smap, 
            initial_feature_width_scales = self.initial_feature_width_scales,
            feature_depths = self.feature_depths,
            initial_record_width_scale = self.initial_record_width_scale,
            record_depth = self.record_depth,
        )

        # Warmup pass to initialize shapes/BN, etc.
        # Use mismatch_share_1 if stage_1 is active, otherwise fall back to stage_2.
        warmup_mismatch_share = (
            self.mismatch_share_1 if self.stage_1 else self.mismatch_share_2
        )

        bsz_warm, _ = self.required_batch_size(
            len(self.matches_train),
            len(self.left_train),
            len(self.right_train),
            warmup_mismatch_share,
            desired_pos_per_batch=16,
        )
        self.model.compile()
        self.model.fit(
            self.left_train,
            self.right_train,
            self.matches_train,
            epochs=1,
            batch_size=bsz_warm,
            mismatch_share=0.05,
            shuffle=True,
        )

        # Count params
        P = self.count_trainable_params(self.model)

        # Round 1 (optional)
        if self.stage_1:
            self._round1(P)

        # Round 2 (optional)
        if self.stage_2:
            self._round2(P)

        # Save, evaluate, export
        self._finalize_and_report()

    # ---------- Rounds ----------
    def _round1(self, P: int):
        assert self.model is not None

        batch_size, expected_pos = self.required_batch_size(
            len(self.matches_train), len(self.left_train), len(self.right_train),
            self.mismatch_share_1, desired_pos_per_batch=16
        )
        peak_lr = self.suggest_peak_lr_adamw(batch_size, expected_pos, P)

        no_obs_1 = self.mismatch_share_1 * (len(self.left_train) * len(self.right_train) - len(self.matches_train)) + len(self.matches_train)
        steps_per_epoch_1 = int(round(no_obs_1 / batch_size, 0))
        total_steps_1 = steps_per_epoch_1 * self.epochs_1
        warmup_steps_1 = int(0.1 * total_steps_1)

        lr_sched_1 = self.WarmupCosine(peak_lr, warmup_steps_1, total_steps_1, min_lr_ratio=0.001)
        wd_1 = self._suggest_weight_decay_adamw(batch_size, P, peak_lr)
        opt_1 = tf.keras.optimizers.AdamW(learning_rate=lr_sched_1, weight_decay=wd_1, clipnorm=1.0)

        loss_1 = self._get_stage1_loss()
        self.model.compile(loss=loss_1, optimizer=opt_1)

        saver = EpochEndSaver(base_dir=self.base_dir, model_name=self.model_name)
        history = self.model.fit(
            self.left_train, self.right_train, self.matches_train,
            epochs=self.epochs_1, batch_size=batch_size,
            mismatch_share=self.mismatch_share_1, shuffle=True, callbacks=[saver]
        )

        df_epoch = pd.DataFrame(history.history)
        df_epoch["epoch"] = pd.Series(history.epoch, dtype=int) + 1
        (self.base_dir / self.model_name / "checkpoints").mkdir(parents=True, exist_ok=True)
        df_epoch.to_csv(self.base_dir / self.model_name / "checkpoints" / "epoch_overview.csv", index=False)

        best_row = df_epoch.loc[df_epoch["f1"].idxmax()]
        best_epoch = int(best_row["epoch"])
        best_epoch_str = f"{best_epoch:02d}"

        # Load best R1 model
        self.model = Model.load(self.base_dir / self.model_name / "checkpoints" / f"epoch_{best_epoch_str}")
        print("[R1] best epoch:", best_epoch, "path:", self.base_dir / self.model_name / "checkpoints_round1" / f"epoch_{best_epoch_str}")

        # Sanity-evaluate the loaded model before Round 2 training
        probe_loss = self._get_stage1_loss()
        self.model.compile(loss=probe_loss, optimizer=tf.keras.optimizers.Adam())  # dummy opt
        eval_probe = self.model.evaluate(
            self.left_train,
            self.right_train,
            self.matches_train,
            mismatch_share=self.mismatch_share_1,
            verbose=0,
        )
        print("[R1->R2] probe metrics after load:", eval_probe)

        # Preserve Round 1 checkpoints
        old_path = self.base_dir / self.model_name / "checkpoints"
        new_path = self.base_dir / self.model_name / "checkpoints_stage_1"
        if old_path.exists():
            old_path.rename(new_path)

    def _get_stage1_loss(self):
        """
        Resolve the configured Stage 1 loss into a *callable* loss function.

        - If self.stage1_loss is "soft_f1": return soft_f1_loss()
        (this preserves the original behavior)
        - If self.stage1_loss is "binary_crossentropy": return a wrapped BCE
        that reshapes labels/preds to be compatible with NeerMatch's evaluate loop.
        - If self.stage1_loss is a callable (e.g. soft_f1_loss(), focal_loss(...)):
        return it as-is.
        """
        # Case 1: user passed a ready-made callable, e.g. soft_f1_loss() or focal_loss(...)
        if callable(self.stage1_loss):
            return self.stage1_loss

        # Case 2: string selector
        if self.stage1_loss == "soft_f1":
            # Default behavior: identical to original code
            return soft_f1_loss()

        if self.stage1_loss == "binary_crossentropy":
            # Wrap BCE to be robust to whatever shapes NeerMatch passes in
            def bce_wrapped(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                # Flatten to 1D so shapes always match
                y_true = tf.reshape(y_true, (-1,))
                y_pred = tf.reshape(y_pred, (-1,))
                return tf.keras.losses.binary_crossentropy(y_true, y_pred)
            return bce_wrapped

        # You can add more named options here if you like
        raise ValueError(f"Unknown stage1_loss string: {self.stage1_loss}")

    def _round2(self, P: int):
        assert self.model is not None

        # alpha bounded by your alpha_balanced (expects max_alpha argument in your implementation)
        alpha = alpha_balanced(
            left=self.left_train, right=self.right_train, matches=self.matches_train,
            mismatch_share=self.mismatch_share_2, max_alpha=self.max_alpha
        )

        batch_size, expected_pos = self.required_batch_size(
            len(self.matches_train), len(self.left_train), len(self.right_train),
            self.mismatch_share_2, desired_pos_per_batch=self.no_tm_pbatch
        )
        no_obs_2 = self.mismatch_share_2 * (len(self.left_train) * len(self.right_train) - len(self.matches_train)) + len(self.matches_train)
        steps_per_epoch_2 = int(round(no_obs_2 / batch_size, 0))
        total_steps_2 = steps_per_epoch_2 * self.epochs_2
        warmup_steps_2 = int(0.1 * total_steps_2)

        peak_lr_2 = self.suggest_peak_lr_adamw(batch_size, expected_pos, P)
        peak_lr_2 = min(peak_lr_2, 1e-4)  # your cap

        lr_sched_2 = self.WarmupCosine(peak_lr_2, warmup_steps_2, total_steps_2, min_lr_ratio=0.1)
        wd_2 = self._suggest_weight_decay_adamw(batch_size, P, peak_lr_2)
        opt_2 = tf.keras.optimizers.AdamW(learning_rate=lr_sched_2, weight_decay=wd_2, clipnorm=1.0)

        self.model.compile(loss=focal_loss(alpha=alpha, gamma=self.gamma), optimizer=opt_2)

        saver = EpochEndSaver(base_dir=self.base_dir, model_name=self.model_name)
        history = self.model.fit(
            self.left_train, self.right_train, self.matches_train,
            epochs=self.epochs_2, batch_size=batch_size,
            mismatch_share=self.mismatch_share_2, shuffle=True, callbacks=[saver]
        )

        df_epoch = pd.DataFrame(history.history)
        df_epoch["epoch"] = pd.Series(history.epoch, dtype=int) + 1
        (self.base_dir / self.model_name / "checkpoints").mkdir(parents=True, exist_ok=True)
        df_epoch.to_csv(self.base_dir / self.model_name / "checkpoints" / "epoch_overview.csv", index=False)

        best_row = df_epoch.loc[df_epoch["f1"].idxmax()]
        best_epoch = int(best_row["epoch"])
        best_epoch_str = f"{best_epoch:02d}"
        self.model = Model.load(self.base_dir / self.model_name / "checkpoints" / f"epoch_{best_epoch_str}")
        print("[R2] best epoch:", best_epoch, "path:", self.base_dir / self.model_name / "checkpoints" / f"epoch_{best_epoch_str}")

        # Re-compile for export/eval
        self.model.compile(loss=focal_loss(alpha=alpha, gamma=self.gamma), optimizer=opt_2)

        # Re-name checkpoints folder
        old_path = self.base_dir / self.model_name / "checkpoints"
        new_path = self.base_dir / self.model_name / "checkpoints_stage_2"
        if old_path.exists():
            if new_path.exists():
                shutil.rmtree(new_path)  # optional: only needed if rerunning
            old_path.rename(new_path)

    # ---------- Save / evaluate ----------
    def _finalize_and_report(self):
        assert self.model is not None and self.training_util is not None

        # Save the final bundle
        Model.save(model=self.model, target_directory=self.base_dir, name=self.model_name, save_architecture=self.save_architecture)

        # Evaluate on supplied splits
        perf_train = self.model.evaluate(self.left_train, self.right_train, self.matches_train, mismatch_share=1.0)
        perf_test = self.model.evaluate(self.left_test, self.right_test, self.matches_test, mismatch_share=1.0)

        # Export stats
        self.training_util.performance_statistics_export(
            model=self.model, model_name=self.model_name, target_directory=self.base_dir,
            evaluation_train=perf_train, evaluation_test=perf_test
        )
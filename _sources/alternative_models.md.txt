# Alternative Classification Models


This document covers all classification models available in
`neer_match_utilities` for entity matching. The examples assume you have
already prepared your data as described in [basic training
pipeline](basic_training_pipeline.md).

## Baseline Models

Baseline models (Logit, Probit, GradientBoost) use the
`BaselineTrainingPipe` class. They are faster to train and serve as good
benchmarks.

### Logit Model

Logistic regression using statsmodels. A simple, interpretable baseline.

``` python
from neer_match_utilities.baseline_training import BaselineTrainingPipe

training_pipeline = BaselineTrainingPipe(
    # Required
    model_name='my_logit_model',
    similarity_map=smap,
    training_data=(left_train, right_train, matches_train),
    testing_data=(left_test, right_test, matches_test),

    # Model type
    model_kind="logit",  # "logit" | "probit" | "gb"

    # ID columns (must match your data)
    id_left_col="company_id",
    id_right_col="company_id",

    # How matches dataframe is structured
    matches_id_left="left",       # column name for left IDs in matches
    matches_id_right="right",     # column name for right IDs in matches
    matches_are_indices=True,     # True if matches contain row indices, False if IDs

    # Sampling: fraction of non-matches to use during fitting
    mismatch_share_fit=1.0,       # 1.0 = use all, 0.1 = use 10%
    random_state=42,
    shuffle_fit=True,

    # Prediction threshold
    threshold=0.5,
    tune_threshold=False,         # Logit/Probit: typically use 0.5

    # Optional: validation data for threshold tuning
    validation_data=None,         # (left_val, right_val, matches_val)

    # Export settings
    base_dir=None,                # defaults to current directory
    export_model=True,
    export_stats=True,
    reload_sanity_check=True,     # verify model can be reloaded
)

training_pipeline.execute()
```

### Probit Model

Probit regression using statsmodels. Similar to Logit but uses the
cumulative normal distribution.

``` python
training_pipeline = BaselineTrainingPipe(
    model_name='my_probit_model',
    similarity_map=smap,
    training_data=(left_train, right_train, matches_train),
    testing_data=(left_test, right_test, matches_test),

    model_kind="probit",

    id_left_col="company_id",
    id_right_col="company_id",
    matches_id_left="left",
    matches_id_right="right",
    matches_are_indices=True,

    mismatch_share_fit=1.0,
    threshold=0.5,
    tune_threshold=False,

    export_model=True,
    export_stats=True,
)

training_pipeline.execute()
```

### Gradient Boosting Model

Gradient Boosting using scikit-learn. More powerful but less
interpretable.

``` python
training_pipeline = BaselineTrainingPipe(
    model_name='my_gb_model',
    similarity_map=smap,
    training_data=(left_train, right_train, matches_train),
    testing_data=(left_test, right_test, matches_test),

    model_kind="gb",

    id_left_col="company_id",
    id_right_col="company_id",
    matches_id_left="left",
    matches_id_right="right",
    matches_are_indices=True,

    # Sampling (GB often works well with subsampling)
    mismatch_share_fit=0.5,       # use 50% of non-matches

    # Threshold tuning (recommended for GB)
    tune_threshold=True,          # automatically find best threshold
    tune_metric="mcc",            # "mcc" or "f1"
    validation_data=(left_val, right_val, matches_val),  # required for tuning

    export_model=True,
    export_stats=True,
)

training_pipeline.execute()
```

## Deep Learning Model (ANN)

The neural network model uses `TrainingPipe` and supports two-stage
training with customizable loss functions.

``` python
from neer_match_utilities.training import TrainingPipe

training_pipeline = TrainingPipe(
    # Required
    model_name='my_ann_model',
    similarity_map=similarity_map,  # dict format, not SimilarityMap object
    training_data=(left_train, right_train, matches_train),
    testing_data=(left_test, right_test, matches_test),

    # ID columns
    id_left_col="company_id",
    id_right_col="company_id",

    # Network architecture
    initial_feature_width_scales=10,  # width multiplier for feature networks
    feature_depths=2,                  # depth of feature networks
    initial_record_width_scale=10,     # width multiplier for record network
    record_depth=4,                    # depth of record network

    # Stage 1: Soft-F1 pretraining
    stage_1=True,
    epochs_1=50,
    mismatch_share_1=0.01,            # fraction of non-matches per epoch
    stage1_loss="soft_f1",            # "soft_f1" | "binary_crossentropy" | callable

    # Stage 2: Focal loss fine-tuning
    stage_2=True,
    epochs_2=30,
    mismatch_share_2=0.1,
    gamma=2.0,                        # focal loss focusing parameter
    max_alpha=0.9,                    # max weight for positive class

    # Batch size control
    no_tm_pbatch=8,                   # target positives per batch

    # Export
    save_architecture=False,          # requires graphviz binaries
)

training_pipeline.execute()
```

### Key Parameters Explained

**Network Architecture:**

- `initial_feature_width_scales`: Controls the width of feature-specific
  networks. Higher values create wider networks.
- `feature_depths`: Number of layers in each feature network.
- `initial_record_width_scale`: Controls the width of the final
  record-comparison network.
- `record_depth`: Number of layers in the record network.

**Training Stages:**

- `stage_1`: Pretraining phase using soft-F1 loss to learn basic
  matching patterns.
- `stage_2`: Fine-tuning phase using focal loss to focus on hard
  examples.
- You can disable either stage by setting it to `False`.

**Sampling:**

- `mismatch_share_1/2`: Fraction of non-matches to sample per epoch.
  Lower values speed up training but may reduce quality.
- `no_tm_pbatch`: Target number of positive pairs per batch. The actual
  batch size is calculated automatically.

**Focal Loss (Stage 2):**

- `gamma`: Focusing parameter. Higher values focus more on hard examples
  (typical: 1.0-3.0).
- `max_alpha`: Maximum class weight for positives. Balances class
  imbalance.

### Single-Stage Training

You can run only one training stage:

``` python
# Only Stage 1 (faster, simpler)
training_pipeline = TrainingPipe(
    model_name='my_model_stage1_only',
    similarity_map=similarity_map,
    training_data=(left_train, right_train, matches_train),
    testing_data=(left_test, right_test, matches_test),
    id_left_col="company_id",
    id_right_col="company_id",

    stage_1=True,
    epochs_1=100,
    mismatch_share_1=0.05,
    no_tm_pbatch=8,

    stage_2=False,  # Skip stage 2
)

training_pipeline.execute()
```

## Model Comparison

| Model  | Linear | Speed  | Interpretability |
|--------|--------|--------|------------------|
| Logit  | Yes    | Fast   | High             |
| Probit | Yes    | Fast   | High             |
| GB     | No     | Medium | Low              |
| ANN    | No     | Slow   | Low              |

**Note on performance:** Model performance depends heavily on the
specific use case, dataset characteristics, and hyperparameter tuning.
ANNs are more prone to getting stuck in local minima, making their
results more volatile across runs. Always compare against baseline
models for your specific use case.

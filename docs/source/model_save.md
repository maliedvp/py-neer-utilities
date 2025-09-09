# Saving a Model


## Prepare the Data

This tutorial illustrates how to correctly store a `neer-match` model.
Before getting to the part that illustrates the saving, we have to
repeat the steps detailed in **Data Preparation for Training**:

``` python
import random
import pandas as pd

from neer_match.similarity_map import SimilarityMap
from neer_match_utilities.panel import SetupData
from neer_match_utilities.prepare import Prepare
from neer_match_utilities.training import Training
from neer_match_utilities.split import split_test_train
from neer_match_utilities.custom_similarities import CustomSimilarities

random.seed(42)

# Load files

matches = pd.read_csv('matches.csv')
left = pd.read_csv('left.csv')
right = pd.read_csv('right.csv')

# Preparation

left, right, matches = SetupData(matches=matches).data_preparation_cs(
    df_left=left,
    df_right=right,
    unique_id='company_id'
)

# Load custom similarities dedicated to missing values from the utilities package

CustomSimilarities()

# Define similarity_map
 
similarity_map = {
    "company_name" : [
        "levenshtein",
        "jaro_winkler",
        "prefix",
        "postfix",
        "token_sort_ratio",
        "token_set_ratio",
        "partial_token_set_ratio",
        "partial_token_sort_ratio",
    ],
    "city" : [
        "levenshtein",
        "jaro_winkler",
        "notmissing",
    ],
    "industry" : [
        "levenshtein",
        "jaro_winkler",
        "notmissing",
    ],
    "purpose" : [
        "levenshtein",
        "jaro_winkler",
        "token_sort_ratio",
        "token_set_ratio",
        "partial_token_set_ratio",
        "partial_token_sort_ratio",
        "notmissing",
    ],
    "bs_text" : [
        "levenshtein",
        "jaro_winkler",
        "token_sort_ratio",
        "token_set_ratio",
        "partial_token_set_ratio",
        "partial_token_sort_ratio",
        "notmissing",
    ],
    "found_year" : [
        "discrete",
        "notzero"
    ],
    "found_date_modified" : [
        "discrete",
        "notmissing",
    ]
}

smap = SimilarityMap(similarity_map)

prepare = Prepare(
    similarity_map=similarity_map, 
    df_left=left, 
    df_right=right, 
    id_left='company_id', 
    id_right='company_id'
)

left, right = prepare.format(
    fill_numeric_na=False,
    to_numeric=['found_year'],
    fill_string_na=True, 
    capitalize=True
)

# Re-Structuring

training = Training(
    similarity_map=similarity_map, 
    df_left=left, 
    df_right=right, 
    id_left='company_id', 
    id_right='company_id'
)

matches = training.matches_reorder(
    matches, 
    matches_id_left='left', 
    matches_id_right='right'
)

# Splitting data

left_train, right_train, matches_train, left_validation, right_validation, matches_validation, left_test, right_test, matches_test = split_test_train(
    left = left,
    right = right,
    matches = matches,
    test_ratio = .7,
    validation_ratio = .0
)
```

## Train the Model

Once the data is in the correct structure, training the model follows.
Note that this example utilizes the custom `combined_loss` function.
This loss function is designed to address class imbalance by assigning
higher weights to the minority class and focusing the model’s learning
on hard-to-classify examples, while simultaneously aligning optimization
with the F1 score to balance precision and recall. By reducing the loss
contribution from well-classified examples, it is particularly effective
for imbalanced datasets. Its design is based on ideas presented in [Lin
et al. (2017)](https://ieeexplore.ieee.org/document/8237586) as well as
[Bénédict et al. (2021)](https://arxiv.org/pdf/2108.10566).

In detail, the `combined_loss` is defined as

$$
\text{CombinedLoss}(y,\hat y)
= w_{\mathrm{F1}}\bigl(1-\text{SoftF1}(y,\hat y;\,\varepsilon)\bigr)
\;+\;
(1-w_{\mathrm{F1}})\,\text{FL}(y,\hat y;\,\alpha,\gamma),
$$

where (y {0,1}) are true labels and (y ) are predicted probabilities.

The soft F1 is

$$
\text{SoftF1}
= \frac{2\,TP + \varepsilon}{2\,TP + FP + FN + \varepsilon},
$$

with (TP, FP, FN) computed as **soft counts** (sums over probabilities)
and (\> 0) for numerical stability.

The binary focal loss is

$$
\text{FL}(y,\hat y;\alpha,\gamma)
= -\,\alpha_t\,(1-p_t)^\gamma \,\log(p_t),
\qquad
p_t =
\begin{cases}
\hat y, & y=1,\\
1-\hat y, & y=0,
\end{cases}
\qquad
\alpha_t =
\begin{cases}
\alpha, & y=1,\\
1-\alpha, & y=0.
\end{cases}
$$

To balance the *total* weight of positives and negatives in the loss,
pick $\alpha$ such that the summed positive weight equals the summed
negative weight:

$$
\alpha \cdot N_{+} \;=\; (1-\alpha)\cdot N_{-}
\;\;\Longrightarrow\;\;
\boxed{\alpha = \frac{N_{-}}{N_{+} + N_{-}}},
$$

with

$$
N_{+} = \mathrm{len}(\text{matches}), \qquad
N_{-} = \mathrm{len}(\text{left}) \times \mathrm{len}(\text{right}) - \mathrm{len}(\text{matches}).
$$

``` python
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
from neer_match_utilities.training import combined_loss, alpha_balanced
import tensorflow as tf
import os

# Initialize the model

model = DLMatchingModel(
    similarity_map=smap,
    initial_feature_width_scales = 10,
    feature_depths = 4,
    initial_record_width_scale=10,
    record_depth = 8,
)

# Compile the model

## Calculate the alpha parameter given the dimensions of the training data
mismatch_share = 1.0

alpha = alpha_balanced(left_train, right_train, matches_train, mismatch_share)  # ≈ N_neg / (N_pos + N_neg)

## Use expontantial decay for the learning rate

initial_lr = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=5000,
    decay_rate=0.96,
    staircase=True
)

model.compile(
    loss = combined_loss(
        weight_f1=0.5, 
        epsilon=1e-07, 
        alpha=alpha, 
        gamma=0.5
    ),
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule, 
        clipnorm=.5
    )
)

# Fit the model

model.fit(
    left_train, right_train, matches_train,
    epochs=20,
    batch_size=64,
    mismatch_share=mismatch_share,
    shuffle=True,
)
```

## Export the Model

Once the model is trained, it can be exported via the `Model` class. The
`name` parameter specifies the label of the folder in which the model
will be stored.

``` python
from neer_match_utilities.model import Model
from pathlib import Path

Model.save(
    model = model,
    target_directory = Path(__file__).resolve().parent,
    name = 'example_model'
)
```

## Export Performance Statistics

In addition to this, we can evaluate the model on both the training and
test datasets, and export the corresponding performance statistics into
the folder created when exporting the model.

``` python
# Evaluate the model

performance_train= model.evaluate(
    left_train, 
    right_train, 
    matches_train,
    mismatch_share=1.0
)

performance_test = model.evaluate(
    left_test, 
    right_test, 
    matches_test,
    mismatch_share=1.0
)

# Export performance statistics

training.performance_statistics_export(
    model = model,
    model_name = 'example_model',
    target_directory = Path(__file__).resolve().parent,
    evaluation_train = performance_train,
    evaluation_test = performance_test,
)
```

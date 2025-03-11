# Saving a Model


## Prepare the Data

This tutorial illustrates how to correctly store a `neer-match` model.
Before getting to the part that illustrates the saving, we have to
repreat the steps detailed in **Data Preparation for Training**:

``` python
import random
import pandas as pd

from neer_match.similarity_map import SimilarityMap
from neer_match_utilities.panel import SetupData
from neer_match_utilities.prepare import Prepare
from neer_match_utilities.training import Training
from neer_match_utilities.split import split_test_train

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
        "notmissing"
    ],
    "industry" : [
        "levenshtein",
        "jaro_winkler",
        "notmissing"
    ],
    "purpose" : [
        "levenshtein",
        "jaro_winkler",
        "notmissing",
        "token_sort_ratio",
        "token_set_ratio",
        "partial_token_set_ratio",
        "partial_token_sort_ratio",
    ],
    "bs_text" : [
        "levenshtein",
        "jaro_winkler",
        "notmissing",
        "token_sort_ratio",
        "token_set_ratio",
        "partial_token_set_ratio",
        "partial_token_sort_ratio",
    ],
    "found_year" : [
        "notzero",
        "discrete"
    ],
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
    test_ratio = .5,
    validation_ratio = .0
)
```

## Train the Model

Once the data is in the correct structure, training the model follows.
Note that this example utilizes the custom `focal_loss` function. This
loss function is designed to address class imbalance by assigning higher
weights to the minority class and focusing the model’s learning on
hard-to-classify examples. By reducing the loss contribution from
well-classified examples, it is particularly effective for imbalanced
datasets. Its design is based on ideas presented in [Lin et
al. (2017)](https://ieeexplore.ieee.org/document/8237586).

``` python
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
from neer_match_utilities.training import focal_loss
import tensorflow as tf
import os

# Initialize the model

model = DLMatchingModel(
    similarity_map=smap,
    initial_feature_width_scales = 20,
    feature_depths = 8,
    initial_record_width_scale=20,
    record_depth = 8,
)


# Compile the model

model.compile(
    loss = focal_loss(
        alpha=0.75, 
        gamma=10
    ),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)


# Fit the model

model.fit(
    left_train, right_train, matches_train,
    epochs=50,
    batch_size=10,
    mismatch_share=.5,
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

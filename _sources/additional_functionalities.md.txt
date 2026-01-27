# Additional Functionalities


## Name Commonness

Standard string similarity measures treat all values equally — “Smith”
matching “Smith” scores the same as “Ximénez-Fatio” matching
“Ximénez-Fatio”. In practice, matching on rare values is far more
informative than matching on common ones. The `Commonness` class
addresses this by computing frequency-based commonness scores for
specified variables and registering a custom `commonness_score`
similarity function that rewards rare-and-equal matches.

For each variable, the class computes how common each value is in a
reference corpus and appends a `<variable>_commonness` column (values in
\[0, 1\]) to both DataFrames. The custom similarity function scores
pairs as `(1 - |x - y|) * (1 - mean(x, y))`, so identical rare values
score high and identical common values score low.

Commonness scores should be computed *after* data harmonization
(`Prepare.format()`) but *before* training, so that the `_commonness`
columns are available as features.

The `df_left_full` and `df_right_full` parameters define the full
datasets used for frequency estimation. If the training data is
representative of the population, you can pass the training DataFrames
themselves (i.e., `df_left_full=left` and `df_right_full=right`). If a
larger or more complete dataset is available, passing it will yield more
reliable frequency estimates.

``` python
from neer_match_utilities.prepare import Commonness

c = Commonness(
    variable_list=['name', 'surname'],
    df_left=left,
    df_right=right,

    # Reference corpus for frequency estimation
    df_left_full=left_full,       # full left dataset (or left if representative)
    df_right_full=right_full,     # full right dataset (or right if representative)

    commonness_source='both',     # "left" | "right" | "both" — which corpus to use
    scoring='minmax',             # "relative" | "minmax" | "log"
    fill_value=0.0,               # score for unseen values
    preprocess=True,              # normalize strings (strip & lowercase) before counting
)

left, right = c.calculate()
```

After calling `calculate()`, the DataFrames contain new columns (e.g.,
`name_commonness`, `surname_commonness`) that can be included in the
`similarity_map` using the `commonness_score` similarity concept.

## Stop Word Removal with spaCy

The `Prepare` class supports removing stop words from string variables
using a [spaCy](https://spacy.io/) language pipeline. Stop words are
common words (e.g., “the”, “and”, “of”) that carry little meaning for
entity matching and can reduce similarity scores between otherwise
matching records.

To enable stop word removal, pass a spaCy pipeline name to
`spacy_pipeline`. The pipeline provides a language-specific list of stop
words. spaCy models are available in more than 20 languages — see
[spacy.io/models](https://spacy.io/models) for the full list. You can
also specify `additional_stop_words` to remove domain-specific terms
that are frequent but uninformative for matching (e.g., legal forms like
“AG”, “GmbH”). Stop words are only removed when `remove_stop_words=True`
is passed to `prepare.format()`.

Note that if the similarity map includes numeric similarity concepts,
the corresponding columns must have a numeric dtype. The `to_numeric`
argument in `prepare.format()` ensures this by converting the specified
columns, which is useful when numeric data was read as strings (e.g.,
from CSV files).

``` python
prepare = Prepare(
    similarity_map=similarity_map,
    df_left=left,
    df_right=right,
    id_left='company_id',
    id_right='company_id',
    spacy_pipeline='de_core_news_sm',
    additional_stop_words=['AG']
)

# Get formatted and harmonized datasets

left, right = prepare.format(
    fill_numeric_na=False,
    to_numeric=['found_year'],
    fill_string_na=True,
    capitalize=True,
    lower_case=False,
    remove_stop_words=True,
)
```

## Feature Selection

The `FeatureSelector` allows you to start with a large similarity map —
many feature pairs and many similarity concepts per pair — to maximize
potential performance, and then automatically reduce it to the most
informative subset. It uses a two-stage procedure:

1.  **Stage 1 — Correlation filtering** (optional): Groups highly
    correlated features and keeps only the one most correlated with the
    target variable. This removes redundant features before
    regularization.
2.  **Stage 2 — Elastic net regularization**: Fits a penalized logistic
    regression (L1/L2 mix) with cross-validation to select features that
    contribute unique predictive information. Features with zero or
    near-zero coefficients are dropped.

The selector is designed for the extreme class imbalance typical in
entity matching, where true matches are rare relative to non-matches.

``` python
from neer_match_utilities.feature_selection import FeatureSelector

fs = FeatureSelector(
    similarity_map=similarity_map,
    training_data=(left_train, right_train, matches_train),

    # ID and match columns
    id_left_col="id_unique",
    id_right_col="id_unique",
    matches_id_left="left",
    matches_id_right="right",
    match_col="match",
    matches_are_indices=True,

    # Stage 1: Correlation filtering
    max_correlation=0.95,       # drop features with pairwise correlation > 0.95

    # Stage 2: Elastic net
    scoring="average_precision", # metric for CV; recommended for imbalanced data
    cv=2,                        # number of cross-validation folds
    Cs=20,                       # number of regularization strengths to search
    class_weight="balanced",     # adjust for class imbalance
    min_coef_threshold=0.01,     # drop features with |coefficient| below this

    random_state=42,
    n_jobs=4,
)

fs_result = fs.execute()
```

The result object contains the reduced similarity map, which can be used
directly in subsequent training:

``` python
# Updated similarity map with only selected features
similarity_map = fs_result.updated_similarity_map

# Inspect feature importance via coefficients
print(fs_result.coef_by_feature)

# Check selection metadata
print(f"Features: {fs_result.meta['n_features_in']} → {fs_result.meta['n_features_selected']}")
```

## Handling Many-to-Many Matches

### Loading the Data

First, we import the necessary libraries and load the datasets.

``` python
import random
import pandas as pd

matches = pd.read_csv('matches.csv')
left = pd.read_csv('left.csv')
right = pd.read_csv('right.csv')
```

### Inspecting the Data

The first few rows of each dataset show their structure.

``` python
matches.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | company_id_left | company_id_right |
|-----|-----------------|------------------|
| 0   | 1e87fc75b4      | 0008e07878       |
| 1   | 810c9c3435      | 8bf51ba8a0       |
| 2   | 571dfb67e2      | 90b6db7ed3       |
| 3   | d67d97da08      | b0c68f1152       |
| 4   | 22ac99ae20      | e9823a3073       |

</div>

``` python
left.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | company_id | oai_identifier | company_name | company_info_1 | company_info_2 | pdf_page_num | found_year | found_date_modified | register_year | register_date_modified | ... | effect_year | item_rank | purpose | city | bs_text | sboard_text | proc_text | capital_text | volume | industry |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 1e87fc75b4 | 1006345701_18970010 | Glückauf-, Actien-Gesellschaft für Braunkohlen... | NaN | NaN | 627 | 1871.0 | 1871-08-03 | NaN | NaN | ... | NaN | 1.0 | Abbau von Braunkohlenlagern u. Brikettfabrikat... | Lichtenau | Grundst cke M Grubenwert M Schachtanlagen M Ge... | sichtsrat Vors Buchh ndler Abel Dietzel Gumper... | NaN | M 660 000 in 386 Priorit tsaktien M 1 500 | 1 | Bergwerke, Hütten- und Salinenwesen. |
| 1 | 810c9c3435 | 1006345701_189900031 | Deutsch-Oesterreichische Mannesmannröhren-Werke | in Berlin W. u. Düsseldorf mit Zweigniederlass... | NaN | 501 | 1890.0 | 1890-07-16 | NaN | NaN | ... | NaN | 1.0 | Betrieb der Mannesmannröhren-Walzwerke in Rems... | Berlin | Generaldirektion D sseldorf Mobiliar u Utensil... | Vors Direktor Max Steinthal Stellv Karl v d He... | Dr M Fuchs A Krusche Berlin G Hethey N Eich | M 25 900 000 in 23 875 Inhaber Aktien Lit | 3 | Bergwerke, Hütten- und Salinenwesen. |
| 2 | 571dfb67e2 | 1006345701_191900231 | Handwerkerbank Spaichingen, Akt.-Ges. in Spaic... | NaN | NaN | 345 | 1889.0 | 1889-11-24 | NaN | NaN | ... | NaN | 1.0 | Betrieb von Bank- und Kommissionsgeschäften in... | Spaichingen | Forderung an Aktion re Immobil Gerichtskosten ... | Vors Wilh Lobmiller Stellv Franz Xav Schmid Sa... | NaN | M 600 000 in 600 Aktien M 1000 Urspr M | 23 | Kredit-Banken und andere Geld-Institute. |
| 3 | d67d97da08 | 1006345701_191300172 | Vorschuss-Anstalt für Malchin A.-G. | NaN | Letzte Statutänd. 10./7. 1900. Kapital: M. 900... | 165 | NaN | NaN | NaN | NaN | ... | NaN | NaN | NaN | A | Forder Effekten u Hypoth Debit Bankguth Kassa ... | W Deutler E Buhr W Fehlow | NaN | NaN | 17 | Geld-Institute etc. |
| 4 | 22ac99ae20 | 1006345701_191200161 | Kaisersteinbruch-Actiengesellschaft in Liqu. i... | NaN | NaN | 1443 | 1900.0 | 1900-03-17 | 1900.0 | 1900-04-11 | ... | 1900.0 | 1.0 | Betrieb von Steinhauereien u. aller mit dem Ba... | Köln | Steinbr che Steinhauerei Immobil Mannheim Mobi... | Vors Dr jur P Stephan Rheinbreitbach b Unkel S... | NaN | M 450 000 in 150abgest Vorz Aktien u 300 doppelt | 16 | Industrie der Steine und Erden. |

<p>5 rows × 21 columns</p>
</div>

``` python
right.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | company_id | oai_identifier | company_name | company_info_1 | company_info_2 | pdf_page_num | found_year | found_date_modified | register_year | register_date_modified | ... | effect_year | item_rank | purpose | city | bs_text | sboard_text | proc_text | capital_text | volume | industry |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 0008e07878 | 1006345701_189800021 | „Glückauf-', Act.-Ges. für Braunkohlen-Verwert... | NaN | NaN | 1038 | 1871.0 | 1871-08-03 | NaN | NaN | ... | NaN | 1.0 | Abbau von Braunkohlenlagern u. Brikettfabrikat... | NaN | Grundst cke M Grubenwert M Schachtanlagen M Ge... | Vors Buchh ndler Abel Dietzel Gumpert Lehmann ... | NaN | M 660 000 in 386 Vorzugsaktien M 1500 14 Aktien | 2 | Nachtrag. |
| 1 | 8bf51ba8a0 | 1006345701_189900032 | Deutsch-Oesterreichische Mannesmannröhren-Werke. | Sitz in Berlin, Generaldirektion in Düsseldorf... | NaN | 222 | 1890.0 | 1890-07-16 | NaN | NaN | ... | NaN | 1.0 | Betrieb der Mannesmannröhren-Walzwerke in Rems... | Berlin | Generaldirektion Grundst ckskonto M Mobilien U... | Vors Bankdirektor Max Steinthal Stellv Bankdir... | Dr M Fuchs A Krusche Berlin G Hethey N Eich | M 25 900 000 in 23 875 Inhaber Aktien Lit | 3 | Bergwerke, Hütten- und Salinenwesen. |
| 2 | 90b6db7ed3 | 1006345701_191900232 | Handwerkerbank Spaichingen, Akt.-Ges. in Spaic... | (in Liquidation). | NaN | 168 | 1889.0 | 1889-11-24 | NaN | NaN | ... | NaN | 1.0 | Betrieb von Bank- und Kommissionsgeschäften in... | Spaichingen | NaN | Vors Wilh Lobmiller Stellv Frans Nav Schmid Sa... | NaN | M 600 000 in 600 Aktien M 1000 Urspr M | 23 | Geld-Institute etc. |
| 3 | b0c68f1152 | 1006345701_191400182 | %% für Malchin A.-G. in Malchin. | (In Liquidation.) Letzte Statutänd. 10./7. 190... | NaN | 193 | NaN | NaN | NaN | NaN | ... | NaN | NaN | NaN | Malchin | Forder Effekten u Hypoth Debit Bankguth Kassa ... | W Deutler E Buhr W Fehlow | NaN | NaN | 18 | Kredit-Banken und andere Geld-Institute. |
| 4 | e9823a3073 | 1006345701_190700112 | Kaisersteinbruch-Actiengesellschaft in Köln, | Zweiggeschäfte in Berlin u. Hamburg. | NaN | 818 | 1900.0 | 1900-03-17 | 1900.0 | 1900-04-11 | ... | 1900.0 | 1.0 | Betrieb von Steinhauereien u. aller mit dem Ba... | Köln | Steinbr che Steinhauerei Grundst ck Mannheim M... | Vors Rechtsanw Dr jur Max Liertz Stellv Stadtb... | NaN | M 900 000 in 900 Aktien wovon 600 abgest M | 11 | Industrie der Steine und Erden. |

<p>5 rows × 21 columns</p>
</div>

All three DataFrames have the same number of observations:

``` python
print(f'Number of observations in matches: {len(matches)}')
print(f'Number of observations in left: {len(left)}')
print(f'Number of observations in right: {len(right)}')
```

    Number of observations in matches: 692
    Number of observations in left: 692
    Number of observations in right: 692

### Simulating a Many-to-Many Relationship

To demonstrate how to handle more complex matching scenarios, we
simulate a many-to-many (m:m) relationship. Assume that the company with
`company_id` *1e87fc75b4* in the *left* DataFrame should match with two
entries in the *right* DataFrame: the original match *0008e07878* and an
additional match *8bf51ba8a0*.

``` python
# Add an extra match to simulate a many-to-many relationship

extra_match = pd.DataFrame({
    'company_id_left' : ['1e87fc75b4'],
    'company_id_right' : ['8bf51ba8a0']
})
matches = pd.concat([matches, extra_match], ignore_index=True)
```

Now, inspect the modified `matches` dataframe for the affected IDs:

``` python
matches[
    matches['company_id_left'].isin(['1e87fc75b4', '810c9c3435']) |
    matches['company_id_right'].isin(['0008e07878', '8bf51ba8a0'])
]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | company_id_left | company_id_right |
|-----|-----------------|------------------|
| 0   | 1e87fc75b4      | 0008e07878       |
| 1   | 810c9c3435      | 8bf51ba8a0       |
| 692 | 1e87fc75b4      | 8bf51ba8a0       |

</div>

### Understanding the Matching Issue

Simply adding a new row to the *matches* DataFrame can be problematic.
Consider this simplified example:

| Left | Right | Implied Real-World Entity |
|------|-------|---------------------------|
| A    | C     | Entity 1                  |
| B    | D     | Entity 2                  |

If further evidence shows that record *A* and record *C* represent the
same entity, then all related records (*A*, *B*, *C*, *D*) should be
grouped together. This grouping implies that every possible pair among
these records should be represented (as shown in the first six rows of
the table below). The observations *B* and *C* would consequently appear
in both the *Left* and *Right* columns, so the *left* and *right*
DataFrames need to be adjusted to include these observations in both.
The *matches* DataFrame must be expanded with additional entries
(highlighted by the <span style="color: orange;">orange</span> rows):

| Left | Right |
|----|----|
| A | B |
| A | C |
| A | D |
| B | C |
| B | D |
| C | D |
| <span style="color: orange;">B</span> | <span style="color: orange;">B</span> |
| <span style="color: orange;">C</span> | <span style="color: orange;">C</span> |

This example highlights why a naive approach of merely adding an extra
match does not fully capture the nature of the linking problem.

### Correcting the Relationships

To correctly group all records representing the same real-world entity,
we use the `data_preparation_cs` method from the `SetupData` class. This
method automatically completes the matching pairs and adjusts the `left`
and `right` datasets accordingly.

``` python
from neer_match_utilities.panel import SetupData

left, right, matches = SetupData(matches=matches).data_preparation_cs(
    df_left=left,
    df_right=right,
    unique_id='company_id'
)
```

### 6. Verifying the Adjustments

Finally, we verify that the adjustments correctly reflect the intended
relationships by checking the relevant company IDs in the updated
datasets.

``` python
# Verify the updated matches for the specific company_ids

artificial_group = ['1e87fc75b4', '810c9c3435', '0008e07878', '8bf51ba8a0']

matches_subset = matches[
    matches['left'].isin(artificial_group) |
    matches['right'].isin(artificial_group)
].sort_values(['left', 'right'])

matches_subset
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | left       | right      |
|-----|------------|------------|
| 0   | 0008e07878 | 1e87fc75b4 |
| 1   | 0008e07878 | 810c9c3435 |
| 2   | 0008e07878 | 8bf51ba8a0 |
| 696 | 1e87fc75b4 | 1e87fc75b4 |
| 183 | 1e87fc75b4 | 810c9c3435 |
| 184 | 1e87fc75b4 | 8bf51ba8a0 |
| 705 | 810c9c3435 | 810c9c3435 |
| 524 | 810c9c3435 | 8bf51ba8a0 |

</div>

``` python
# Check the corresponding records in the left dataset

left_subset = left[
    left['company_id'].isin(artificial_group)
][['company_id']]
left_subset.head(10)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | company_id |
|-----|------------|
| 0   | 0008e07878 |
| 181 | 1e87fc75b4 |
| 521 | 810c9c3435 |

</div>

``` python
# Check the corresponding records in the right dataset

right_subset = right[
    right['company_id'].isin(artificial_group)
][['company_id']]
right_subset
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | company_id |
|-----|------------|
| 19  | 1e87fc75b4 |
| 191 | 810c9c3435 |
| 223 | 8bf51ba8a0 |

</div>

These steps ensure that the data accurately represents the underlying
real-world relationships, even when the matching is more complex than a
simple 1:1 mapping. To prevent the simulated change from affecting
subsequent steps, we drop the observations associated with these IDs.

``` python
left = left[~left['company_id'].isin(artificial_group)].reset_index(drop=False)
right = right[~right['company_id'].isin(artificial_group)].reset_index(drop=False)
matches = matches[
    (~matches['left'].isin(artificial_group))
    &
    (~matches['right'].isin(artificial_group))
].reset_index(drop=False)
```

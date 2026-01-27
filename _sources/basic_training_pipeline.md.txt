# A Minimal Training Pipeline


This document explains the data preparation process for training our
matching model. The example data comes from a research project that
digitized historic records of German joint-stock companies [(Gram et
al. 2022)](https://dl.acm.org/doi/10.1145/3531533). The data contains
inconsistencies in spelling, primarily due to variations in abbreviation
conventions and OCR errors, across most variables. These challenges make
it a compelling real-world use case for entity matching.

The data consists of three files:

- *left.csv*
- *right.csv*
- *matches.csv*

## Loading the Data

Training the pipelines requires three datasets:

- `left` (observations from one source or period)
- `right` (observations from another source or period)
- `matches` (a dataframe where each row contains the unique IDs of
  matching entities from `left` and `right`)

``` python
import random
import pandas as pd

matches = pd.read_csv('matches.csv')
left = pd.read_csv('left.csv')
right = pd.read_csv('right.csv')
```

Preview of the matches data:

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

Preview of the left dataset:

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

Preview of the right dataset:

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

## Defining Features and Similarity Concepts

The `similarity_map` defines which similarity concepts (values) to apply
to each feature pair (keys). Note that this example uses a minimal
similarity map for simplicity rather than optimal performance.

``` python
from neer_match.similarity_map import SimilarityMap
from neer_match_utilities.custom_similarities import CustomSimilarities

CustomSimilarities() # Ensures Similarity concepts are always scaled between 0 and 1.

# Define similarity_map

similarity_map = {
    "company_name" : [
        "levenshtein",
        "jaro_winkler",
        "partial_token_sort_ratio",
    ],
    "city" : [
        "levenshtein",
    ],
    "industry" : [
        "levenshtein",
        "jaro_winkler",
        "notmissing",
    ],
}

smap = SimilarityMap(similarity_map)
```

## Harmonizing the data

### Left and Right

Next, data formatting can be harmonized using the `Prepare` class. This
class offers flexible arguments for operations such as capitalizing
strings, converting values to numeric types, and filling missing values.
Additionally, a spaCy pipeline and custom stop words can be specified to
remove noise from string variables (see [additional
functionalities](additional_functionalities.md)). All operations are
applied consistently to both the *left* and *right* DataFrames.

``` python
from neer_match_utilities.prepare import Prepare

# Initialize the Prepare object

prepare = Prepare(
    similarity_map=similarity_map, 
    df_left=left, 
    df_right=right, 
    id_left='company_id', 
    id_right='company_id',
)

# Get formatted and harmonized datasets

left, right = prepare.format(
    fill_numeric_na=False,
    to_numeric=['found_year'],
    fill_string_na=True, 
    capitalize=True,
    lower_case=False,
)
```

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

|  | company_id | company_name | city | industry |
|----|----|----|----|----|
| 0 | 1e87fc75b4 | GLÜCKAUF-, ACTIEN-GESELLSCHAFT FÜR BRAUNKOHLEN... | LICHTENAU | BERGWERKE, HÜTTEN- UND SALINENWESEN. |
| 1 | 810c9c3435 | DEUTSCH-OESTERREICHISCHE MANNESMANNRÖHREN-WERKE | BERLIN | BERGWERKE, HÜTTEN- UND SALINENWESEN. |
| 2 | 571dfb67e2 | HANDWERKERBANK SPAICHINGEN, AKT.-GES. IN SPAIC... | SPAICHINGEN | KREDIT-BANKEN UND ANDERE GELD-INSTITUTE. |
| 3 | d67d97da08 | VORSCHUSS-ANSTALT FÜR MALCHIN A.-G. | A | GELD-INSTITUTE ETC. |
| 4 | 22ac99ae20 | KAISERSTEINBRUCH-ACTIENGESELLSCHAFT IN LIQU. I... | KÖLN | INDUSTRIE DER STEINE UND ERDEN. |

</div>

## Re-Structuring the `Matches` dataframe

`neer-match` requires that the *matches* DataFrame be structured with
the indices from the left and right datasets instead of their unique
IDs. To convert your *matches* DataFrame into the required format, you
can run:

``` python
from neer_match_utilities.training import Training

training = Training(
    similarity_map=similarity_map, 
    df_left=left, 
    df_right=right, 
    id_left='company_id', 
    id_right='company_id',
)

matches = training.matches_reorder(
    matches, 
    matches_id_left='company_id_left', 
    matches_id_right='company_id_right'
)

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

|     | left | right |
|-----|------|-------|
| 0   | 0    | 0     |
| 1   | 1    | 1     |
| 2   | 2    | 2     |
| 3   | 3    | 3     |
| 4   | 4    | 4     |

</div>

## Splitting Data

Subsequently, we need to split the data into training and test sets,
each consisting of three DataFrames. The training ratio is given by
$\text{training_ratio} = 1 - (\text{test_ratio} + \text{validation_ratio})$.
Note that since validation is not implemented yet, you can set
$\text{validation_ratio} = 0$.

``` python
from neer_match_utilities.split import split_test_train

left_train, right_train, matches_train, left_validation, right_validation, matches_validation, left_test, right_test, matches_test = split_test_train(
    left = left,
    right = right,
    matches = matches,
    test_ratio = .5,
    validation_ratio = .0
)
```

## Training and Exporting the Model

For this tutorial, we use a simple Logit model. Other models (ANN,
Probit, or GradientBoost) follow a similar syntax and are covered in
[alternative models](alternative_models.md).

``` python
from neer_match_utilities.baseline_training import BaselineTrainingPipe
import pandas as pd
import os

training_pipeline = BaselineTrainingPipe(
    model_name='demonstration_model',
    similarity_map=smap,
    training_data=(left_train, right_train, matches_train),
    validation_data=(left_validation, right_validation, matches_validation),  # only needed if tune_threshold for GB
    testing_data=(left_test, right_test, matches_test),
    id_left_col="company_id",
    id_right_col="company_id",
    # matches_id_left="left",
    # matches_id_right="right",
    model_kind="logit", # "logit" | "probit" | "gb"
    mismatch_share_fit=1.0,
    # tune_threshold=False, # recommended for "gb"
    # tune_metric="mcc",
)

training_pipeline.execute()
```

    Performance metrics saved to /Users/marli453/develop/py-neer-utilities/docs/source/_static/examples/demonstration_model/performance.csv
    Similarity map saved to /Users/marli453/develop/py-neer-utilities/docs/source/_static/examples/demonstration_model/similarity_map.dill
    Baseline model saved to /Users/marli453/develop/py-neer-utilities/docs/source/_static/examples/demonstration_model/model

    LogitMatchingModel(result=<statsmodels.discrete.discrete_model.BinaryResultsWrapper object at 0x114e6a5d0>, feature_cols=['col_city_city_levenshtein', 'col_company_name_company_name_jaro_winkler', 'col_company_name_company_name_levenshtein', 'col_company_name_company_name_partial_token_sort_ratio', 'col_industry_industry_jaro_winkler', 'col_industry_industry_levenshtein'])

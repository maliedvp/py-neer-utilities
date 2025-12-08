# Loading a Model


## Load Data & Model

Assume you have trained a model and stored it under
`demonstration_model`. We can now load the model and use it to make
predictions using *left* and *right* DataFrames. In the first step, both
the model and the data are loaded into memory.

``` python
import pandas as pd

from neer_match_utilities.model import Model
from neer_match_utilities.prepare import Prepare, similarity_map_to_dict
from pathlib import Path
from neer_match_utilities.custom_similarities import CustomSimilarities

# Load files

left = pd.read_csv('left.csv')
right = pd.read_csv('right.csv')

# Load custom similarity functions

CustomSimilarities()

# Load model

loaded_model = Model.load(
    'demonstration_model'
)
```

## Harmonize Format

Next, we must ensure that the formatting logic remains consistent with
that applied before training. Note that it is not necessary to redefine
the similarity map, as it was stored and is loaded along with the model.

``` python
prepare = Prepare(
    similarity_map=similarity_map_to_dict(
        loaded_model.similarity_map
    ), 
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
```

## Make Suggestions

Now we can make suggestions:

``` python
# Make suggestions for the first observation in left

suggestions = loaded_model.suggest(
    left[:2], 
    right, 
    count=10, 
    verbose=0
)

suggestions
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

|     | left | right | prediction |
|-----|------|-------|------------|
| 0   | 0    | 0     | 0.458794   |
| 1   | 0    | 263   | 0.031737   |
| 2   | 0    | 530   | 0.007070   |
| 3   | 0    | 336   | 0.001577   |
| 4   | 0    | 602   | 0.001301   |
| 5   | 0    | 381   | 0.001137   |
| 6   | 0    | 436   | 0.001031   |
| 7   | 0    | 169   | 0.000992   |
| 8   | 0    | 517   | 0.000974   |
| 9   | 0    | 633   | 0.000819   |
| 10  | 1    | 1     | 0.999379   |
| 11  | 1    | 169   | 0.997775   |
| 12  | 1    | 245   | 0.949566   |
| 13  | 1    | 506   | 0.620147   |
| 14  | 1    | 279   | 0.372858   |
| 15  | 1    | 244   | 0.089429   |
| 16  | 1    | 207   | 0.046766   |
| 17  | 1    | 338   | 0.034049   |
| 18  | 1    | 607   | 0.027497   |
| 19  | 1    | 223   | 0.019611   |

</div>

Based on this output, we can assess whether the suggestion is correct.

``` python
left.iloc[1]
```

    company_id                                                    810c9c3435
    company_name             DEUTSCH-OESTERREICHISCHE MANNESMANNRÖHREN-WERKE
    city                                                              BERLIN
    industry                            BERGWERKE, HÜTTEN- UND SALINENWESEN.
    purpose                BETRIEB DER MANNESMANNRÖHREN-WALZWERKE IN REMS...
    bs_text                GENERALDIREKTION D SSELDORF MOBILIAR U UTENSIL...
    found_year                                                        1890.0
    found_date_modified                                           1890-07-16
    Name: 1, dtype: object

``` python
right.iloc[1]
```

    company_id                                                    8bf51ba8a0
    company_name            DEUTSCH-OESTERREICHISCHE MANNESMANNRÖHREN-WERKE.
    city                                                              BERLIN
    industry                            BERGWERKE, HÜTTEN- UND SALINENWESEN.
    purpose                BETRIEB DER MANNESMANNRÖHREN-WALZWERKE IN REMS...
    bs_text                GENERALDIREKTION GRUNDST CKSKONTO M MOBILIEN U...
    found_year                                                        1890.0
    found_date_modified                                           1890-07-16
    Name: 1, dtype: object

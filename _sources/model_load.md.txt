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

# Load files

left = pd.read_csv('left.csv')
right = pd.read_csv('right.csv')

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
    left[:1], 
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
| 0   | 0    | 0     | 0.473703   |
| 411 | 0    | 411   | 0.381956   |
| 675 | 0    | 675   | 0.362005   |
| 256 | 0    | 256   | 0.347396   |
| 497 | 0    | 497   | 0.345034   |
| 439 | 0    | 439   | 0.341827   |
| 132 | 0    | 132   | 0.323066   |
| 529 | 0    | 529   | 0.322083   |
| 181 | 0    | 181   | 0.319886   |
| 633 | 0    | 633   | 0.319725   |

</div>

Based on this output, we can assess whether the suggestion is correct.

``` python
left.iloc[0]
```

    company_id                                             1e87fc75b4
    company_name    GLÜCKAUF-, ACTIEN-GESELLSCHAFT FÜR BRAUNKOHLEN...
    city                                                    LICHTENAU
    industry                     BERGWERKE, HÜTTEN- UND SALINENWESEN.
    purpose         ABBAU VON BRAUNKOHLENLAGERN U. BRIKETTFABRIKAT...
    bs_text         GRUNDST CKE M GRUBENWERT M SCHACHTANLAGEN M GE...
    found_year                                                 1871.0
    Name: 0, dtype: object

``` python
right.iloc[
    suggestions.loc[0, 'right']
]
```

    company_id                                             0008e07878
    company_name    „GLÜCKAUF-', ACT.-GES. FÜR BRAUNKOHLEN-VERWERT...
    city                                                             
    industry                                                NACHTRAG.
    purpose         ABBAU VON BRAUNKOHLENLAGERN U. BRIKETTFABRIKAT...
    bs_text         GRUNDST CKE M GRUBENWERT M SCHACHTANLAGEN M GE...
    found_year                                                 1871.0
    Name: 0, dtype: object

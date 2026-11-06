# Creating a Common Identifier (ID)


## Between Two Sources

### 1. Load Data & Model

Use the *demonstration_model* to generate a common ID for observations
from the *left* and *right* DataFrames. To do so, we first have to load
both the model and the data into memory.

``` python
from neer_match_utilities.prepare import Prepare
from neer_match_utilities.custom_similarities import CustomSimilarities
from neer_match_utilities.baseline_io import ModelBaseline

import pandas as pd
from pathlib import Path

# Load custom similarity functions

CustomSimilarities()

# Load model (and the similarity map used during training)

loaded_model = ModelBaseline.load(
    'demonstration_model'
)

# Load files

left = pd.read_csv('left.csv')
right = pd.read_csv('right.csv')
```

### 2. Harmonize Format

After loading the model and data, ensure that the data formatting
remains consistent with the preprocessing used during training. The
`Prepare` class harmonizes the left and right DataFrames. Note that the
`similarity_map` is automatically loaded with the model, so there is no
need to redefine it.

``` python
from neer_match_utilities.prepare import Prepare

prepare = Prepare(
    similarity_map=loaded_model.similarity_map, 
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

### 3. Generate a Common ID

The `GenerateID` class creates a common identifier across multiple
repeated cross sections. Creating an ID for observations in the *left*
and *right* datasets can be seen as a special case with two periods.

Key parameters:

- `relation`: Specifies the relationship type between observations
  (`1:1`, `1:m`, `m:1`, or `m:m`)
- `panel_var`: Name of the variable that stores the common identifiers
- `time_var`: Indicates the different cross sections (e.g., *year* for
  annual data)
- `subgroups`: Implements a blocking strategy by restricting comparisons
  to observations within each subgroup, which can significantly reduce
  computation time

To generate an ID for the *left* and *right* DataFrames, we first create
a column (*side*) to distinguish the two sources, then stack the
DataFrames vertically.

``` python
from neer_match_utilities.panel import GenerateID

left['side'] = 'left'
right['side'] = 'right'

df = pd.concat(
    [
        left,
        right
    ],
    axis=0,
    ignore_index=True
)

# Create GenerateID instance

id_generator = GenerateID(
    df_panel=df,
    panel_var='panel_id',
    time_var='side',
    model=loaded_model,
    prediction_threshold=0.5,
    subgroups=[],
    relation='m:m',
)

# Execute the ID generation

result = id_generator.execute()
result.head()
```

    Processing periods left-right at 2026-11-06 13:58:33.032651

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

|     | index | panel_id                             |
|-----|-------|--------------------------------------|
| 0   | 0     | 1b98893c-564f-46f1-8404-59dbb1febe26 |
| 1   | 1     | 170a8b74-3312-4f6e-a9ea-0d4edcdcebd7 |
| 2   | 2     | 56a11796-34bb-407c-ab7a-21bdbaa8f531 |
| 3   | 3     | e5f0b448-18f0-4572-83ae-e4a0a28cb248 |
| 4   | 4     | 97b389fd-aa8d-43d6-9d75-3da1aae8cbaf |

</div>

### 4. Merge Results

``` python
df = pd.merge(
    df,
    result,
    left_index=True,
    right_on='index',
    validate='1:1'
)

df = df.sort_values(['panel_id', 'side', 'company_id']).reset_index(drop=True)

# Prepare selection to be viewed

selected_ids = ['22ac99ae20', 'e9823a3073']
columns_to_show = [
    'panel_id',
    'company_id',
    'side',
    'company_name',
    'city',
]

df_selection= df[df['company_id'].isin(selected_ids)][columns_to_show]

df_selection
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

|  | panel_id | company_id | side | company_name | city |
|----|----|----|----|----|----|
| 800 | 97b389fd-aa8d-43d6-9d75-3da1aae8cbaf | 22ac99ae20 | left | KAISERSTEINBRUCH-ACTIENGESELLSCHAFT IN LIQU. I... | KÖLN |
| 801 | 97b389fd-aa8d-43d6-9d75-3da1aae8cbaf | e9823a3073 | right | KAISERSTEINBRUCH-ACTIENGESELLSCHAFT IN KÖLN, | KÖLN |

</div>

## Repeated Cross-Sections (Panel ID)

The approach demonstrated above generalizes to multiple repeated cross
sections, not just the two sources (*left* and *right*) shown here. The
same logic applies regardless of the number of cross sections (i.e.,
periods), enabling consistent ID generation across your entire dataset.

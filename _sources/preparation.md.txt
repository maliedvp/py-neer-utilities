# Data Preparation for Training


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

## Different Data Relationships

### 1. Loading the Data

First, we import the necessary libraries and load the datasets.

``` python
import random
import pandas as pd

matches = pd.read_csv('matches.csv')
left = pd.read_csv('left.csv')
right = pd.read_csv('right.csv')
```

### 2. Inspecting the Data

Let’s view the first few rows of each dataset to understand their
structure.

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

### 3. Simulating a Many-to-Many Relationship

To demonstrate how to handle more complex matching scenarios, we
simulate a many-to-many (m:m) relationship. For instance, assume that
the company with `company_id` *1e87fc75b4* in the *left* DataFrame
should match with two entries in the *right* DataFrame: the original
match *0008e07878* and an additional match *8bf51ba8a0*.

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

### 4. Understanding the Matching Issue

Simply adding a new row to the *matches* DataFrame can be problematic.
Consider this simplified example:

| Left | Right | Implied Real-World Entity |
|------|-------|---------------------------|
| A    | C     | Entity 1                  |
| B    | D     | Entity 2                  |

If further evidence shows that record *A* and record *C* represent the
same entity, then all related records (*A*, *B*, *C*, *D*) should be
grouped together. This comprehensive grouping implies that every
possible pair among these records should be represented (as shown in the
first six rows of the table below). Notice that the observations *B* and
*C* would consequently appear in both the *Left* and *Right* columns.
Therefore, the *left* and *right* DataFrames need to be adjusted,
ensuring these observations will be included in both of these
DataFrames. As a result, the *matches* DataFrame must be expanded with
an additional set of corresponding entries (highlighted by the
<span style="color: orange;">orange</span> rows):

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

This example highlights why a naive approach (merely adding an extra
match) does not fully capture the nature of the linking problem.

### 5. Correcting the Relationships

To resolve this issue and correctly group all records representing the
same real-world entity, we use the `data_preparation_cs` method from the
`SetupData` class in the `neer_match_utilities.panel` module. This
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

By following these steps, we ensure that the data accurately represents
the underlying real-world relationships, even when the matching is more
complex than a simple 1:1 mapping. To not have the manual change affect
the next steps, we drop observations associated with these IDs.

``` python
left = left[~left['company_id'].isin(artificial_group)].reset_index(drop=False)
right = right[~right['company_id'].isin(artificial_group)].reset_index(drop=False)
matches = matches[
    (~matches['left'].isin(artificial_group))
    &
    (~matches['right'].isin(artificial_group))
].reset_index(drop=False)
```

------------------------------------------------------------------------

## Formatting

### 1. A customized `similarity_map`

Set up the `similarity_map`. Note that the columns as `city`,
`industry`, and `purpose` contain missing values. One way to improve the
handling of these is to include a custom [similarity
function](https://github.com/maliedvp/py-neer-match/blob/custom_similarity_functions/src/neer_match/similarity_map.py)
`notmissing` to the `similarity_map` that returns 0 if a least one
observation of a record pair is any missing value (`None`, `np.nan`,
`pd.nan` or and empty string) and 1 otherwise. Similarly, for numeric
columns, the custom function `notzero` is added. These functions are not
part of the released version of
`neer_match.similarity_map.available_similarities()`, which is why they
are outcommented in the example below.

``` python
from neer_match.similarity_map import SimilarityMap

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
        # "notmissing"
    ],
    "industry" : [
        "levenshtein",
        "jaro_winkler",
        # "notmissing"
    ],
    "purpose" : [
        "levenshtein",
        "jaro_winkler",
        # "notmissing",
        "token_sort_ratio",
        "token_set_ratio",
        "partial_token_set_ratio",
        "partial_token_sort_ratio",
    ],
    "bs_text" : [
        "levenshtein",
        "jaro_winkler",
        # "notmissing",
        "token_sort_ratio",
        "token_set_ratio",
        "partial_token_set_ratio",
        "partial_token_sort_ratio",
    ],
    "found_year" : [
        # "notzero",
        "discrete"
    ],
}

smap = SimilarityMap(similarity_map)
```

### 2. Harmonizing the data

Next, data formatting can be harmonized using the `Prepare` class. This
class enables operations such as capitalizing string variables and
converting other values to numeric types. Importantly, these operations
are applied consistently to both the *left* and *right* DataFrames.

``` python
from neer_match_utilities.prepare import Prepare

# Initialize the Prepare object

prepare = Prepare(
    similarity_map=similarity_map, 
    df_left=left, 
    df_right=right, 
    id_left='company_id', 
    id_right='company_id',
    spacy_pipeline='de_core_news_sm',
    additional_stop_words=[]
)

# Get formatted and harmonized datasets

left, right = prepare.format(
    fill_numeric_na=False,
    to_numeric=['found_year'],
    fill_string_na=True, 
    capitalize=True,
    lower_case=False,
    remove_stop_words=False,
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

|  | company_id | company_name | city | industry | purpose | bs_text | found_year |
|----|----|----|----|----|----|----|----|
| 0 | 008fbe2454 | BETRAG IN RÜCKZAHLBAR VERSTÄRKTE | RÜCKZAHLBAR | ELEKTRISCHE STRASSENBAHNEN, KLEIN- UND PFERDEB... |  | BAHNH FE U GRUNDST CKE BAHNBAU U H LEITUNG MOT... | NaN |
| 1 | 00a050af9d | ZUCKERFABRIK HARSUM IN HARSUM, PROV. HANNOVER. | HANNOVER | ZUCKER-FABRIKEN UND ZUCKER-RAFFINERIEN. | FABRIKATION VON ROHZUCKER. PRODUKTION 1896/97–... | GRUNDST CK M GEB UDE M MASCHINEN U APPARATE M | 1873.0 |
| 2 | 00ce66e5a8 | AKTIENGESELLSCHAFT FÜR LINIIR-APPARATE, PATENT... | LEIPZIG | METALL-INDUSTRIE. | ERWERB, AUSBEUTUNG UND SONSTIGE VERWERTUNG DER... | PATENTE BETRIEBSMASCHINEN INVENTAR UTENSIL WAR... | 1899.0 |
| 3 | 00d74f0e0b | DEUTSCHE RÜCK- U. MITVERSICHERUNGS-GESELLSCHAF... | BERLIN | VERSICHERUNGS-GESELLSCHAFTEN ALLER BRANCHEN. |  | AKTIENWECHSEL M EFFEKTEN M HYP DARLEHEN M INVE... | 1885.0 |
| 4 | 01077b1f46 | GEBRÜDER ZSCHILLE TUCHFABRIK | GROSSENHAIN | TEXTIL-INDUSTRIE. | ÜBERNAHME UND FORTBETRIEB DER DER FIRMA GEBR. ... |  | 1899.0 |

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

|  | company_id | company_name | city | industry | purpose | bs_text | found_year |
|----|----|----|----|----|----|----|----|
| 0 | 0562ddb063 | DEUTSCHE GAS-SELBSTZÜNDER-A.-G. IN BERLIN, HOL... | BERLIN | GESELLSCHAFTEN FÜR GAS-, PETROLEUM- UND SPIRIT... | HERSTELLUNG U. VERTRIEB VON GASSELBSTZÜNDERN, ... | KASSA BANKGUTH AUSSENST NDE VORSCHUSS ZAHLUNG ... | 1897.0 |
| 1 | 05cdac2565 | ACT.-GES. FÜR GRUNDBESITZ U. HYPOTHEKENVERKEHR... | BERLIN | BAU-BANKEN, BAU-, TERRAIN- UND IMMOBILIEN-GESE... |  | IMMOBIL I DO II EIG HYPOTH WERTP PFLASTERKAUTI... | 1883.0 |
| 2 | 0a1e2ae043 | KIELER DOCK GESELLSCHAFT J. W. SEIBEL IN KIEL. | KIEL | SCHIFFSBAU-ANSTALTEN UND DOCK-GESELLSCHAFTEN. | ERWERB U. BETRIEB VON SCHWIMMDOCKS. GEDOCKT WU... | DOCKBAU INVENTAR KASSA BANKGUTH SWENTINE DOCK ... | 1876.0 |
| 3 | 0bab236590 | HESSISCHE EISENBAHN-AKTIENGESELLSCHAFT IN DARM... | DARMSTADT | ELEKTRISCHE STRASSENBAHNEN, KLEIN- UND PFERDEB... | ERBAUUNG, ERWERBUNG, PACHTUNG U. BETRIEB VON B... |  | 1912.0 |
| 4 | 1090024903 | \* BEVENSER MASCHINENFABRIK AKT.-GES. IN BEVENSEN | BEVENSEN | MASCHINEN- UND ARMATUREN-FABRIKEN, EISENGIESSE... | ÜBERNAHME U. FORTBETRIEB DES FABRIKATIONS- U. ... |  | 1909.0 |

</div>

## Re-Structuring

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
    id_right='company_id'
)

matches = training.matches_reorder(
    matches, 
    matches_id_left='left', 
    matches_id_right='right'
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
| 0   | 0    | 34    |
| 1   | 1    | 267   |
| 2   | 2    | 141   |
| 3   | 3    | 46    |
| 4   | 4    | 149   |

</div>

Let’s track down the observations from *matches* in *left* .

``` python
left_index = matches.loc[4,'left']
left[left.index==left_index]
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

|  | company_id | company_name | city | industry | purpose | bs_text | found_year |
|----|----|----|----|----|----|----|----|
| 4 | 01077b1f46 | GEBRÜDER ZSCHILLE TUCHFABRIK | GROSSENHAIN | TEXTIL-INDUSTRIE. | ÜBERNAHME UND FORTBETRIEB DER DER FIRMA GEBR. ... |  | 1899.0 |

</div>

and *right*

``` python
right_index = matches.loc[4,'right']
right[right.index==right_index]
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

|  | company_id | company_name | city | industry | purpose | bs_text | found_year |
|----|----|----|----|----|----|----|----|
| 149 | 721ce62f33 | GEBRÜDER ZSCHILLE TUCHFABRIK AKTIENGESELLSCHAF... | GROSSENHAIN | TEXTIL-INDUSTRIE. | ÜBERNAHME UND FORTBETRIEB DER DER FIRMA GEBR. ... | GRUNDST CKE U GEB UDE MASCHINEN U UTENSILIEN F... | 1899.0 |

</div>

## Splitting data

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

``` python
matches_train.head()
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
| 0   | 0    | 13    |
| 1   | 1    | 75    |
| 2   | 2    | 16    |
| 3   | 3    | 111   |
| 4   | 4    | 207   |

</div>

``` python
left_train[
    left_train.index.isin(
        matches_train['left'].head()
    )
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

|  | company_id | company_name | city | industry | purpose | bs_text | found_year | index_original |
|----|----|----|----|----|----|----|----|----|
| 0 | 008fbe2454 | BETRAG IN RÜCKZAHLBAR VERSTÄRKTE | RÜCKZAHLBAR | ELEKTRISCHE STRASSENBAHNEN, KLEIN- UND PFERDEB... |  | BAHNH FE U GRUNDST CKE BAHNBAU U H LEITUNG MOT... | NaN | 0 |
| 1 | 00ce66e5a8 | AKTIENGESELLSCHAFT FÜR LINIIR-APPARATE, PATENT... | LEIPZIG | METALL-INDUSTRIE. | ERWERB, AUSBEUTUNG UND SONSTIGE VERWERTUNG DER... | PATENTE BETRIEBSMASCHINEN INVENTAR UTENSIL WAR... | 1899.0 | 2 |
| 2 | 00d74f0e0b | DEUTSCHE RÜCK- U. MITVERSICHERUNGS-GESELLSCHAF... | BERLIN | VERSICHERUNGS-GESELLSCHAFTEN ALLER BRANCHEN. |  | AKTIENWECHSEL M EFFEKTEN M HYP DARLEHEN M INVE... | 1885.0 | 3 |
| 3 | 0118303343 | CUXHAVENER GAS-ACTIEN-GESELLSCHAFT IN CUXHAVEN. | CUXHAVEN | GASANSTALTEN UND GASGLÜHLICHT-FABRIKEN. |  | ANLAGEKONTO M GRUNDST CK M WOANHAUS M KAUTION ... | 1884.0 | 5 |
| 4 | 022f81a1cb | BERLINER KRONEN-BRAUEREI ACTIENGESELLSCHAFT IN... | BERLIN | BRAUEREIEN. | BIERBRAUEREIBETRIEB. ABSATZ CA. 45 676 HL. KAP... | GRUNDST CKE U MIETSH USER M BRAUEREIGEB UDE M ... | 1891.0 | 7 |

</div>

``` python
right_train[
    right_train.index.isin(
        matches_train['right'].head()
    )
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

|  | company_id | company_name | city | industry | purpose | bs_text | found_year | index_original |
|----|----|----|----|----|----|----|----|----|
| 13 | 2eece3dd52 | BETRAG IN RÜCKZAHLBAR VERSTÄRKTE | RÜCKZAHLBAR | ELEKTRISCHE STRASSENBAHNEN, KLEIN- UND PFERDEB... |  | BAHNH FE U GRUNDST CKE BAHNBAU OBERIRDISCHE LE... | NaN | 34 |
| 16 | 3a50b36f26 | DEUTSCHE RÜCK- U. MITVERSICHERUNGS-GESELLSCHAF... | BERLIN | VERSICHERUNGS-GESELLSCHAFTEN ALLER BRANCHEN. |  | AKTIENWECHSEL M EFFEKTEN M HYPOTHEKENDARLEHEN ... | 1885.0 | 46 |
| 75 | 6ed393b0a9 | A.-G. F. LINIIR-APPARATE, PATENT GROSSE IN LEI... | LEIPZIG | METALL-INDUSTRIE. | ERWERB, AUSBEUTUNG UND SONST. VERWERTUNG DER V... | PATENTE BETRIEBSMASCHINEN INVENTAR UTENSIL WAR... | 1899.0 | 141 |
| 111 | 832a174691 | CUXHAVENER GAS-ACTIEN-GESELLSCHAFT IN CUXHAVEN | CUXHAVEN | GAS-GESELLSCHAFTEN. | BETRIEB EINES GASWERKES, SOWIE VERWERTUNG DER ... |  | 1884.0 | 196 |
| 207 | c2b13c295b | BERLINER KRONEN-BRAUEREI ACTIENGESELLSCHAFT IN... | BERLIN | BRAUEREIEN. | BIERBRAUEREIBETRIEB. BIERABSATZ 1895/96–1897/9... |  | 1891.0 | 417 |

</div>

# Data Preparation for Training


This document explains the data preparation process for training our
matching model. The example data comes from a research project that
digitized historic records of German companies (from the *Handbuch der
Deutschen Aktiengesellschaften*) using OCR. The dataset includes three
files:

- **left.csv**
- **right.csv**
- **matches.csv**

Initially, the data has a one-to-one relationship: each record in
`left.csv` corresponds to exactly one record in `right.csv`, and vice
versa.

## Different Data Relationship

### 1. Loading the Data

First, we import the necessary libraries and load the datasets.

``` python
import random
import pandas as pd

random.seed(42)

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

Since the original data is based on 1:1 relationships, all three
datasets have the same number of observations:

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
the company with `company_id` **1e87fc75b4** in the left dataset should
match with two entries in the right dataset: the original match
**0008e07878** and an additional match **8bf51ba8a0**.

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

Simply adding a new row to `matches` can be problematic. Consider this
simplified example:

| Left | Right | Implied Real-World Entity |
|------|-------|---------------------------|
| A    | C     | Entity 1                  |
| B    | D     | Entity 2                  |

If further evidence shows that record **A** and record **C** represent
the same entity, then all related records (A, B, C, D) should be grouped
together. The complete relationship should reflect every possible pair
among these records:

| Left | Right |
|------|-------|
| A    | B     |
| A    | C     |
| A    | D     |
| B    | C     |
| B    | D     |
| C    | D     |

This example highlights why a naive approach (merely adding an extra
match) can distort the true relationships between records.

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
matches_subset = matches[
    matches['left'].isin(['1e87fc75b4', '810c9c3435', '0008e07878', '8bf51ba8a0']) |
    matches['right'].isin(['1e87fc75b4', '810c9c3435', '0008e07878', '8bf51ba8a0'])
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
| 694 | 0008e07878 | 810c9c3435 |
| 695 | 0008e07878 | 8bf51ba8a0 |
| 0   | 1e87fc75b4 | 0008e07878 |
| 693 | 1e87fc75b4 | 810c9c3435 |
| 692 | 1e87fc75b4 | 8bf51ba8a0 |
| 1   | 810c9c3435 | 8bf51ba8a0 |

</div>

``` python
# Check the corresponding records in the left dataset
left_subset = left[
    left['company_id'].isin(['1e87fc75b4', '810c9c3435', '0008e07878', '8bf51ba8a0'])
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
| 0   | 1e87fc75b4 |
| 1   | 810c9c3435 |
| 692 | 0008e07878 |

</div>

``` python
# Check the corresponding records in the right dataset
right_subset = right[
    right['company_id'].isin(['1e87fc75b4', '810c9c3435', '0008e07878', '8bf51ba8a0'])
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
| 0   | 810c9c3435 |
| 19  | 0008e07878 |
| 20  | 8bf51ba8a0 |

</div>

By following these steps, we ensure that the dataset accurately
represents the underlying real-world relationships—even when the
matching is more complex than a simple 1:1 mapping.

------------------------------------------------------------------------

## Number 2

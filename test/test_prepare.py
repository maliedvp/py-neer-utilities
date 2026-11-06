import pytest
import pandas as pd
import numpy as np
from neer_match_utilities.prepare import Prepare

@pytest.fixture
def sample_data():
    """
    Provides sample data for testing the Prepare class.

    Returns:
    --------
    tuple:
        - df_left: pandas.DataFrame, the left DataFrame to be processed.
        - df_right: pandas.DataFrame, the right DataFrame to be processed.
        - similarity_map: dict, a mapping of column pairs between df_left and df_right.
    """
    df_left = pd.DataFrame({
        'name': ['ALICE', 'BOB', None],  # String column with a missing value
        'age': [None, 22, 34],           # Numeric column with a missing value
        'id': [1, 2, 3]                 # ID column
    })

    df_right = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],  # String column
        'Age': ['25', 'N/A', '30'],          # String representation of numeric column with an invalid value ('N/A')
        'id': [1, 2, 4]                      # ID column
    })

    similarity_map = {
        'name~Name': 'string',  # Mapping for name columns
        'age~Age': 'numeric'    # Mapping for age columns
    }

    return df_left, df_right, similarity_map


def test_prepare_format(sample_data):
    """
    Tests the Prepare class's format method with various configurations.

    Verifies:
    ---------
    - Proper handling of NaN values in numeric and string columns.
    - Correct numeric conversion for specified columns.
    - Consistent capitalization of string columns.
    - Alignment of column data types across the two DataFrames.
    """

    # Unpack the sample data
    df_left, df_right, similarity_map = sample_data

    # Initialize the Prepare object
    prepare = Prepare(similarity_map=similarity_map, df_left=df_left, df_right=df_right, id_left='id', id_right='id')

    # Test Case 1: fill_numeric_na=True
    # =================================
    df_left_processed, df_right_processed = prepare.format(
        fill_numeric_na=True, to_numeric=['age', 'Age'], fill_string_na=True, capitalize=True
    )

    # Check that NaN values are filled
    assert df_left_processed.isna().sum().sum() == 0, "NaN values should be filled in df_left"
    assert df_right_processed.isna().sum().sum() == 0, "NaN values should be filled in df_right"

    # Check capitalization and fill_string_na behavior
    assert all(df_left_processed['name'] == ['ALICE', 'BOB', '']), "String values should be capitalized and missing strings filled"
    assert all(df_right_processed['Name'] == ['ALICE', 'BOB', 'CHARLIE']), "String values should match and be capitalized"

    # Check numeric conversion
    assert pd.api.types.is_numeric_dtype(df_left_processed['age']), "df_left 'age' column should be numeric"
    assert pd.api.types.is_numeric_dtype(df_right_processed['Age']), "df_right 'Age' column should be numeric"

    # Verify ID columns remain unchanged
    assert all(df_left_processed['id'] == [1, 2, 3]), "df_left 'id' column should remain unchanged"
    assert all(df_right_processed['id'] == [1, 2, 4]), "df_right 'id' column should remain unchanged"

    # Verify dtype alignment for matched columns
    assert df_left_processed['name'].dtype == df_right_processed['Name'].dtype, "String columns should have the same dtype"
    assert df_left_processed['age'].dtype == df_right_processed['Age'].dtype, "Numeric columns should have the same dtype"

    # Test Case 2: fill_numeric_na=False
    # ==================================
    df_left_processed, df_right_processed = prepare.format(
        fill_numeric_na=False, to_numeric=['age', 'Age'], fill_string_na=True, capitalize=True
    )

    # Check numeric conversion
    assert pd.api.types.is_numeric_dtype(df_left_processed['age']), "df_left 'age' column should still be numeric"
    assert pd.api.types.is_numeric_dtype(df_right_processed['Age']), "df_right 'Age' column should still be numeric"

    # Verify dtype alignment for matched columns
    assert df_left_processed['name'].dtype == df_right_processed['Name'].dtype, "String columns should remain aligned"
    assert df_left_processed['age'].dtype == df_right_processed['Age'].dtype, "Numeric columns should remain aligned"

    # Test Case 3: fill_string_na=False
    # =================================
    df_left_processed, df_right_processed = prepare.format(
        fill_numeric_na=False, to_numeric=['age', 'Age'], fill_string_na=False, capitalize=True
    )

    # Print for debugging purposes (optional)
    print(df_left_processed, '\t')
    print(df_right_processed)

    # Check that NaN values remain in string columns when fill_string_na=False
    assert df_left_processed['name'].isna().max() == True, "NaN values should remain in string columns if fill_string_na=False"


def test_to_numeric_fills_missing_numbers():
    """fill_numeric_na fills the missing values of a float column, under pandas 2 and 3 alike."""
    left = pd.DataFrame({"id": [1, 2, 3], "x": [1.5, np.nan, 3.0]})
    right = pd.DataFrame({"id": [1, 2, 3], "x": [np.nan, None, 2.0]})
    prepare = Prepare(similarity_map={"x": ["euclidean"]}, df_left=left, df_right=right, id_left="id", id_right="id")
    filled_left, filled_right = prepare.format(fill_numeric_na=True, to_numeric=["x"])
    assert filled_left["x"].tolist() == [1.5, 0.0, 3.0]
    assert filled_right["x"].tolist() == [0.0, 0.0, 2.0]
    kept_left, kept_right = prepare.format(fill_numeric_na=False, to_numeric=["x"])
    assert kept_left["x"].isna().tolist() == [False, True, False]
    assert kept_right["x"].isna().tolist() == [True, True, False]


def test_to_numeric_keeps_signs_and_exponents():
    """Numbers, and strings that are numbers, keep their sign and exponent."""
    left = pd.DataFrame({"id": [1, 2, 3, 4], "x": [-73.98, 1e-05, 2.5e-07, 12.0]})
    right = pd.DataFrame({"id": [1, 2, 3, 4], "x": ["-73.98", "1e-05", " 4 ", "+3"]})
    prepare = Prepare(similarity_map={"x": ["euclidean"]}, df_left=left, df_right=right, id_left="id", id_right="id")
    out_left, out_right = prepare.format(fill_numeric_na=True, to_numeric=["x"])
    assert out_left["x"].tolist() == [-73.98, 1e-05, 2.5e-07, 12.0]
    assert out_right["x"].tolist() == [-73.98, 1e-05, 4.0, 3.0]


def test_to_numeric_cleans_strings_that_are_no_numbers():
    """Of a string that is no number only the digits and dots count; one with none is missing or 0."""
    left = pd.DataFrame({"id": [1, 2, 3, 4], "x": ["1.5 km", "1200 inhabitants", "N/A", None]})
    right = pd.DataFrame({"id": [1, 2, 3, 4], "x": [1.0, 2.0, 3.0, 4.0]})
    prepare = Prepare(similarity_map={"x": ["euclidean"]}, df_left=left, df_right=right, id_left="id", id_right="id")
    filled, _ = prepare.format(fill_numeric_na=True, to_numeric=["x"])
    assert filled["x"].tolist() == [1.5, 1200.0, 0.0, 0.0]
    kept, _ = prepare.format(fill_numeric_na=False, to_numeric=["x"])
    assert kept["x"].tolist()[:2] == [1.5, 1200.0]
    assert kept["x"].isna().tolist() == [False, False, True, True]

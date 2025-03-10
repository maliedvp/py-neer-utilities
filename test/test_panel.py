import pytest
import pandas as pd
from neer_match_utilities.panel import SetupData


# helper function
def reorder_matches(df, col_left='left', col_right='right'):
    """
    Re-orders the matches DataFrame so that, for each row, the smaller element (or alphabetically first element)
    is in the left column. Finally, the DataFrame is sorted by the left column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the match pairs.
    col_left : str, optional
        Name of the left column (default is 'left').
    col_right : str, optional
        Name of the right column (default is 'right').

    Returns
    -------
    pd.DataFrame
        The re-ordered DataFrame.
    """
    df = df.copy()  # avoid modifying the original DataFrame

    # For each row, sort the two elements and assign the smaller to col_left.
    df[[col_left, col_right]] = df.apply(
        lambda row: pd.Series(sorted([row[col_left], row[col_right]])),
        axis=1
    )
    
    # Sort the DataFrame by the left column
    df = df.sort_values(by=[col_left, col_right]).reset_index(drop=True)
    return df


# Example 1: A simple 1:1 matching case.
@pytest.fixture
def example_1():
    return {
        "df": pd.DataFrame(
            {"id": [1, 2, 3, 4, 5, 6, 7, 8], "col": ["a", "b", "c", "d", "a", "b", "c", "d"]}
        ),
        "matches": [(1, 5), (2, 6), (3, 7), (4, 8)],
        "expected_df_matches": pd.DataFrame(
            {"left": [1, 2, 3, 4], "right": [5, 6, 7, 8]}
        ),
    }


# Example 2: A case where an overlap creates additional match pairs.
@pytest.fixture
def example_2():
    return {
        "df": pd.DataFrame(
            {"id": [1, 2, 3, 7, 5, 6, 7, 8], "col": ["a", "b", "c", "c", "a", "b", "c", "c"]}
        ),
        "matches": [(1, 5), (2, 6), (3, 7), (7, 8)],
        "expected_df_matches": pd.DataFrame(
            {"left": [1, 2, 3, 3, 7, 7], "right": [5, 6, 7, 8, 8, 7]}
        ),
    }


# Example 3: Panel data with an additional panel_id column and no initial matches.
@pytest.fixture
def example_3():
    return {
        "df": pd.DataFrame(
            {
                "pid": [10, 10, 20, 20, 30, 30, 40, 40],
                "id": [1, 2, 3, 4, 5, 6, 7, 8],
                "col": ["a", "a", "b", "b", "c", "c", "d", "d"],
            }
        ),
        "expected_df_matches": pd.DataFrame(
            {"left": [1, 3, 5, 7], "right": [2, 4, 6, 8]}
        ),
    }


# Example 4: Panel data where an initial match creates transitive connections.
@pytest.fixture
def example_4():
    return {
        "df": pd.DataFrame(
            {
                "pid": [10, 10, 20, 20, 30, 30, 40, 40],
                "id": [1, 2, 3, 4, 5, 6, 7, 8],
                "col": ["a", "a", "b", "b", "c", "c", "a*", "a*"],
            }
        ),
        "matches": [(1, 8)],
        "expected_df_matches": pd.DataFrame(
            {
                "left": [1, 1, 1, 2, 2, 3, 7, 5, 2, 7],
                "right": [2, 7, 8, 7, 8, 4, 8, 6, 2, 7],
            }
        ),
    }


def test_example_1(example_1):
    setup_data = SetupData(matches=example_1["matches"])
    df_left, df_right, df_matches = setup_data.data_preparation_panel(
        df_panel=example_1["df"], unique_id="id"
    )

    # Assert the matches are as expected
    pd.testing.assert_frame_equal(
        df_matches.sort_values(by=["left", "right"]).reset_index(drop=True),
        example_1["expected_df_matches"].sort_values(by=["left", "right"]).reset_index(drop=True),
    )

    # Assert all values in 'left' of df_matches exist in df_left['id']
    assert set(df_matches['left']).issubset(df_left['id'])

    # Assert all values in 'right' of df_matches exist in df_right['id']
    assert set(df_matches['right']).issubset(df_right['id'])


def test_example_2(example_2):
    setup_data = SetupData(matches=example_2["matches"])
    df_left, df_right, df_matches = setup_data.data_preparation_panel(
        df_panel=example_2["df"], unique_id="id"
    )

    # Assert number of length of matches
    assert len(example_2["expected_df_matches"]) == len(df_matches) == 6

    # assert identical matches (after re-ordering)
    pd.testing.assert_frame_equal(
        reorder_matches(df_matches),
        reorder_matches(example_2["expected_df_matches"])
    )

    # Assert all values in 'left' of df_matches exist in df_left['id']
    assert set(df_matches['left']).issubset(df_left['id'])

    # Assert all values in 'right' of df_matches exist in df_right['id']
    assert set(df_matches['right']).issubset(df_right['id'])


def test_example_3(example_3):
    setup_data = SetupData()
    df_left, df_right, df_matches = setup_data.data_preparation_panel(
        df_panel=example_3["df"], unique_id="id", panel_id="pid"
    )

    # Assert number of length of matches
    assert len(example_3["expected_df_matches"]) == len(df_matches) == 4

    # assert identical matches (after re-ordering)
    pd.testing.assert_frame_equal(
        reorder_matches(df_matches),
        reorder_matches(example_3["expected_df_matches"])
    )

    # Assert all values in 'left' of df_matches exist in df_left['id']
    assert set(df_matches['left']).issubset(df_left['id'])

    # Assert all values in 'right' of df_matches exist in df_right['id']
    assert set(df_matches['right']).issubset(df_right['id'])


def test_example_4(example_4):
    setup_data = SetupData(matches=example_4["matches"])
    df_left, df_right, df_matches = setup_data.data_preparation_panel(
        df_panel=example_4["df"], unique_id="id", panel_id="pid"
    )

    print(df_matches)

    # Assert number of length of matches
    assert len(example_4["expected_df_matches"]) == len(df_matches) == 10

    # assert identical matches (after re-ordering)
    pd.testing.assert_frame_equal(
        reorder_matches(df_matches),
        reorder_matches(example_4["expected_df_matches"])
    )

    # Assert all values in 'left' of df_matches exist in df_left['id']
    assert set(df_matches['left']).issubset(df_left['id'])

    # Assert all values in 'right' of df_matches exist in df_right['id']
    assert set(df_matches['right']).issubset(df_right['id'])

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


# Example: Complicated nested dependency strucutre
@pytest.fixture
def example_1():
    return {
        "left": pd.DataFrame(
            {"id": [1, 3, 5], "col": ["a", "c", "e"]}
        ),
        "right": pd.DataFrame(
            {"id": [2, 4, 6], "col": ["b", "d", "f"]}
        ),
        "matches": [(1, 2), (2, 3), (3, 4), (5, 6)],
        "expected_df_matches": pd.DataFrame(
            {"left": [1,1,1,2,2,3,5,2,3], "right": [2,3,4,3,4,4,6,2,3]}
        ),
    }



def test_example_1(example_1):
    df_left, df_right, df_matches = SetupData(
        matches=example_1["matches"]
    ).data_preparation_cs(
        df_left=example_1['left'],
        df_right=example_1['right'],
        unique_id='id'
    )

    # Assert the matches are as expected
    pd.testing.assert_frame_equal(
        reorder_matches(df_matches),
        reorder_matches(example_1["expected_df_matches"])
    )

    # Assert all values in 'left' of df_matches exist in df_left['id']
    assert set(df_matches['left']).issubset(df_left['id'])

    # Assert all values in 'right' of df_matches exist in df_right['id']
    assert set(df_matches['right']).issubset(df_right['id'])

    # Assert if one observation in left and right, this must be also in the matches
    all_ids = list(set(df_left['id']).union(set(df_right['id'])))
    lr_list = list(set(df_left['id']).intersection(set(df_right['id'])))
    print('HEEEERE', lr_list)

    for i in all_ids:
        duplicated_matches = len(
            df_matches[
                (df_matches['left']==i)
                &
                (df_matches['right']==i)
            ]
        )

        if i in lr_list:
            assert duplicated_matches==1
        else:
            assert duplicated_matches==0



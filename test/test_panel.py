import pytest
import pandas as pd
from neer_match_utilities.panel import SetupData


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


@pytest.fixture
def example_2():
    return {
        "df": pd.DataFrame(
            {"id": [1, 2, 3, 7, 5, 6, 7, 8], "col": ["a", "b", "c", "c", "a", "b", "c", "c"]}
        ),
        "matches": [(1, 5), (2, 6), (3, 7), (7, 8)],
        "expected_df_matches": pd.DataFrame(
            {"left": [1, 2, 3, 3, 7], "right": [5, 6, 7, 8, 8]}
        ),
    }


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
                "left": [1, 1, 1, 2, 2, 3, 5, 7],
                "right": [8, 2, 7, 7, 8, 4, 6, 8],
            }
        ),
    }


def test_example_1(example_1):
    setup_data = SetupData(matches=example_1["matches"])
    df_left, df_right, df_matches = setup_data.data_preparation(
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
    df_left, df_right, df_matches = setup_data.data_preparation(
        df_panel=example_2["df"], unique_id="id"
    )

    # Assert the matches are as expected
    pd.testing.assert_frame_equal(
        df_matches.sort_values(by=["left", "right"]).reset_index(drop=True),
        example_2["expected_df_matches"].sort_values(by=["left", "right"]).reset_index(drop=True),
    )

    # Assert all values in 'left' of df_matches exist in df_left['id']
    assert set(df_matches['left']).issubset(df_left['id'])

    # Assert all values in 'right' of df_matches exist in df_right['id']
    assert set(df_matches['right']).issubset(df_right['id'])


def test_example_3(example_3):
    setup_data = SetupData()
    df_left, df_right, df_matches = setup_data.data_preparation(
        df_panel=example_3["df"], unique_id="id", panel_id="pid"
    )

    # Assert the matches are as expected
    pd.testing.assert_frame_equal(
        df_matches.sort_values(by=["left", "right"]).reset_index(drop=True),
        example_3["expected_df_matches"].sort_values(by=["left", "right"]).reset_index(drop=True),
    )

    # Assert all values in 'left' of df_matches exist in df_left['id']
    assert set(df_matches['left']).issubset(df_left['id'])

    # Assert all values in 'right' of df_matches exist in df_right['id']
    assert set(df_matches['right']).issubset(df_right['id'])


def test_example_4(example_4):
    setup_data = SetupData(matches=example_4["matches"])
    df_left, df_right, df_matches = setup_data.data_preparation(
        df_panel=example_4["df"], unique_id="id", panel_id="pid"
    )

    # Assert the matches are as expected
    pd.testing.assert_frame_equal(
        df_matches.sort_values(by=["left", "right"]).reset_index(drop=True),
        example_4["expected_df_matches"].sort_values(by=["left", "right"]).reset_index(drop=True),
    )

    # Assert all values in 'left' of df_matches exist in df_left['id']
    assert set(df_matches['left']).issubset(df_left['id'])

    # Assert all values in 'right' of df_matches exist in df_right['id']
    assert set(df_matches['right']).issubset(df_right['id'])

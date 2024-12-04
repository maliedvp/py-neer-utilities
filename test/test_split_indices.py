import pytest
import pandas as pd
from neer_match_utilities.split import split_test_train, SplitError


@pytest.fixture
def sample_data_no_duplicates():
    """Provides sample data for testing."""
    df_left = pd.DataFrame({'id': [100, 200, 400, 300, 500], 'value_left': ['A', 'B', 'C', 'D', 'E']})
    df_right = pd.DataFrame({'id': [10, 20, 30, 40, 50], 'value_right': ['A', 'B', 'D', 'C', 'E']})
    matches = pd.DataFrame({'left': [0, 1, 3, 2, 4], 'right': [0, 1, 2, 3, 4]})
    return df_left, df_right, matches


# without duplicates
def test_split_train_test(sample_data_no_duplicates):
    """Tests the split_train_test function to ensure proper splitting."""
    df_left, df_right, matches = sample_data_no_duplicates
    test_ratio = 0.2
    validation_ratio = 0.2
    results = split_test_train(df_left, df_right, matches, test_ratio, validation_ratio)

    def assert_index_in_range(df):
        assert list(df.index) == list(range(len(df))), \
            "The index starts with 0, is sorted, and increases by one in each line"

    for frame in results:
        assert_index_in_range(frame)

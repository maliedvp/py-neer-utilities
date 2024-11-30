import pytest
import pandas as pd
from neer_match_utilities.split import split_test_train, SplitError

@pytest.fixture
def sample_data_with_duplicates():
    """Provides sample data for testing."""
    df_left = pd.DataFrame({'id': [100, 400, 300, 500], 'value_left': ['A', 'B', 'D', 'E']})
    df_right = pd.DataFrame({'id': [10, 11, 30, 40, 50], 'value_right': ['A', 'A*', 'D', 'C', 'E']})
    matches = pd.DataFrame({'left': [0, 0, 1, 2, 3], 'right': [0, 1, 3, 2, 4]})

    return df_left, df_right, matches



# with duplicates
def test_split_train_test(sample_data_with_duplicates):
    """Tests the split_train_test function to ensure proper splitting."""
    df_left, df_right, matches = sample_data_with_duplicates
    test_ratio = 0.2
    validation_ratio = 0.2
    results = split_test_train(df_left, df_right, matches, test_ratio, validation_ratio)

    left_train, right_train, matches_train, left_validation, right_validation, matches_validation, left_test, right_test, matches_test = results

    print('TEST','\n')
    print(left_test,'\n')
    print(right_test,'\n')
    print(matches_test,'\n')

    print('TRAIN','\n')
    print(left_train,'\n')
    print(right_train,'\n')
    print(matches_train,'\n')

    print('VALIDATION','\n')
    print(left_validation,'\n')
    print(right_validation,'\n')
    print(matches_validation,'\n')

    # Verify sizes
    total_matches = len(matches)
    expected_test_size = int(round(test_ratio * total_matches))
    expected_validation_size = int(round(validation_ratio * total_matches))
    expected_train_size = total_matches - expected_test_size - expected_validation_size

    assert len(matches_test) == expected_test_size
    assert len(matches_validation) == expected_validation_size
    assert len(matches_train) == expected_train_size

    # Verify no overlap (not feasible as duplicates create overlap by definition)


def test_randomness_of_split(sample_data_with_duplicates):
    """Ensures that the splits are random."""
    df_left, df_right, matches = sample_data_with_duplicates
    split_1 = split_test_train(df_left, df_right, matches, test_ratio=0.3, validation_ratio=0.2)
    split_2 = split_test_train(df_left, df_right, matches, test_ratio=0.3, validation_ratio=0.2)

    matches_train_1, _, _, _, _, _, matches_test_1, _, _ = split_1
    matches_train_2, _, _, _, _, _, matches_test_2, _, _ = split_2

    assert not matches_train_1.equals(matches_train_2) or not matches_test_1.equals(matches_test_2), "Splits are not random."

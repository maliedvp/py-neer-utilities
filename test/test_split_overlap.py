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

    # Verify no overlap
    ## test
    left_test_dict = dict(zip(left_test.index, left_test['index_original']))
    test_indices_left = set([left_test_dict[i] for i in list(set(matches_test['left'].tolist()))])

    right_test_dict = dict(zip(right_test.index, right_test['index_original']))
    test_indices_right = set([right_test_dict[i] for i in list(set(matches_test['right'].tolist()))])
    
    # train
    left_train_dict = dict(zip(left_train.index, left_train['index_original']))
    train_indices_left = set([left_train_dict[i] for i in list(set(matches_train['left'].tolist()))])

    right_train_dict = dict(zip(right_train.index, right_train['index_original']))
    train_indices_right = set([right_train_dict[i] for i in list(set(matches_train['right'].tolist()))])

    # validation
    left_validation_dict = dict(zip(left_validation.index, left_validation['index_original']))
    validation_indices_left = set([left_validation_dict[i] for i in list(set(matches_validation['left'].tolist()))])

    right_validation_dict = dict(zip(right_validation.index, right_validation['index_original']))
    validation_indices_right = set([right_validation_dict[i] for i in list(set(matches_validation['right'].tolist()))])
    
    assert test_indices_left.isdisjoint(validation_indices_left), "Test and validation sets overlap in the left dataset."
    assert train_indices_left.isdisjoint(validation_indices_left), "Test and training sets overlap in the left dataset."
    assert validation_indices_left.isdisjoint(test_indices_left), "Validation and training sets overlap in the left dataset."

    assert test_indices_right.isdisjoint(validation_indices_right), "Test and validation sets overlap in the right dataset."
    assert train_indices_right.isdisjoint(validation_indices_right), "Test and training sets overlap in the right dataset."
    assert validation_indices_right.isdisjoint(test_indices_right), "Validation and training sets overlap in the right dataset."



def test_randomness_of_split(sample_data_no_duplicates):
    """Ensures that the splits are random."""
    df_left, df_right, matches = sample_data_no_duplicates
    split_1 = split_test_train(df_left, df_right, matches, test_ratio=0.3, validation_ratio=0.2)
    split_2 = split_test_train(df_left, df_right, matches, test_ratio=0.3, validation_ratio=0.2)

    matches_train_1, _, _, _, _, _, matches_test_1, _, _ = split_1
    matches_train_2, _, _, _, _, _, matches_test_2, _, _ = split_2

    assert not matches_train_1.equals(matches_train_2) or not matches_test_1.equals(matches_test_2), "Splits are not random."



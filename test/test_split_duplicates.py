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


def test_randomness_of_split(sample_data_with_duplicates, split_signature):
    """Ensures that the splits are random when no random_state is given."""
    df_left, df_right, matches = sample_data_with_duplicates

    # Randomness can only be observed across repeated draws. Comparing just two
    # is a coin flip: there are ~30 possible splits of 5 matches into 2/1/2, so
    # two draws collide about 1 time in 30. Over 10 draws a spurious failure
    # needs all 10 to coincide, i.e. roughly (1/30)**9.
    signatures = {
        split_signature(df_left, df_right, matches, test_ratio=0.3, validation_ratio=0.2)
        for _ in range(10)
    }

    assert len(signatures) > 1, "Splits are not random: 10 draws produced an identical split."


def test_split_is_reproducible_with_random_state(sample_data_with_duplicates, split_signature):
    """The same random_state must reproduce the same split, exactly."""
    df_left, df_right, matches = sample_data_with_duplicates

    signatures = {
        split_signature(
            df_left, df_right, matches,
            test_ratio=0.3, validation_ratio=0.2, random_state=42,
        )
        for _ in range(5)
    }

    assert len(signatures) == 1, "random_state=42 did not reproduce an identical split."


def test_different_random_states_give_different_splits(sample_data_with_duplicates, split_signature):
    """Different seeds must explore different splits."""
    df_left, df_right, matches = sample_data_with_duplicates

    # Asserted over a range of seeds rather than a single pair: any two given
    # seeds may legitimately land on the same split (~1 in 30 here).
    signatures = {
        split_signature(
            df_left, df_right, matches,
            test_ratio=0.3, validation_ratio=0.2, random_state=seed,
        )
        for seed in range(10)
    }

    assert len(signatures) > 1, "Every seed produced the same split; random_state is being ignored."

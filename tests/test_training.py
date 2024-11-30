import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from neer_match_utilities.training import Training


@pytest.fixture
def sample_data():
    """
    Provides sample data for testing Training methods.
    """
    df_left = pd.DataFrame({
        'id': [100, 200, 300],
        'value': ['A', 'B', 'C']
    })

    df_right = pd.DataFrame({
        'id': [20, 10, 40],
        'value': ['B', 'A', 'D']
    })

    matches = pd.DataFrame({
        'left_id': [100, 200],
        'right_id': [10, 20]
    })

    return df_left, df_right, matches


def test_matches_reorder(sample_data):
    """
    Tests the matches_reorder function of the Training class.
    """
    df_left, df_right, matches = sample_data

    # Create a mock Training object with required attributes
    training = Training(similarity_map={}, df_left=df_left, df_right=df_right, id_left='id', id_right='id')

    # Reorder matches
    reordered_matches = training.matches_reorder(matches, matches_id_left='left_id', matches_id_right='right_id')

    # Check the output
    assert list(reordered_matches.columns) == ['left', 'right'], "Reordered matches should have columns 'left' and 'right'"
    assert len(reordered_matches) == 2, "The length of the returned dataframe should only be two, corresponding to the number of matches specified above"
    assert reordered_matches['left'].tolist() == [0, 1], "Indices of left matches should be [0, 1]"
    assert reordered_matches['right'].tolist() == [1, 0], "Indices of right matches should be [1, 0]"


# def test_evaluate_dataframe():
#     """
#     Tests the evaluate_dataframe function of the Training class.
#     """
#     # Sample evaluation metrics
#     evaluation_test = {'true_positives': 50, 'true_negatives': 40, 'false_positives': 10, 'false_negatives': 20}
#     evaluation_train = {'true_positives': 60, 'true_negatives': 30, 'false_positives': 20, 'false_negatives': 10}

#     # Create a mock Training object
#     training = Training(similarity_map={}, df_left=pd.DataFrame(), df_right=pd.DataFrame(), id_left='id', id_right='id')

#     # Evaluate metrics
#     evaluation_df = training.evaluate_dataframe(evaluation_test, evaluation_train)

#     # Check the DataFrame structure and values
#     assert 'accuracy' in evaluation_df.columns, "The output should include an 'accuracy' column"
#     assert 'precision' in evaluation_df.columns, "The output should include a 'precision' column"
#     assert 'recall' in evaluation_df.columns, "The output should include a 'recall' column"
#     assert 'F-score' in evaluation_df.columns, "The output should include an 'F-score' column"

#     # Verify calculated metrics
#     test_row = evaluation_df.loc[evaluation_df['data'] == 'test']
#     assert test_row['accuracy'].values[0] == pytest.approx(0.75, rel=1e-2), "Test accuracy should be approximately 0.75"
#     assert test_row['precision'].values[0] == pytest.approx(0.83, rel=1e-2), "Test precision should be approximately 0.83"
#     assert test_row['recall'].values[0] == pytest.approx(0.71, rel=1e-2), "Test recall should be approximately 0.71"


# @patch("builtins.input", side_effect=['y'])
# def test_model_export(mock_input, tmp_path):
#     """
#     Tests the model_export function of the Training class.
#     """
#     # Mock model object
#     mock_model = Mock()
#     mock_model.save = Mock()

#     # Sample similarity map and evaluation data
#     similarity_map = {'col1~col2': 'numeric'}
#     evaluation_test = {'true_positives': 50, 'true_negatives': 40, 'false_positives': 10, 'false_negatives': 20}
#     evaluation_train = {'true_positives': 60, 'true_negatives': 30, 'false_positives': 20, 'false_negatives': 10}

#     # Create a mock Training object
#     training = Training(similarity_map=similarity_map, df_left=pd.DataFrame(), df_right=pd.DataFrame(), id_left='id', id_right='id')

#     # Call model_export
#     model_name = "test_model"
#     target_directory = tmp_path
#     training.model_export(mock_model, model_name, target_directory, evaluation_train, evaluation_test)

#     # Verify the model was saved
#     model_dir = target_dire

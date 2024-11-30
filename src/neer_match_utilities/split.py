import pandas as pd

class SplitError(Exception):
    """Custom exception for errors in data splitting."""
    pass


def split_test_train(left: pd.DataFrame, right: pd.DataFrame, matches: pd.DataFrame, test_ratio: float = 0.3, validation_ratio: float = 0.1):
    """
    Splits datasets into training, validation, and testing subsets.

    This function ensures that only observations from `left` and `right` that are referenced
    in the `matches` DataFrame are included in the split process.

    Parameters:
    -----------
    left : pd.DataFrame
        The left dataset to split.
    right : pd.DataFrame
        The right dataset to split.
    matches : pd.DataFrame
        A DataFrame containing matching pairs between the left and right datasets.
        It must include columns 'left' and 'right', referencing indices in `left` and `right`.
    test_ratio : float, optional
        The proportion of the data to be used for testing (default is 0.3).
    validation_ratio : float, optional
        The proportion of the data to be used for validation (default is 0.1).

    Returns:
    --------
    tuple:
        - left_train : pd.DataFrame
        - right_train : pd.DataFrame
        - matches_train : pd.DataFrame
        - left_validation : pd.DataFrame
        - right_validation : pd.DataFrame
        - matches_validation : pd.DataFrame
        - left_test : pd.DataFrame
        - right_test : pd.DataFrame
        - matches_test : pd.DataFrame

    Raises:
    -------
    SplitError
        If the total counts of split subsets do not match the original dataset size.
    """
    
    # Filter `left` and `right` to only include rows referenced in `matches`
    left = left[left.index.isin(matches['left'])]
    right = right[right.index.isin(matches['right'])]

    # Get absolute counts for each subset
    matches_test_count = int(round(test_ratio * len(matches)))
    matches_validation_count = int(round(validation_ratio * len(matches)))
    matches_train_count = len(matches) - matches_test_count - matches_validation_count

    # Validate the requested counts
    if matches_test_count + matches_validation_count + matches_train_count != len(matches):
        raise SplitError("Split does not work correctly: The sum of observations in the three datasets is not equal to the number of observations in matches")

    def subsets(matches_count: int, matches_exclude_index: list):
        """Generates random subsets for matches DataFrame and corresponding rows in `left` and `right`."""
        m = matches[~matches.index.isin(matches_exclude_index)].sample(n=matches_count)
        l = left[left.index.isin(m['left'])]
        r = right[right.index.isin(m['right'])]
        return m, l, r

    matches_test, left_test, right_test = subsets(matches_test_count, matches_exclude_index=[])
    matches_exclude_index = matches_test.index.tolist()

    matches_validation, left_validation, right_validation = subsets(matches_validation_count, matches_exclude_index=matches_exclude_index)
    matches_exclude_index += matches_validation.index.tolist()

    matches_train, left_train, right_train = subsets(matches_train_count, matches_exclude_index=matches_exclude_index)

    # adjust the indices of the matches dataframes (required by pyneer)
    matches_train = matches_train.sort_values(by='left', ascending=True).reset_index(drop=True)
    matches_test = matches_test.sort_values(by='left', ascending=True).reset_index(drop=True)
    matches_validation = matches_validation.sort_values(by='left', ascending=True).reset_index(drop=True)


    return left_train, right_train, matches_train, left_validation, right_validation, matches_validation, left_test, right_test, matches_test
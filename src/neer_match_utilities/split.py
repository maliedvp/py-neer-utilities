import numpy as np
import pandas as pd

class SplitError(Exception):
    """Custom exception for errors in data splitting."""
    pass


def _match_components(matches: pd.DataFrame) -> pd.Series:
    """
    Labels every row of `matches` with its connected component.

    Two rows belong to the same component if they share a left or a right
    record, directly or through other rows. Components are numbered in the
    order in which they first appear in `matches`.
    """
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for l, r in zip(matches['left'], matches['right']):
        a, b = find(('l', l)), find(('r', r))
        if a != b:
            parent[a] = b

    labels = {}
    return pd.Series(
        [labels.setdefault(find(('l', l)), len(labels)) for l in matches['left']],
        index=matches.index,
    )


def split_test_train(left: pd.DataFrame, right: pd.DataFrame, matches: pd.DataFrame, test_ratio: float = 0.3, validation_ratio: float = 0.1, random_state: int | None = None, keep_components: bool = False):
    """
    Splits datasets into training, validation, and testing subsets.

    This function ensures that only observations from `left` and `right` that are referenced
    in the `matches` DataFrame are included in the split process.

    Parameters
    ----------
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
    random_state : int or None, optional
        Seed for the sampling of the three subsets. Pass an integer to obtain a
        reproducible split, which is required to re-create a training run. If
        None (default), the split is drawn from unseeded entropy and differs on
        every call.
    keep_components : bool, optional
        If False (default), the match pairs are drawn one by one. If True, pairs
        that share a left or a right record, directly or through other pairs,
        stay in the same subset. Use it when one entity has several pairs, for
        example links along a chain of volumes (1922 -> 1925 -> 1926) completed
        by `SetupData.data_preparation_cs`: drawn one by one, the pairs of a
        chain end up in different subsets, and inside a subset a true match
        whose pair went to another subset counts as a non-match. Whole groups
        fill the test subset, then the validation subset, in random order, so
        these two can exceed their requested size by less than one group; the
        training subset takes the rest. No left or right record then appears
        in two subsets.

    Returns
    -------
    tuple
        A tuple containing:
        - left_train : pd.DataFrame
        - right_train : pd.DataFrame
        - matches_train : pd.DataFrame
        - left_validation : pd.DataFrame
        - right_validation : pd.DataFrame
        - matches_validation : pd.DataFrame
        - left_test : pd.DataFrame
        - right_test : pd.DataFrame
        - matches_test : pd.DataFrame

    Raises
    ------
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

    # A single generator is shared by the three draws below, so that its state
    # advances between them (the draws stay independent) while the whole split
    # remains reproducible from `random_state`.
    rng = np.random.default_rng(random_state)

    if keep_components:
        component = _match_components(matches)
        sizes = component.value_counts(sort=False).sort_index()
        targets = (matches_test_count, matches_validation_count)
        filled = [0, 0]
        part_of = {}
        for c in sizes.index[rng.permutation(len(sizes))]:
            k = next((k for k in (0, 1) if filled[k] < targets[k]), 2)
            part_of[c] = k
            if k < 2:
                filled[k] += sizes[c]
        part = component.map(part_of)

        def take(k: int):
            """Returns the matches of subset k and their rows in `left` and `right`."""
            m = matches[part == k].copy()
            l = left[left.index.isin(m['left'])].copy()
            r = right[right.index.isin(m['right'])].copy()
            return m, l, r

        matches_test, left_test, right_test = take(0)
        matches_validation, left_validation, right_validation = take(1)
        matches_train, left_train, right_train = take(2)
        return _finish(
            left_train, right_train, matches_train,
            left_validation, right_validation, matches_validation,
            left_test, right_test, matches_test,
        )

    def subsets(matches_count: int, matches_exclude_index: list):
        """Generates random subsets for matches DataFrame and corresponding rows in `left` and `right`."""
        m = matches[~matches.index.isin(matches_exclude_index)].sample(n=matches_count, random_state=rng).copy()
        l = left[left.index.isin(m['left'])].copy()
        r = right[right.index.isin(m['right'])].copy()
        return m, l, r

    matches_test, left_test, right_test = subsets(
        matches_test_count, 
        matches_exclude_index=[]
    )
    matches_exclude_index = matches_test.index.tolist()

    matches_validation, left_validation, right_validation = subsets(
        matches_validation_count, 
        matches_exclude_index=matches_exclude_index
    )
    matches_exclude_index += matches_validation.index.tolist()

    matches_train, left_train, right_train = subsets(
        matches_train_count, 
        matches_exclude_index=matches_exclude_index
    )

    return _finish(
        left_train, right_train, matches_train,
        left_validation, right_validation, matches_validation,
        left_test, right_test, matches_test,
    )


def _finish(left_train, right_train, matches_train,
            left_validation, right_validation, matches_validation,
            left_test, right_test, matches_test):
    """
    Re-indexes the subsets of `split_test_train` and maps their matches to the
    new positions; the original row labels are kept in `index_original`.
    """
    def process_dataset(df):
        """
        Processes a single dataset by adding an 'index_original' column,
        resetting the index, and creating a mapping dictionary.
        """
        df['index_original'] = df.index.copy()
        df = df.reset_index(drop=True)
        mapping_dict = dict(zip(df['index_original'], df.index))
        return df, mapping_dict

    # Process datasets individually
    left_train, left_train_dict = process_dataset(left_train)
    right_train, right_train_dict = process_dataset(right_train)

    left_test, left_test_dict = process_dataset(left_test)
    right_test, right_test_dict = process_dataset(right_test)

    left_validation, left_validate_dict = process_dataset(left_validation)
    right_validation, right_validation_dict = process_dataset(right_validation)

    # Update matches dataframes
    def update_matches(matches_df, left_dict, right_dict):
        matches_df['left'] = matches_df['left'].map(left_dict)
        matches_df['right'] = matches_df['right'].map(right_dict)
        return matches_df.sort_values(by='left', ascending=True).reset_index(drop=True)

    matches_train = update_matches(
        matches_train, 
        left_train_dict, 
        right_train_dict
    )

    matches_test = update_matches(
        matches_test, 
        left_test_dict, 
        right_test_dict
    )

    matches_validation = update_matches(
        matches_validation, 
        left_validate_dict, 
        right_validation_dict
    )


    return left_train, right_train, matches_train, left_validation, right_validation, matches_validation, left_test, right_test, matches_test

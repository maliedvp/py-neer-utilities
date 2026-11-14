import warnings

import numpy as np
import pandas as pd

# Column with the original row labels that `split_test_train` adds to every subset
ORIGINAL_INDEX = 'index_original'


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


def _check_non_matches(non_matches: pd.DataFrame, matches: pd.DataFrame,
                       left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Validates the labelled non-match pairs and drops duplicate pairs."""
    if not {'left', 'right'} <= set(non_matches.columns):
        raise ValueError("non_matches must have the columns 'left' and 'right'.")
    unknown = ~non_matches['left'].isin(left.index) | ~non_matches['right'].isin(right.index)
    if unknown.any():
        raise ValueError(
            f"{int(unknown.sum())} pairs of non_matches refer to rows that are not in "
            "left or right (they must use the row labels of left and right, as matches does; "
            "convert IDs with Training.matches_reorder)."
        )
    both = set(zip(matches['left'], matches['right'])) & set(zip(non_matches['left'], non_matches['right']))
    if both:
        raise ValueError(f"{len(both)} pairs are in both matches and non_matches, e.g. {sorted(both)[:3]}.")
    return non_matches.drop_duplicates(subset=['left', 'right'])


def _drop_partners(right_test: pd.DataFrame, matches_test: pd.DataFrame, share: float,
                   rng: np.random.Generator, protected_right) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes the right record of a share of the test matches, and the pair itself.

    Only pairs whose left and right records have no other match in the test
    subset are eligible, so that each removal leaves exactly one left record
    without a partner. Right records in `protected_right` (those of labelled
    non-match pairs) are kept.
    """
    n = int(round(share * len(matches_test)))
    if n == 0:
        return right_test, matches_test
    n_left = matches_test['left'].map(matches_test['left'].value_counts())
    n_right = matches_test['right'].map(matches_test['right'].value_counts())
    eligible = matches_test[(n_left == 1) & (n_right == 1) & ~matches_test['right'].isin(protected_right)]
    if n > len(eligible):
        warnings.warn(
            f"drop_partner_share={share} asks for {n} test pairs, but only {len(eligible)} "
            "are one-to-one pairs whose right record can be removed; removing those."
        )
        n = len(eligible)
    drop = eligible.sample(n=n, random_state=rng)
    return right_test[~right_test.index.isin(drop['right'])].copy(), matches_test.drop(index=drop.index)


def pairs_in_subset(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    Maps pairs given in the original row labels to the rows of one subset.

    `split_test_train` re-indexes every subset and keeps the original row labels
    in the column ``index_original``. This function translates pairs that refer
    to the original labels, e.g. the labelled non-matches passed as
    `non_matches`, to the row labels of one subset, and keeps only the pairs
    whose two records are both in that subset. Rows without an original label
    (synthetic rows added by `synth_matches` or `synth_mismatches`) never match.

    Parameters
    ----------
    pairs : pd.DataFrame
        Pairs with columns 'left' and 'right' in the original row labels; other
        columns are kept.
    left, right : pd.DataFrame
        One subset returned by `split_test_train`, e.g. `left_test` and `right_test`.

    Returns
    -------
    pd.DataFrame
        The pairs inside the subset, with 'left' and 'right' as row labels of
        `left` and `right`.
    """
    def labels(df: pd.DataFrame) -> dict:
        if ORIGINAL_INDEX not in df.columns:
            raise ValueError(f"'{ORIGINAL_INDEX}' is missing: pass a subset returned by split_test_train.")
        s = df[ORIGINAL_INDEX].dropna()
        return dict(zip(s.to_numpy(), s.index))

    pos_l, pos_r = labels(left), labels(right)
    keep = pairs['left'].isin(list(pos_l)) & pairs['right'].isin(list(pos_r))
    out = pairs[keep].copy()
    out['left'] = out['left'].map(pos_l)
    out['right'] = out['right'].map(pos_r)
    return out.reset_index(drop=True)


def split_test_train(left: pd.DataFrame, right: pd.DataFrame, matches: pd.DataFrame, test_ratio: float = 0.3, validation_ratio: float = 0.1, random_state: int | None = None, keep_components: bool = False, non_matches: pd.DataFrame | None = None, drop_partner_share: float = 0.0):
    """
    Splits datasets into training, validation, and testing subsets.

    This function ensures that only observations from `left` and `right` that are referenced
    in the `matches` DataFrame (or in `non_matches`, if given) are included in the split process.

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
    non_matches : pd.DataFrame or None, optional
        Pairs with columns 'left' and 'right' (row labels of `left` and `right`,
        as in `matches`) that were checked by hand and are known not to match,
        e.g. confusable records. Keep them in a file like `matches.csv`
        (e.g. `non_matches.csv`, with the IDs of both records) and convert the
        IDs with `Training.matches_reorder`, as for `matches`. Default None:
        records without a match are dropped, as before. If given, the records
        of these pairs are kept, and
        the subsets are drawn as with `keep_components=True` over matches and
        non-matches together, so that both records of a non-match pair, and the
        matches of these records, end up in the same subset; inside it the pair
        is a non-match like every pair not in `matches`. A record that appears
        only in `non_matches` has no partner in its subset, so the subsets then
        contain records for which "no match" is the correct answer. Groups
        without any match fall into the subset that is being filled when they
        are drawn, which spreads them over the subsets roughly in proportion.
        To find the pairs in a subset, use `pairs_in_subset`.
    drop_partner_share : float, optional
        Share of the test matches (default 0.0) whose right record is removed
        from `right_test`, with the pair from `matches_test`. Their left records
        stay in `left_test` without a partner, like persons who died, emigrated
        or were not recorded, so the test shows how often a model links a record
        that has no true counterpart. Only pairs whose left and right records
        have no other match in the test subset are eligible (and no right record
        of `non_matches`); if fewer are eligible, all of them are removed, with
        a warning. The removal is drawn after the split from the same generator,
        so the subsets are otherwise identical to those for 0.0.

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
    ValueError
        If `non_matches` refers to rows that are not in `left` or `right`, or
        shares a pair with `matches`, or if `drop_partner_share` is not in [0, 1].
    """
    if not 0.0 <= drop_partner_share <= 1.0:
        raise ValueError("drop_partner_share must be between 0 and 1.")
    if non_matches is not None:
        non_matches = _check_non_matches(non_matches, matches, left, right)
        keep_components = True
        protected_right = set(non_matches['right'])
    else:
        protected_right = set()

    # Filter `left` and `right` to only include rows referenced in `matches` (or `non_matches`)
    keep_left, keep_right = matches['left'], matches['right']
    if non_matches is not None:
        keep_left = pd.concat([keep_left, non_matches['left']])
        keep_right = pd.concat([keep_right, non_matches['right']])
    left = left[left.index.isin(keep_left)]
    right = right[right.index.isin(keep_right)]

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
        if non_matches is None:
            component = _match_components(matches)
            sizes = component.value_counts(sort=False).sort_index()
        else:
            # one graph over matches and non-matches; groups without a match get size 0
            pairs = pd.concat([matches[['left', 'right']], non_matches[['left', 'right']]], ignore_index=True)
            labels = _match_components(pairs).to_numpy()
            component = pd.Series(labels[:len(matches)], index=matches.index)
            non_match_component = pd.Series(labels[len(matches):], index=non_matches.index)
            sizes = component.value_counts(sort=False).reindex(np.unique(labels), fill_value=0)
        targets = (matches_test_count, matches_validation_count)
        filled = [0, 0]
        part_of = {}
        for c in sizes.index[rng.permutation(len(sizes))]:
            k = next((k for k in (0, 1) if filled[k] < targets[k]), 2)
            part_of[c] = k
            if k < 2:
                filled[k] += sizes[c]
        part = component.map(part_of)
        non_match_part = None if non_matches is None else non_match_component.map(part_of)

        def take(k: int):
            """Returns the matches of subset k and their rows in `left` and `right`."""
            m = matches[part == k].copy()
            keep_l, keep_r = m['left'], m['right']
            if non_matches is not None:
                nm = non_matches[non_match_part == k]
                keep_l = pd.concat([keep_l, nm['left']])
                keep_r = pd.concat([keep_r, nm['right']])
            l = left[left.index.isin(keep_l)].copy()
            r = right[right.index.isin(keep_r)].copy()
            return m, l, r

        matches_test, left_test, right_test = take(0)
        matches_validation, left_validation, right_validation = take(1)
        matches_train, left_train, right_train = take(2)
        right_test, matches_test = _drop_partners(right_test, matches_test, drop_partner_share, rng, protected_right)
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
    right_test, matches_test = _drop_partners(right_test, matches_test, drop_partner_share, rng, protected_right)

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
        df[ORIGINAL_INDEX] = df.index.copy()
        df = df.reset_index(drop=True)
        mapping_dict = dict(zip(df[ORIGINAL_INDEX], df.index))
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

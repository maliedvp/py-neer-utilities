import pytest

from neer_match_utilities.split import split_test_train


@pytest.fixture
def split_signature():
    """
    Returns a callable that draws one split and reduces it to a comparable value.

    Identifying a split is subtler than it looks. The returned `matches` frames
    are re-indexed to their own subset -- `left`/`right` hold positions within
    `left_train`/`right_train`, not the original row labels -- so many different
    splits share the same remapped pairs, e.g. ((0, 0), (1, 1)). Comparing those
    directly would report distinct splits as identical.

    The original labels survive in the `index_original` column that
    split_test_train attaches to every left/right frame. Mapping the remapped
    positions back through it recovers the actual match pairs, which do identify
    a split.
    """
    def _signature(df_left, df_right, matches, **kwargs):
        (
            left_train, right_train, matches_train,
            left_validation, right_validation, matches_validation,
            left_test, right_test, matches_test,
        ) = split_test_train(df_left, df_right, matches, **kwargs)

        def original_pairs(left_df, right_df, matches_df):
            left_labels = left_df['index_original'].to_dict()
            right_labels = right_df['index_original'].to_dict()
            return tuple(sorted(
                (left_labels[l], right_labels[r])
                for l, r in matches_df[['left', 'right']].itertuples(index=False, name=None)
            ))

        return tuple(
            original_pairs(left_df, right_df, matches_df)
            for left_df, right_df, matches_df in (
                (left_train, right_train, matches_train),
                (left_validation, right_validation, matches_validation),
                (left_test, right_test, matches_test),
            )
        )

    return _signature

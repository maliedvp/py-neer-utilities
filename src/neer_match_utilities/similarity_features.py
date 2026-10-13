from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from neer_match import similarity_map as _sim

@dataclass
class SimilarityFeatures:
    similarity_map: _sim.SimilarityMap

    def pairwise_similarity_dataframe(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        matches: pd.DataFrame,
        left_id_col: str,
        right_id_col: str,
        match_col: str = "match",
        matches_id_left: str = "left",
        matches_id_right: str = "right",
        matches_are_indices: bool = True,
    ) -> pd.DataFrame:
        """
        Build full cross-join of left × right, compute similarity features
        specified in SimilarityMap, and attach match indicator.

        Parameters
        ----------
        left, right :
            Left- and right-hand side entity tables.
        matches :
            DataFrame describing which pairs are true matches.
            If `matches_are_indices=True`, `matches[matches_id_left]` and
            `matches[matches_id_right]` are interpreted as row indices into
            `left` and `right` (0..n-1). If False, they are interpreted as
            IDs in the same space as `left[left_id_col]` / `right[right_id_col]`.
            A pair listed more than once is counted once.
        left_id_col, right_id_col :
            Column names in `left` / `right` that contain the entity IDs.
        match_col :
            Name of the binary match indicator column in the output.
        matches_id_left, matches_id_right :
            Column names in `matches` identifying the left/right side.
        matches_are_indices :
            If True (default), treat `matches_id_left` / `matches_id_right` as
            row indices into `left` and `right`. If False, treat them as IDs.

        Notes
        -----
        The frame holds one row per pair, so its memory grows with
        ``len(left) * len(right)``. :meth:`pairs` builds the same rows in blocks,
        or only selected ones.
        """
        return self.pairs(
            left, right, matches, left_id_col, right_id_col,
            match_col=match_col,
            matches_id_left=matches_id_left,
            matches_id_right=matches_id_right,
            matches_are_indices=matches_are_indices,
        ).frame()

    def pairs(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        matches: pd.DataFrame,
        left_id_col: str,
        right_id_col: str,
        match_col: str = "match",
        matches_id_left: str = "left",
        matches_id_right: str = "right",
        matches_are_indices: bool = True,
    ) -> "SimilarityPairs":
        """
        The rows of :meth:`pairwise_similarity_dataframe`, built on demand.

        Takes the same arguments. Nothing is computed per pair until rows are
        requested from the returned object, in full, in blocks of left rows, or
        selected by position.
        """
        return SimilarityPairs(
            self.similarity_map, left, right, matches, left_id_col, right_id_col,
            match_col, matches_id_left, matches_id_right, matches_are_indices,
        )


class SimilarityPairs:
    """
    The rows of a pairwise similarity frame, built on demand.

    Row ``k`` pairs left row ``k // len(right)`` with right row
    ``k % len(right)``, as in the full cross join. Everything that does not
    depend on the pair -- the ID lookups, each compared column's values, the
    match keys -- is derived once here, so building the rows in several calls
    costs no more than building them in one, and gives the same values.
    """

    def __init__(
        self,
        similarity_map: _sim.SimilarityMap,
        left: pd.DataFrame,
        right: pd.DataFrame,
        matches: pd.DataFrame,
        left_id_col: str,
        right_id_col: str,
        match_col: str,
        matches_id_left: str,
        matches_id_right: str,
        matches_are_indices: bool,
    ):
        # ------------------------------------------------------------------
        # 1. Extract ID arrays explicitly from left/right columns
        # ------------------------------------------------------------------
        if left_id_col not in left.columns:
            raise KeyError(f"{left_id_col!r} not in left.columns")

        if right_id_col not in right.columns:
            raise KeyError(f"{right_id_col!r} not in right.columns")

        self.left_ids = left[left_id_col].to_numpy()
        self.right_ids = right[right_id_col].to_numpy()
        self.n_left, self.n_right = len(self.left_ids), len(self.right_ids)

        self.match_col = match_col
        self.matches_id_left = matches_id_left
        self.matches_id_right = matches_id_right
        self.matches_are_indices = matches_are_indices

        # Decide output column names for IDs
        if left_id_col == right_id_col:
            self.out_left_id = f"{left_id_col}_left"
            self.out_right_id = f"{right_id_col}_right"
        else:
            self.out_left_id = left_id_col
            self.out_right_id = right_id_col

        if len(self) == 0:
            return

        # For lookup (ID → row)
        left_indexed = left.set_index(left_id_col)
        right_indexed = right.set_index(right_id_col)
        left_id_series = pd.Series(self.left_ids)
        right_id_series = pd.Series(self.right_ids)

        # ------------------------------------------------------------------
        # 2. Load similarity functions (built-in + custom)
        # ------------------------------------------------------------------
        sim_funcs = _sim.available_similarities()

        # ------------------------------------------------------------------
        # 3. Per similarity, each left and right row's value, looked up by ID.
        #    Kept as Series so that every block sees the same values and dtype.
        # ------------------------------------------------------------------
        self.features = []
        for lcol, rcol, sim_name in similarity_map:
            if sim_name not in sim_funcs:
                raise KeyError(
                    f"Similarity '{sim_name}' not found. "
                    f"Available: {sorted(sim_funcs.keys())}"
                )

            # Dict-based lookup to avoid index alignment quirks
            left_values = left_id_series.map(left_indexed[lcol].to_dict())
            right_values = right_id_series.map(right_indexed[rcol].to_dict())

            self.features.append((
                f"col_{lcol}_{rcol}_{sim_name}",
                sim_funcs[sim_name],
                left_values,
                right_values,
                left_values.isna().to_numpy(),
                right_values.isna().to_numpy(),
            ))

        # ------------------------------------------------------------------
        # 4. Match keys
        # ------------------------------------------------------------------
        if not {matches_id_left, matches_id_right}.issubset(matches.columns):
            raise KeyError(
                f"'matches' must contain columns {matches_id_left!r} and {matches_id_right!r}"
            )

        matches_tmp = matches[[matches_id_left, matches_id_right]].copy()

        if matches_are_indices:
            # matches[left/right] are row indices into left/right.
            # Map our ID-columns to row indices, then merge on those.
            left_id_to_pos = (
                left.reset_index()
                .set_index(left_id_col)["index"]
                .to_dict()
            )
            right_id_to_pos = (
                right.reset_index()
                .set_index(right_id_col)["index"]
                .to_dict()
            )
            self.left_keys = left_id_series.map(left_id_to_pos).to_numpy()
            self.right_keys = right_id_series.map(right_id_to_pos).to_numpy()
        else:
            # Old behavior: matches[left/right] are in the same ID space
            # as out_left_id / out_right_id

            # Ensure dtype compatibility before merging
            matches_tmp[matches_id_left] = matches_tmp[matches_id_left].astype(left_id_series.dtype)
            matches_tmp[matches_id_right] = matches_tmp[matches_id_right].astype(right_id_series.dtype)

        # A pair listed twice would otherwise appear twice in the frame
        matches_tmp = matches_tmp.drop_duplicates()
        matches_tmp[match_col] = 1
        self.matches_tmp = matches_tmp

    def __len__(self) -> int:
        return self.n_left * self.n_right

    def frame(self) -> pd.DataFrame:
        """All rows, as :meth:`SimilarityFeatures.pairwise_similarity_dataframe` returns them."""
        return self._block(0, self.n_left)

    def chunks(self, pairs_per_chunk: int = 250_000) -> Iterator[pd.DataFrame]:
        """
        All rows, in consecutive blocks of whole left rows.

        Each block holds as many left rows as fit in ``pairs_per_chunk`` pairs, and
        at least one. Its index continues the full frame's, so concatenated, the
        blocks equal :meth:`frame`.
        """
        for start, stop in self._row_blocks(pairs_per_chunk):
            yield self._block(start, stop)

    def match_flags(self, pairs_per_chunk: int = 250_000) -> np.ndarray:
        """The match column of :meth:`frame`, without computing any similarity."""
        flags = [
            self._block(start, stop, features=False)[self.match_col].to_numpy()
            for start, stop in self._row_blocks(pairs_per_chunk)
        ]
        return np.concatenate(flags) if flags else np.empty(0, dtype=int)

    def rows(self, positions) -> pd.DataFrame:
        """The rows of :meth:`frame` at ``positions``, in that order and indexed by them."""
        positions = np.asarray(positions, dtype=np.intp)
        if len(self) == 0:
            return self.frame()
        return self._build(
            positions // self.n_right, positions % self.n_right, pd.Index(positions)
        )

    def _row_blocks(self, pairs_per_chunk: int):
        if pairs_per_chunk < 1:
            raise ValueError(f"pairs_per_chunk must be at least 1, got {pairs_per_chunk}")
        if len(self) == 0:
            yield 0, self.n_left
            return
        rows_per_block = max(1, pairs_per_chunk // self.n_right)
        for start in range(0, self.n_left, rows_per_block):
            yield start, min(start + rows_per_block, self.n_left)

    def _block(self, start: int, stop: int, features: bool = True) -> pd.DataFrame:
        if len(self) == 0:
            return pd.DataFrame(columns=[self.out_left_id, self.out_right_id, self.match_col])
        return self._build(
            np.repeat(np.arange(start, stop), self.n_right),
            np.tile(np.arange(self.n_right), stop - start),
            pd.RangeIndex(start * self.n_right, stop * self.n_right),
            features=features,
        )

    def _build(self, left_pos, right_pos, index, features: bool = True) -> pd.DataFrame:
        match_col = self.match_col
        matches_id_left, matches_id_right = self.matches_id_left, self.matches_id_right
        out_left_id, out_right_id = self.out_left_id, self.out_right_id

        df_pairs = pd.DataFrame(
            {
                out_left_id: self.left_ids[left_pos],
                out_right_id: self.right_ids[right_pos],
            }
        )

        # ------------------------------------------------------------------
        # 5. Compute similarity features only for entries in SimilarityMap
        # ------------------------------------------------------------------
        for col_name, func, left_values, right_values, left_nan, right_nan in (
            self.features if features else []
        ):
            left_series = left_values.take(left_pos)
            right_series = right_values.take(right_pos)

            # Mask where any side is missing
            nan_mask = left_nan[left_pos] | right_nan[right_pos]

            sim_vals = []
            for x, y, is_nan in zip(left_series, right_series, nan_mask):
                if is_nan:
                    # if any NaN involved → similarity 0
                    sim_vals.append(0.0)
                else:
                    sim_vals.append(func(x, y))

            df_pairs[col_name] = sim_vals

        # ------------------------------------------------------------------
        # 6. Match indicator
        # ------------------------------------------------------------------
        if self.matches_are_indices:
            df_pairs["_left_pos"] = self.left_keys[left_pos]
            df_pairs["_right_pos"] = self.right_keys[right_pos]

            df_pairs = df_pairs.merge(
                self.matches_tmp,
                left_on=["_left_pos", "_right_pos"],
                right_on=[matches_id_left, matches_id_right],
                how="left",
            )

            df_pairs[match_col] = df_pairs[match_col].fillna(0).astype(int)

            # Clean up helper columns
            df_pairs = df_pairs.drop(columns=["_left_pos", "_right_pos", matches_id_left, matches_id_right])

        else:
            df_pairs = df_pairs.merge(
                self.matches_tmp,
                left_on=[out_left_id, out_right_id],
                right_on=[matches_id_left, matches_id_right],
                how="left",
            )
            df_pairs[match_col] = df_pairs[match_col].fillna(0).astype(int)

            if matches_id_left != out_left_id:
                df_pairs = df_pairs.drop(columns=[matches_id_left])
            if matches_id_right != out_right_id:
                df_pairs = df_pairs.drop(columns=[matches_id_right])

        df_pairs.index = index
        return df_pairs


def to_X_y(df: pd.DataFrame, match_col: str = "match"):
    """
    Extract (X, y) from a pairwise similarity DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by SimilarityFeatures.pairwise_similarity_dataframe().
    match_col : str, default "match"
        Name of the binary match indicator column.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix containing all similarity columns (col_*).
    y : np.ndarray
        Target array (0/1).
    """
    if match_col not in df.columns:
        raise KeyError(f"Match column '{match_col}' not found in DataFrame.")

    # Select all similarity feature columns
    feature_cols = sorted([c for c in df.columns if c.startswith("col_")])

    X = df[feature_cols]
    y = df[match_col].to_numpy()

    return X, y

def subsample_non_matches(
    df: pd.DataFrame,
    match_col: str = "match",
    mismatch_share: float = 1.0,
    random_state: int | None = None,
    shuffle: bool = True,
) -> pd.DataFrame:
    """
    Return a subsample of df where all matches are kept and a fraction of
    non-matches is sampled.

    Parameters
    ----------
    df :
        DataFrame with a binary match column.
    match_col :
        Name of the binary target column (1 = match, 0 = non-match).
    mismatch_share :
        Share of non-matches to keep. Must satisfy 0 < mismatch_share <= 1.0.
        - 1.0 → keep all non-matches
        - 0.1 → keep 10% of non-matches
    random_state :
        Random seed for reproducible sampling.
    shuffle :
        If True, shuffle the resulting DataFrame.

    Returns
    -------
    df_sub :
        Subsampled DataFrame with all matches and a subset of non-matches.
    """
    if not (0 < mismatch_share <= 1.0):
        raise ValueError(
            f"mismatch_share must be in (0, 1], got {mismatch_share}"
        )

    if match_col not in df.columns:
        raise KeyError(f"Match column '{match_col}' not found in df.columns")

    mask_pos = df[match_col] == 1
    mask_neg = ~mask_pos

    df_pos = df[mask_pos]
    df_neg = df[mask_neg]

    if mismatch_share < 1.0:
        df_neg = df_neg.sample(
            frac=mismatch_share,
            random_state=random_state,
        )

    df_sub = pd.concat([df_pos, df_neg], axis=0)

    if shuffle:
        df_sub = df_sub.sample(frac=1.0, random_state=random_state)

    return df_sub

def subsample_non_match_positions(
    match: np.ndarray,
    mismatch_share: float = 1.0,
    random_state: int | None = None,
    shuffle: bool = True,
) -> np.ndarray:
    """
    Positions of the rows :func:`subsample_non_matches` keeps, in its order.

    ``subsample_non_matches(df, ...)`` equals ``df.iloc[positions]`` for
    ``positions = subsample_non_match_positions(df[match_col].to_numpy(), ...)``
    and the same remaining arguments. Only the match column is needed, so the
    similarity features can be computed afterwards for the kept rows alone.

    Both draws are made as pandas' ``DataFrame.sample`` makes them: an integer
    seed starts a fresh ``numpy.random.RandomState`` for each, None uses numpy's
    global one, and ``round(mismatch_share * n)`` non-matches are kept. Calling
    numpy directly keeps the result independent of pandas' implementation;
    numpy holds the ``RandomState`` stream fixed across versions (NEP 19).

    Parameters
    ----------
    match :
        Binary match indicator per row (1 = match).
    mismatch_share, random_state, shuffle :
        As for :func:`subsample_non_matches`.

    Returns
    -------
    np.ndarray
        Row positions: all matches, then the sampled non-matches, shuffled
        together if ``shuffle`` is True.
    """
    if not (0 < mismatch_share <= 1.0):
        raise ValueError(
            f"mismatch_share must be in (0, 1], got {mismatch_share}"
        )

    is_match = np.asarray(match) == 1
    positions_pos = np.flatnonzero(is_match)
    positions_neg = np.flatnonzero(~is_match)

    if mismatch_share < 1.0:
        size = round(mismatch_share * len(positions_neg))
        positions_neg = positions_neg[
            _random_state(random_state).choice(len(positions_neg), size=size, replace=False)
        ]

    positions = np.concatenate([positions_pos, positions_neg])

    if shuffle:
        positions = positions[
            _random_state(random_state).choice(len(positions), size=len(positions), replace=False)
        ]

    return positions


def _random_state(random_state):
    """The numpy random state pandas' ``DataFrame.sample`` uses for ``random_state``."""
    if random_state is None:
        return np.random
    if isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        return random_state
    return np.random.RandomState(random_state)

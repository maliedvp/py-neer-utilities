from __future__ import annotations

from dataclasses import dataclass

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
        left_id_col, right_id_col :
            Column names in `left` / `right` that contain the entity IDs.
        match_col :
            Name of the binary match indicator column in the output.
        matches_id_left, matches_id_right :
            Column names in `matches` identifying the left/right side.
        matches_are_indices :
            If True (default), treat `matches_id_left` / `matches_id_right` as
            row indices into `left` and `right`. If False, treat them as IDs.
        """

        # ------------------------------------------------------------------
        # 1. Extract ID arrays explicitly from left/right columns
        # ------------------------------------------------------------------
        if left_id_col not in left.columns:
            raise KeyError(f"{left_id_col!r} not in left.columns")

        if right_id_col not in right.columns:
            raise KeyError(f"{right_id_col!r} not in right.columns")

        left_ids = left[left_id_col].to_numpy()
        right_ids = right[right_id_col].to_numpy()

        if len(left_ids) == 0 or len(right_ids) == 0:
            # Decide output ID column names
            if left_id_col == right_id_col:
                out_left_id = f"{left_id_col}_left"
                out_right_id = f"{right_id_col}_right"
            else:
                out_left_id = left_id_col
                out_right_id = right_id_col

            return pd.DataFrame(columns=[out_left_id, out_right_id, match_col])

        # Decide output column names for IDs
        if left_id_col == right_id_col:
            out_left_id = f"{left_id_col}_left"
            out_right_id = f"{right_id_col}_right"
        else:
            out_left_id = left_id_col
            out_right_id = right_id_col

        # full cross join
        df_pairs = pd.DataFrame(
            {
                out_left_id: np.repeat(left_ids, len(right_ids)),
                out_right_id: np.tile(right_ids, len(left_ids)),
            }
        )

        # For lookup (ID → row)
        left_indexed = left.set_index(left_id_col)
        right_indexed = right.set_index(right_id_col)

        # ------------------------------------------------------------------
        # 2. Load similarity functions (built-in + custom)
        # ------------------------------------------------------------------
        sim_funcs = _sim.available_similarities()

        # ------------------------------------------------------------------
        # 3. Compute similarity features only for entries in SimilarityMap
        # ------------------------------------------------------------------
        for lcol, rcol, sim_name in self.similarity_map:
            if sim_name not in sim_funcs:
                raise KeyError(
                    f"Similarity '{sim_name}' not found. "
                    f"Available: {sorted(sim_funcs.keys())}"
                )

            func = sim_funcs[sim_name]
            col_name = f"col_{lcol}_{rcol}_{sim_name}"

            # Dict-based lookup to avoid index alignment quirks
            left_map = left_indexed[lcol].to_dict()
            right_map = right_indexed[rcol].to_dict()

            left_series = df_pairs[out_left_id].map(left_map)
            right_series = df_pairs[out_right_id].map(right_map)

            # Mask where any side is missing
            nan_mask = left_series.isna() | right_series.isna()

            sim_vals = []
            for x, y, is_nan in zip(left_series, right_series, nan_mask):
                if is_nan:
                    # if any NaN involved → similarity 0
                    sim_vals.append(0.0)
                else:
                    sim_vals.append(func(x, y))

            df_pairs[col_name] = sim_vals

        # ------------------------------------------------------------------
        # 4. Match indicator
        # ------------------------------------------------------------------
        if not {matches_id_left, matches_id_right}.issubset(matches.columns):
            raise KeyError(
                f"'matches' must contain columns {matches_id_left!r} and {matches_id_right!r}"
            )

        matches_tmp = matches[[matches_id_left, matches_id_right]].copy()
        matches_tmp[match_col] = 1

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

            df_pairs["_left_pos"] = df_pairs[out_left_id].map(left_id_to_pos)
            df_pairs["_right_pos"] = df_pairs[out_right_id].map(right_id_to_pos)

            df_pairs = df_pairs.merge(
                matches_tmp,
                left_on=["_left_pos", "_right_pos"],
                right_on=[matches_id_left, matches_id_right],
                how="left",
            )

            df_pairs[match_col] = df_pairs[match_col].fillna(0).astype(int)

            # Clean up helper columns
            df_pairs = df_pairs.drop(columns=["_left_pos", "_right_pos", matches_id_left, matches_id_right])

        else:
            # Old behavior: matches[left/right] are in the same ID space
            # as out_left_id / out_right_id
            df_pairs = df_pairs.merge(
                matches_tmp,
                left_on=[out_left_id, out_right_id],
                right_on=[matches_id_left, matches_id_right],
                how="left",
            )
            df_pairs[match_col] = df_pairs[match_col].fillna(0).astype(int)

            if matches_id_left != out_left_id:
                df_pairs = df_pairs.drop(columns=[matches_id_left])
            if matches_id_right != out_right_id:
                df_pairs = df_pairs.drop(columns=[matches_id_right])

        return df_pairs
    
def to_X_y(df: pd.DataFrame, match_col: str = "match"):
    """
    Extract (X, y) from a pairwise similarity DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by AlternativeModels.pairwise_similarity_dataframe().
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
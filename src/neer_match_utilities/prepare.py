from __future__ import annotations

import re
import numpy as np
import pandas as pd
from collections import OrderedDict
from .base import SuperClass
from .custom_similarities import register_similarities
from .split import ORIGINAL_INDEX
import spacy
import random
import string
import uuid
import warnings
import pandas as pd
from neer_match import similarity_map as _sim
from typing import List, Optional, Tuple, Literal


def available_similarities() -> dict:
    """
    Return the currently registered similarity functions.

    Resolves through :mod:`neer_match.similarity_map` on every call so that
    similarities registered at runtime (e.g. by
    :class:`neer_match_utilities.custom_similarities.CustomSimilarities` or
    :class:`Commonness`) are picked up, including the rescaling of fuzz-based
    metrics from ``[0, 100]`` to ``[0, 1]``.

    Returns
    -------
    dict
        Mapping ``{similarity_name: callable}``.
    """
    return _sim.available_similarities()


class Prepare(SuperClass):
    """
    A class for preparing and processing data based on similarity mappings.

    The Prepare class inherits from SuperClass and provides functionality to
    clean, preprocess, and align two pandas DataFrames (`df_left` and `df_right`)
    based on a given similarity map. This is useful for data cleaning and ensuring
    data compatibility before comparison or matching operations.

    Attributes:
    -----------
    similarity_map : dict
        A dictionary defining column mappings between the left and right DataFrames.
    df_left : pandas.DataFrame
        The left DataFrame to be processed.
    df_right : pandas.DataFrame
        The right DataFrame to be processed.
    id_left : str
        Column name representing unique IDs in the left DataFrame.
    id_right : str
        Column name representing unique IDs in the right DataFrame.
    spacy_pipeline : str
        Name of the spaCy model loaded for NLP tasks (e.g., "en_core_web_sm").
        If empty, no spaCy pipeline is used. (see https://spacy.io/models for avaiable models)
    additional_stop_words : list of str
        Extra tokens to mark as stop-words in the spaCy pipeline.
    """


    def __init__(
        self,
        similarity_map: dict,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_left: str,
        id_right: str,
        spacy_pipeline: str = '',
        additional_stop_words: list = [],
    ):
        super().__init__(similarity_map, df_left, df_right, id_left, id_right)
        
        # Load spaCy model once and store for reuse
        # Attempt to load the spaCy model, downloading if necessary
        if spacy_pipeline != '':
            try:
                self.nlp = spacy.load(spacy_pipeline)
            except OSError:
                from spacy.cli import download
                download(spacy_pipeline)
                self.nlp = spacy.load(spacy_pipeline)

            # Register any additional stop words
            self.additional_stop_words = additional_stop_words or []
            for stop in self.additional_stop_words:
                self.nlp.vocab[stop].is_stop = True
                self.nlp.Defaults.stop_words.add(stop)
        else:
            self.nlp = spacy.blank("en")


    def do_remove_stop_words(self, text: str) -> str:
        """
        Removes stop words and non-alphabetic tokens from text.

        Parameters
        ----------
        text : str
            The input text to process.

        Returns
        -------
        str
            A space-separated string of unique lemmas after tokenization, lemmatization,
            and duplicate removal.
        """
        doc = self.nlp(text)
        lemmas = [
            token.lemma_
            for token in doc
            if token.is_alpha and not token.is_stop
        ]

        unique_lemmas = list(dict.fromkeys(lemmas))
        return ' '.join(unique_lemmas)


    @staticmethod
    def _to_number(column: pd.Series, fill_na: bool) -> pd.Series:
        """
        A column as numbers, for ``format``'s ``to_numeric``.

        Numbers stay as they are, signs and exponents included. A string that
        is not a number keeps only its digits and dots ('1.5 km' becomes 1.5);
        one that is still not a number then is missing. With ``fill_na``, the
        missing values become 0.
        """
        numbers = pd.to_numeric(column, errors="coerce")
        if not pd.api.types.is_numeric_dtype(column):
            digits = column.astype("string").str.replace(r"[^\d\.]", "", regex=True)
            numbers = numbers.fillna(pd.to_numeric(digits, errors="coerce"))
        numbers = numbers.astype(float)
        # After the conversion: pandas 3 keeps a missing value missing through
        # astype(str), where pandas 2 made it 'nan' and so a string to fill
        if fill_na:
            numbers = numbers.fillna(0.0)
        return numbers


    def format(
            self, 
            fill_numeric_na: bool = False, 
            to_numeric: list = [], 
            fill_string_na: bool = False, 
            capitalize: bool = False, 
            lower_case: bool = False, 
            remove_stop_words: bool = False,
        ):
        """
        Cleans, processes, and aligns the columns of two DataFrames (`df_left` and `df_right`).

        This method applies transformations based on column mappings defined in `similarity_map`.
        It handles numeric and string conversions, fills missing values, and ensures
        consistent data types between the columns of the two DataFrames.

        Parameters
        ----------
        fill_numeric_na : bool, optional
            If True, the columns in `to_numeric` get `0` where a value is
            missing or no number. Default is False.
        to_numeric : list, optional
            A list of column names to be converted to numbers (float). Numbers
            stay as they are; of a string that is no number only its digits
            and dots are kept ('1.5 km' becomes 1.5).
            Default is an empty list.
        fill_string_na : bool, optional
            If True, fills missing string values with empty strings.
            Default is False.
        capitalize : bool, optional
            If True, capitalizes string values in non-numeric columns.
            Default is False.
        lower_case : bool, optional
            If True, uses lower-case string values in non-numeric columns.
            Default is False.
        remove_stop_words : bool, optional
            If True, applies stop-word removal and lemmatization to non-numeric columns using the `do_remove_stop_words` method.
            Importantly, this only works if a proper Spacy pipeline is defined when initializing the Prepare object.
            Default is False.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame]
            A tuple containing the processed left (`df_left_processed`) and right
            (`df_right_processed`) DataFrames.

        Notes
        -----
        - Columns are processed and aligned according to the `similarity_map`:
            - If both columns are numeric, their types are aligned.
            - If types differ, columns are converted to strings while preserving `NaN`.
        - Supports flexible handling of missing values and type conversions.
        """

        def process_df(df, columns, id_column):
            """
            Clean and process a DataFrame based on specified columns and an ID column.

            This function performs a series of cleaning and transformation steps
            on a DataFrame, including renaming columns, handling missing values,
            converting data types, and optionally capitalizing strings.

            Parameters
            ----------
            df : pd.DataFrame
                The DataFrame to process.
            columns : list of str
                A list of column names to be processed.
            id_column : str
                The name of the ID column to retain in the DataFrame.

            Returns
            -------
            pd.DataFrame
                A cleaned and processed DataFrame.

            Notes
            -----
            - Columns specified in `to_numeric` are converted to numbers, with
              missing values optionally filled with 0 (see `_to_number`).
            - Non-numeric columns are converted to strings, with missing values 
              optionally replaced by empty strings or left as NaN.
            - If `capitalize` is True, string columns are converted to uppercase.
            """

            # Select and rename relevant columns
            df = df[
                [id_column] + [
                re.sub(r'\s', '', col) for col in columns
                ]
            ].copy()


            # Dtype
            for col in columns:
                # Convert to numeric if included in to_numeric argument
                if col in to_numeric:
                    df[col] = self._to_number(df[col], fill_na=fill_numeric_na)

                # If not, convert to string while replacing nans with empty strings
                else:
                    if fill_string_na == True:
                        df[col] = df[col].fillna('').astype(str)
                    else:
                         df[col] = df[col].fillna(np.nan)

            # Remove Stop Words
            if remove_stop_words == True:
                for col in columns:
                    if not col in to_numeric:
                        df[col] = df[col].apply(self.do_remove_stop_words)

            # Capitalize if wished
            if capitalize == True:
                for col in columns:
                    if not col in to_numeric:
                        df[col] = df[col].str.upper()

            # Lower Case if specified
            if lower_case == True:
                for col in columns:
                    if not col in to_numeric:
                        df[col] = df[col].str.lower()

            return df

        # Prepare columns for both DataFrames
        columns_left = list(OrderedDict.fromkeys([
            key.split('~')[0] if '~' in key else key
            for key in self.similarity_map
        ]))

        columns_right = list(OrderedDict.fromkeys([
            key.split('~')[1] if '~' in key else key
            for key in self.similarity_map
        ]))


        # Process both DataFrames
        df_left_processed = process_df(self.df_left, columns_left, self.id_left)
        df_right_processed = process_df(self.df_right, columns_right, self.id_right)

        # Ensure matched columns have the same dtype
        for key in self.similarity_map:
            cl, cr = (key.split('~') + [key])[:2]  # Handles both cases where '~' exists or not
            if df_left_processed[cl].dtype != df_right_processed[cr].dtype:
                # Check if both are numeric
                if pd.api.types.is_numeric_dtype(df_left_processed[cl]) and pd.api.types.is_numeric_dtype(df_right_processed[cr]):
                    # Align numeric types (e.g., float over int if needed)
                    if pd.api.types.is_integer_dtype(df_left_processed[cl]) and pd.api.types.is_float_dtype(df_right_processed[cr]):
                        df_left_processed[cl] = df_left_processed[cl].astype(float)
                    elif pd.api.types.is_float_dtype(df_left_processed[cl]) and pd.api.types.is_integer_dtype(df_right_processed[cr]):
                        df_right_processed[cr] = df_right_processed[cr].astype(float)
                    # Both are numeric and no conversion needed beyond alignment
                else:
                    # Convert both to string if types don't match
                    df_left_processed[cl] = df_left_processed[cl].apply(lambda x: str(x) if pd.notna(x) else x)
                    df_right_processed[cr] = df_right_processed[cr].apply(lambda x: str(x) if pd.notna(x) else x)

        return df_left_processed, df_right_processed


class Commonness:
    def __init__(
        self,
        variable_list: List[str],
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        df_left_full: pd.DataFrame,
        df_right_full: pd.DataFrame,
        commonness_source: Literal["left", "right", "both"] = "left",
        *,
        scoring: Literal["relative", "minmax", "log"] = "minmax",
        scale: Literal["quantile", "percentile"] = "quantile",  # quantile: 0..1, percentile: 0..100
        fill_value: float = 0.0,
        preprocess: bool = False,
    ):
        """
        Initialize a Commonness helper and extend available similarity functions.

        This initializer stores references to the input DataFrames and the
        configuration for how to compute "commonness" scores, and registers a
        custom similarity function named `"commonness_score"` with neer_match.

        Parameters
        ----------
        variable_list : list[str]
            Names of columns for which commonness scores should be computed and
            appended to `df_left` and `df_right` (as `<col>_commonness`).
        df_left : pandas.DataFrame
            Left DataFrame whose selected columns will receive commonness scores.
        df_right : pandas.DataFrame
            Right DataFrame whose selected columns will receive commonness scores.
        df_left_full : pandas.DataFrame
            Full left DataFrame used as a source for frequency estimation when
            `commonness_source` is `"left"` or `"both"`.
        df_right_full : pandas.DataFrame
            Full right DataFrame used as a source for frequency estimation when
            `commonness_source` is `"right"` or `"both"`.
        commonness_source : {"left","right","both"}, optional
            Which corpus to use for computing frequencies: the left full dataset,
            the right full dataset, or the concatenation of both. Default is "left".
        scoring : {"relative","minmax","log"}, optional
            Strategy to convert raw value counts to a [0,1] commonness score:
            - "relative": count / total (rare values near 0);
            - "minmax": (count - min)/(max - min) with guard for equal counts;
            - "log": log1p(count)/log1p(max_count).
            Default is "minmax".
        scale : {"quantile","percentile"}, optional
            Reserved for future scaling behavior; currently unused. Default "quantile".
        fill_value : float, optional
            Value used when a category in `df_left`/`df_right` is not present in
            the chosen frequency source. Default is 0.0.
        preprocess : bool, optional
            If True, string values are normalized (strip & lowercase) before
            counting and mapping. Default is False.

        Notes
        -----
        - `"commonness_score"` is registered through
          :func:`neer_match_utilities.custom_similarities.register_similarities`,
          next to the similarities added by `CustomSimilarities`. Constructing
          either one keeps what the other added, including the rescaling of the
          fuzz-based similarities to [0, 1].
        """
        self.variable_list = list(variable_list)
        self.df_left = df_left.copy()
        self.df_right = df_right.copy()
        self.df_left_full = df_left_full
        self.df_right_full = df_right_full
        self.commonness_source = commonness_source
        self.scoring = scoring
        self.scale = scale
        self.fill_value = fill_value
        self.preprocess = preprocess

        if self.commonness_source not in {"left", "right", "both"}:
            raise ValueError(
                "Argument `commonness_source` must be one of {'left','right','both'}."
            )

        # ---- Make "commonness_score" available to neer_match ----
        register_similarities({"commonness_score": self.commonness_score})

    @staticmethod
    def _prep_series(s: pd.Series, preprocess: bool) -> pd.Series:
        """
        Normalize a Series for robust frequency counting and mapping.

        Drops missing values and, if requested, normalizes textual content by
        stripping surrounding whitespace and lowercasing.

        Parameters
        ----------
        s : pandas.Series
            Input Series to clean.
        preprocess : bool
            If True, cast to string, strip, and lowercase values.

        Returns
        -------
        pandas.Series
            Cleaned Series (with NA removed; optionally normalized).
        """
        # Drop NA; optionally normalize for robust matching
        s = s.dropna()
        if preprocess:
            s = s.astype(str).str.strip().str.lower()
        return s

    @staticmethod
    def _frequency_score(series: pd.Series, scoring: str = "minmax") -> dict:
        """
        Convert value frequencies into a [0,1] commonness score mapping.

        Counts unique values in `series` and transforms raw counts into scores
        according to the chosen `scoring` strategy.

        Parameters
        ----------
        series : pandas.Series
            Input data used to compute value frequencies.
        scoring : {"relative","minmax","log"}, optional
            Scoring strategy:
            - "relative": score = count / total_count
            - "minmax":  score = (count - min_count) / (max_count - min_count)
                         (returns 1.0 for all values if max_count == min_count)
            - "log":     score = log1p(count) / log1p(max_count)
            Default is "minmax".

        Returns
        -------
        dict
            Mapping `{value: score}` with scores in [0,1].

        Notes
        -----
        - Returns an empty dict if `series` has no non-NA values.
        - For "minmax" with equal counts, all values receive 1.0 to avoid
          division by zero and to indicate no variation in frequency.
        """
        vc = series.value_counts()           # index: value, values: counts
        if vc.empty:
            return {}

        if scoring == "relative":
            scores = vc / vc.sum()          # harshest on singletons
        elif scoring == "minmax":
            cmin, cmax = vc.min(), vc.max()
            if cmax == cmin:                # all counts equal
                scores = pd.Series(1.0, index=vc.index)
            else:
                scores = (vc - cmin) / (cmax - cmin)
        elif scoring == "log":
            cmax = vc.max()
            if cmax == 0:
                scores = pd.Series(0.0, index=vc.index)
            else:
                scores = np.log1p(vc) / np.log1p(cmax)
        else:
            raise ValueError("scoring must be one of {'relative','minmax','log'}")

        return scores.to_dict()

    def calculate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute and append commonness scores for configured variables.

        For each variable in `variable_list`, this method:
        1) builds the frequency source per `commonness_source`,
        2) computes a `{value: score}` mapping via `_frequency_score`,
        3) maps scores to `df_left[v]` and `df_right[v]`, producing
           `<v>_commonness` columns in both frames.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame]
            The augmented left and right DataFrames with added
            `<variable>_commonness` columns.

        Notes
        -----
        - Unseen values (not present in the chosen frequency source) are filled
          with `fill_value`.
        - If `preprocess=True`, normalization (strip/lower) is applied both
          when computing frequencies and when mapping, to ensure consistency.
        """
        for v in self.variable_list:
            # Build source pool for frequency ranks
            if self.commonness_source == "left":
                source = self.df_left_full[v]
            elif self.commonness_source == "right":
                source = self.df_right_full[v]
            else:  # both
                source = pd.concat([self.df_left_full[v], self.df_right_full[v]], ignore_index=True)

            source = self._prep_series(source, self.preprocess)

            # Build mapping dict {value: quantile/percentile rank}
            mapping = self._frequency_score(source, self.scoring)

            # Map to both frames (prep their columns the same way for matching)
            left_col = self._prep_series(self.df_left[v], self.preprocess)
            right_col = self._prep_series(self.df_right[v], self.preprocess)

            # To preserve original values but map by normalized view:
            # create temporary normalized columns for mapping, then drop
            self.df_left[v + "_commonness"] = left_col.map(mapping).reindex(self.df_left.index).fillna(self.fill_value)
            self.df_right[v + "_commonness"] = right_col.map(mapping).reindex(self.df_right.index).fillna(self.fill_value)

        return self.df_left, self.df_right

    @staticmethod
    def commonness_score(x: float, y: float) -> float:
        """
        Commonness-aware similarity in [0,1] favoring rare-and-equal matches.

        The score rewards both *closeness* and *rareness*, defined on inputs
        `x, y ∈ [0,1]` (interpreted as commonness scores). Identical rare values
        receive high score; identical common values receive low score.

        Formally:
            diff = |x - y|
            mean = (x + y)/2
            score = (1 - diff) * (1 - mean)

        Parameters
        ----------
        x : float
            Commonness score of the left item, expected in [0,1].
        y : float
            Commonness score of the right item, expected in [0,1].

        Returns
        -------
        float
            Similarity value in [0,1].

        Examples
        --------
        - x = y = 0    → score = 1.0  (rare & equal)
        - x = y = 1    → score = 0.0  (common & equal)
        - x = 0, y = 1 → score = 0.0  (maximally different)
        - x = y = 0.5  → score = 0.5
        """
        x, y = float(x), float(y)
        diff = abs(x - y)
        mean = (x + y) / 2
        # closeness = 1 - diff
        # rarity factor = 1 - mean  (rare=1, common=0)
        return (1.0 - diff) * (1.0 - mean)


def similarity_map_to_dict(items: list) -> dict:
    """
    Convert a list of similarity mappings into a dictionary representation.

    The function accepts a list of tuples, where each tuple represents a mapping
    with the form `(left, right, similarity)`. If the left and right column names
    are identical, the dictionary key is that column name; otherwise, the key is formed
    as `left~right`.

    Returns
    -------
    dict
        A dictionary where keys are column names (or `left~right` for differing columns)
        and values are lists of similarity functions associated with those columns.
    """
    result = {}
    for left, right, similarity in items:
        # Use the left value as key if both columns are identical; otherwise, use 'left~right'
        key = left if left == right else f"{left}~{right}"
        if key in result:
            result[key].append(similarity)
        else:
            result[key] = [similarity]
    return result


def synth_mismatches(
    right: pd.DataFrame,
    columns_fix: List[str],
    columns_change: List[str],
    str_metric: str,
    str_similarity_range: Tuple[float, float],
    pct_diff_range: Tuple[float, float],
    n_cols: int,
    n_mismatches: int = 1,
    keep_missing: bool = True,
    nan_share: float = 0.0,
    empty_share: float = 0.0,
    sample_share: float = 1.0,
    id_right: Optional[str] = None,
    random_state: Optional[int] = None,
    candidate_weights: Literal["uniform", "frequency"] = "uniform",
    variants: Optional[dict] = None,
    fallback: Literal["random", "skip"] = "random",
) -> pd.DataFrame:
    """
    Generate synthetic mismatches for a subset of rows in ``right``.

    The output contains all original rows plus additional synthetic rows created by
    modifying selected columns. Synthetic rows are deduplicated against the original
    data and against each other.

    A synthetic row is not in ``matches``, so it counts as a non-match of the left
    record its source row is matched to: a hard non-match if the change is small.
    Apply the function to the training subset after ``split_test_train`` (which
    drops records without a match), and change only columns that identify the
    entity: a synthetic row that differs from its source only in a recorded
    variant of the same value (e.g. "Johann" for "Johannes") is the same entity,
    and labelling it a non-match teaches the model the wrong thing. ``variants``
    excludes such replacements.

    Parameters
    ----------
    right : pandas.DataFrame
        The DataFrame containing the original (true) observations.
    columns_fix : list of str
        Column names whose values remain unchanged (copied from the original row).
    columns_change : list of str
        Column names whose values are modified to create mismatches.
    str_metric : str
        Name of the string-similarity metric (key in ``available_similarities()``).
    str_similarity_range : tuple of (float, float)
        Allowed range ``(min_str_sim, max_str_sim)`` for normalized string similarity
        of STRING columns in ``columns_change``. Candidates must satisfy
        ``min_str_sim <= similarity(orig, candidate) <= max_str_sim``.
    pct_diff_range : tuple of (float, float)
        Allowed range ``(min_pct_diff, max_pct_diff)`` for percentage difference of
        NUMERIC columns in ``columns_change``: ``abs(orig - candidate) / abs(orig)``.
        If ``orig == 0``, any ``candidate != 0`` is treated as percentage difference 1.0.
    n_cols : int
        Number of columns from ``columns_change`` to modify per synthetic row.
        If ``n_cols < len(columns_change)``, pick that many at random, once per
        synthetic row. If ``n_cols > len(columns_change)``, all columns in
        ``columns_change`` are modified.
    n_mismatches : int, default=1
        Number of synthetic mismatches to generate per selected original row.
    keep_missing : bool, default=True
        If True, preserve ``NaN`` or empty-string values in ``columns_change`` of the
        original row (no change applied to those cells).
    nan_share : float, default=0.0
        After deduplication, probability to inject ``NaN`` into each synthetic cell of
        ``columns_change``.
    empty_share : float, default=0.0
        After deduplication, probability to inject ``""`` into each synthetic cell of
        ``columns_change``. Applied after ``nan_share``.
    sample_share : float, default=1.0
        Proportion in ``[0, 1]`` of original rows in ``right`` to select at random
        for synthetic generation. For example, ``0.5`` selects ``floor(0.5 * n_rows)``.
    id_right : str or None, default=None
        Name of a unique-ID column in ``right``. If provided, synthetic rows receive
        new UUID4 IDs in this column. If None, no ID column is created or modified.
    random_state : int or None, default=None
        Seed for all random draws of this call. If None, the global generators of
        ``random`` and ``numpy.random`` are used, as before.
    candidate_weights : {"uniform", "frequency"}, default="uniform"
        How a replacement is drawn from the qualifying values of a column:
        uniformly, or in proportion to how often each value occurs in ``right``,
        so that common names are drawn as often as they occur.
    variants : dict or None, default=None
        Known variants of the same value, ``{column: {value: set_of_variants}}``,
        e.g. from :func:`learn_variants`. A listed variant of the original value
        is never used as its replacement.
    fallback : {"random", "skip"}, default="random"
        What happens if no value of a column qualifies: "random" perturbs the
        original (random letters for strings), "skip" leaves the cell unchanged.


    Returns
    -------
    pandas.DataFrame
        Expanded DataFrame with the original rows plus the synthetic mismatch rows.
        The original rows keep their row labels; with an integer index the
        synthetic rows are numbered after the largest label, so that pairs that
        refer to the row labels stay valid. The column ``index_original`` added
        by ``split_test_train`` is missing in synthetic rows.

    Notes
    -----
    - STRING columns in ``columns_change`` must meet the configured string-similarity
      range. If no candidate qualifies, the original string is perturbed until the
      similarity lies within the requested bounds.
    - NUMERIC columns in ``columns_change`` must meet the configured percentage
      difference range. If no candidate qualifies, the value is perturbed toward
      the boundary (e.g., ``orig * (1 ± min_pct_diff)`` or ``orig * (1 ± max_pct_diff)``).
    - After generating synthetics, any synthetic row whose modified data portion
      exactly matches an original row (or another synthetic row) is dropped.
    """

    # Validate shares and ranges
    min_str_sim, max_str_sim = str_similarity_range
    min_pct_diff, max_pct_diff = pct_diff_range
    if not (0 <= min_str_sim <= max_str_sim <= 1):
        raise ValueError("str_similarity_range must satisfy 0 ≤ min ≤ max ≤ 1.")
    if not (0 <= min_pct_diff <= max_pct_diff <= 1):
        raise ValueError("pct_diff_range must satisfy 0 ≤ min ≤ max ≤ 1.")
    if not (0 <= nan_share <= 1 and 0 <= empty_share <= 1):
        raise ValueError("nan_share and empty_share must be between 0 and 1.")
    if nan_share + empty_share > 1:
        raise ValueError("nan_share + empty_share must be ≤ 1.0.")
    if not (0 <= sample_share <= 1):
        raise ValueError("sample_share must be between 0 and 1.")
    if candidate_weights not in ("uniform", "frequency"):
        raise ValueError("candidate_weights must be 'uniform' or 'frequency'.")
    if fallback not in ("random", "skip"):
        raise ValueError("fallback must be 'random' or 'skip'.")
    variants = variants or {}

    # Random generators: local ones if seeded, else the global ones (earlier behaviour)
    if random_state is None:
        rnd, np_rng, sample_state = random, np.random, None
    else:
        rnd = random.Random(random_state)
        np_rng = np.random.default_rng(random_state)
        sample_state = np_rng

    # Validate n_cols vs. columns_change length
    if n_cols > len(columns_change):
        warnings.warn(
            f"Requested n_cols={n_cols} > len(columns_change)={len(columns_change)}. "
            "All columns in columns_change will be modified."
        )
        n_cols_effective = len(columns_change)
    else:
        n_cols_effective = n_cols

    # Grab the string‐similarity function
    sim_funcs = available_similarities()
    if str_metric not in sim_funcs:
        raise ValueError(f"String metric '{str_metric}' not found in available_similarities().")
    str_sim = sim_funcs[str_metric]

    # Build final column list: include any columns_fix or columns_change not already in right
    final_columns = list(right.columns)
    for col in set(columns_fix + columns_change):
        if col not in final_columns:
            final_columns.append(col)

    # Prepare a working copy of original right, adding missing columns with NaN
    original_right = right.copy(deep=True)
    for col in final_columns:
        if col not in original_right.columns:
            original_right[col] = np.nan

    # Determine subset of rows to generate synthetics for
    n_original = len(original_right)
    if sample_share < 1.0:
        n_to_sample = int(np.floor(sample_share * n_original))
        sampled_idx = (
            original_right
            .sample(n=n_to_sample, random_state=sample_state)
            .index
            .tolist()
        )
    else:
        sampled_idx = original_right.index.tolist()

    # Track existing IDs (as strings), to avoid collisions
    existing_ids = set()
    if id_right:
        existing_ids = set(original_right[id_right].astype(str).tolist())

    # Precompute candidate pools for columns_change from original right
    candidate_pools = {}
    candidate_counts = {}
    for col in columns_change:
        if col in original_right.columns:
            candidate_pools[col] = original_right[col].dropna().unique().tolist()
            candidate_counts[col] = original_right[col].value_counts().to_dict()
        else:
            candidate_pools[col] = []
            candidate_counts[col] = {}

    # Helper: percentage‐based numeric filter
    def pct_diff(orig: float, candidate: float) -> float:
        """
        Returns │orig - candidate│/│orig│ if orig != 0; else returns 1.0 if candidate != 0, 0.0 if candidate == 0.
        """
        try:
            if orig == 0:
                return 0.0 if candidate == 0 else 1.0
            else:
                return abs(orig - candidate) / abs(orig)
        except Exception:
            return 0.0

    # Helper: perturb string until similarity is in [min_str_sim, max_str_sim]
    def _perturb_string(orig: str) -> str:
        length = len(orig) if (isinstance(orig, str) and len(orig) > 0) else 1
        attempts = 0
        while attempts < 50:
            candidate = "".join(rnd.choices(string.ascii_letters, k=length))
            try:
                sim = str_sim(orig, candidate)
            except Exception:
                sim = 1.0
            if min_str_sim <= sim <= max_str_sim:
                return candidate
            attempts += 1
        return candidate  # last attempt if none succeeded

    # Helper: perturb numeric until pct_diff in [min_pct_diff, max_pct_diff]
    def _perturb_numeric(orig: float, col_series: pd.Series) -> float:
        """
        Generate a numeric candidate until pct_diff(orig, candidate) in [min_pct_diff, max_pct_diff]
        (50 attempts). If none succeed, explicitly create orig*(1 ± min_pct_diff) or orig*(1 ± max_pct_diff).
        """
        attempts = 0
        col_std = col_series.std() if (col_series.std() > 0) else 1.0
        while attempts < 50:
            candidate = orig + np_rng.normal(loc=0.0, scale=col_std)
            pdiff = pct_diff(orig, candidate)
            if min_pct_diff <= pdiff <= max_pct_diff:
                return candidate
            attempts += 1

        # Fallback: choose exactly at boundaries
        if orig == 0:
            nonzeros = [v for v in col_series if v != 0]
            return nonzeros[0] if nonzeros else 0.0

        low_val_min = orig * (1 - min_pct_diff)
        high_val_min = orig * (1 + min_pct_diff)
        low_val_max = orig * (1 - max_pct_diff)
        high_val_max = orig * (1 + max_pct_diff)

        # Prefer a value within the narrower range if possible
        # Try orig*(1 + min_pct_diff)
        return high_val_min

    # Build synthetic rows only for sampled_idx
    synthetic_rows = []
    for idx in sampled_idx:
        orig_row = original_right.loc[idx]
        for _ in range(n_mismatches):
            new_row = {}
            # The columns to modify in this synthetic row, drawn once per row
            chosen = set(rnd.sample(columns_change, n_cols_effective))
            for col in final_columns:
                if id_right and col == id_right:
                    new_row[col] = None  # assign later
                    continue

                orig_val = orig_row.get(col, np.nan)

                if col in columns_change:
                    # Decide if we modify this column in this synthetic row
                    if col in chosen:
                        # If keep_missing=True and orig_val is NaN or "", preserve it
                        if keep_missing and (pd.isna(orig_val) or (isinstance(orig_val, str) and orig_val == "")):
                            new_row[col] = orig_val
                        else:
                            same = _known_variants(variants, col, orig_val)
                            pool = [v for v in candidate_pools[col] if v != orig_val and v not in same]

                            # Filter pool by string or percentage‐difference range
                            filtered = []
                            if pd.isna(orig_val):
                                filtered = pool.copy()
                            else:
                                for v in pool:
                                    try:
                                        if isinstance(orig_val, str) or isinstance(v, str):
                                            sim = str_sim(str(orig_val), str(v))
                                            if min_str_sim <= sim <= max_str_sim:
                                                filtered.append(v)
                                        else:
                                            pdiff = pct_diff(float(orig_val), float(v))
                                            if min_pct_diff <= pdiff <= max_pct_diff:
                                                filtered.append(v)
                                    except:
                                        continue

                            if filtered:
                                if candidate_weights == "frequency":
                                    weights = [candidate_counts[col].get(v, 1) for v in filtered]
                                    new_row[col] = rnd.choices(filtered, weights=weights, k=1)[0]
                                else:
                                    new_row[col] = rnd.choice(filtered)
                            elif fallback == "skip":
                                new_row[col] = orig_val
                            else:
                                # Fallback: perturb orig_val
                                if pd.isna(orig_val):
                                    new_row[col] = orig_val
                                elif isinstance(orig_val, str):
                                    new_row[col] = _perturb_string(orig_val)
                                elif isinstance(orig_val, (int, float, np.integer, np.floating)):
                                    combined = (
                                        original_right[col].dropna()
                                        if col in original_right.columns
                                        else pd.Series(dtype="float64")
                                    )
                                    new_row[col] = _perturb_numeric(float(orig_val), combined)
                                else:
                                    new_row[col] = orig_val
                    else:
                        # Not chosen for change: copy original
                        new_row[col] = orig_val

                elif col in columns_fix:
                    # Copy original unchanged
                    new_row[col] = orig_val

                else:
                    # Neither fix nor change: copy original if present, else NaN
                    new_row[col] = orig_val

            # Assign a new UUID4 for id_right, if specified
            if id_right:
                new_id = str(uuid.uuid4())
                while new_id in existing_ids:
                    new_id = str(uuid.uuid4())
                new_row[id_right] = new_id
                existing_ids.add(new_id)

            synthetic_rows.append(new_row)

    # Build DataFrame of synthetic candidates
    if not synthetic_rows:
        return original_right

    df_new = pd.DataFrame(synthetic_rows, columns=final_columns)

    # Cast newly generated numeric columns back to original dtype (handling NaNs)
    for col in columns_change:
        if col in right.columns:
            orig_dtype = right[col].dtype
            if pd.api.types.is_integer_dtype(orig_dtype):
                # If any NaNs present, use pandas nullable Int64; otherwise cast to original int
                if df_new[col].isna().any():
                    df_new[col] = df_new[col].round().astype("Int64")
                else:
                    df_new[col] = df_new[col].round().astype(orig_dtype)
            elif pd.api.types.is_float_dtype(orig_dtype):
                df_new[col] = df_new[col].astype(orig_dtype)

    # Drop duplicates (by all columns except id_right, if specified, and the
    # original row labels that split_test_train adds)
    data_columns = [c for c in final_columns if c not in (id_right, ORIGINAL_INDEX)]
    original_data = original_right[data_columns].drop_duplicates().reset_index(drop=True)

    df_new_data = df_new[data_columns].reset_index()

    # Force object dtype for safe merging
    original_data_obj = original_data.astype(object)
    for c in data_columns:
        df_new_data[c] = df_new_data[c].astype(object)

    # Duplicate with original right?
    dup_with_right = df_new_data.merge(
        original_data_obj.assign(_flag=1),
        on=data_columns,
        how="left"
    )["_flag"].fillna(0).astype(bool)

    # Duplicate within synthetic set?
    dup_within_new = df_new_data.duplicated(subset=data_columns, keep=False)

    to_drop = dup_with_right | dup_within_new
    drop_indices = df_new_data.loc[to_drop, "index"].tolist()

    if drop_indices:
        df_new_filtered = df_new.drop(index=drop_indices).reset_index(drop=True)
    else:
        df_new_filtered = df_new

    # Inject random NaNs and empty strings into columns_change
    if nan_share > 0 or empty_share > 0:
        for col in columns_change:
            df_new_filtered[col] = df_new_filtered[col].astype(object)
            rand_vals = np_rng.random(len(df_new_filtered))
            nan_mask = rand_vals < nan_share
            empty_mask = (rand_vals >= nan_share) & (rand_vals < nan_share + empty_share)
            df_new_filtered.loc[nan_mask, col] = np.nan
            df_new_filtered.loc[empty_mask, col] = ""

    # Concatenate the filtered synthetic rows onto original_right
    if not df_new_filtered.empty:
        result = _append_synthetic(original_right, df_new_filtered)
    else:
        result = original_right.copy()

    return result


def _known_variants(variants: dict, column: str, value) -> set:
    """The listed variants of `value` in `column` (empty if none or if `value` is missing)."""
    if not variants or column not in variants:
        return set()
    try:
        if pd.isna(value):
            return set()
    except (TypeError, ValueError):
        pass
    return set(variants[column].get(value, ()))


def _append_synthetic(original: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Appends synthetic rows to `original` without changing its row labels.

    With an integer index, the new rows are numbered after the largest label;
    otherwise all rows are renumbered from 0 (with a warning), as before. The
    original row labels of `split_test_train` (``index_original``) are set to
    missing in the new rows, so that they cannot be taken for original records.
    """
    new = new.copy()
    if ORIGINAL_INDEX in new.columns:
        new[ORIGINAL_INDEX] = np.nan
    if len(original) == 0 or pd.api.types.is_integer_dtype(original.index):
        start = int(original.index.max()) + 1 if len(original) else 0
        new.index = pd.RangeIndex(start, start + len(new))
        out = pd.concat([original, new])
    else:
        warnings.warn(
            "right has a non-integer index: all rows are renumbered from 0, so pairs "
            "that refer to its row labels no longer apply."
        )
        out = pd.concat([original, new], ignore_index=True)
    if ORIGINAL_INDEX in original.columns and pd.api.types.is_integer_dtype(original[ORIGINAL_INDEX]):
        out[ORIGINAL_INDEX] = out[ORIGINAL_INDEX].astype("Int64")
    return out


# Characters that OCR of printed (incl. Fraktur) sources confuses; each pair is
# used in both directions and in lower and upper case (data are often
# capitalized after OCR, so "Horn" read as "Hom" appears as "HOM").
OCR_CONFUSIONS: Tuple[Tuple[str, str], ...] = (
    ("rn", "m"), ("cl", "d"), ("li", "h"), ("vv", "w"), ("ii", "ü"),
    ("u", "n"), ("c", "e"), ("o", "c"), ("h", "b"), ("s", "f"), ("ſ", "f"),
    ("i", "l"), ("l", "1"), ("o", "0"), ("ü", "u"), ("ä", "a"), ("ö", "o"),
    ("å", "a"), ("ß", "ss"),
)


def _directed_confusions(pairs) -> List[Tuple[str, str]]:
    """Both directions of each pair, in lower and upper case, without no-op or repeated pairs."""
    out = []
    for a, b in pairs:
        for x, y in ((a, b), (b, a), (a.upper(), b.upper()), (b.upper(), a.upper())):
            if x != y and (x, y) not in out:
                out.append((x, y))
    return out


def _corrupt_ocr(value: str, confusions, rnd) -> Optional[str]:
    """Replaces one occurrence of a confusable character sequence, or None if none occurs."""
    spots = [
        (i, a, b)
        for a, b in confusions
        for i in range(len(value) - len(a) + 1)
        if value.startswith(a, i)
    ]
    if not spots:
        return None
    i, a, b = rnd.choice(spots)
    return value[:i] + b + value[i + len(a):]


def _corrupt_typo(value: str, rnd) -> Optional[str]:
    """One typing or transcription slip: a character dropped, doubled, or two neighbours swapped."""
    if len(value) < 3:
        return None
    ops = ["drop", "double"] + (["swap"] if any(value[i] != value[i + 1] for i in range(len(value) - 1)) else [])
    op = rnd.choice(ops)
    if op == "swap":
        i = rnd.choice([i for i in range(len(value) - 1) if value[i] != value[i + 1]])
        return value[:i] + value[i + 1] + value[i] + value[i + 2:]
    i = rnd.randrange(len(value))
    return value[:i] + value[i + 1:] if op == "drop" else value[:i] + value[i] + value[i:]


def learn_variants(
    left: pd.DataFrame,
    right: pd.DataFrame,
    matches: pd.DataFrame,
    columns,
    min_count: int = 1,
) -> dict:
    """
    Collect the variants of a value that occur in true matches.

    For every match, the values of a column in the left and the right record are
    compared; if they differ (e.g. "JOHANN" and "JOHANNES", or "CARL" and "KARL"),
    each is recorded as a variant of the other. The result tells which
    replacements keep the identity of a record: :func:`synth_matches` uses them
    to build synthetic matches, :func:`synth_mismatches` excludes them when it
    builds synthetic non-matches. Learn them from the training matches only, so
    that nothing of the test subset enters the training data.

    Parameters
    ----------
    left, right : pandas.DataFrame
        The records; ``matches`` refers to their row labels.
    matches : pandas.DataFrame
        True pairs, with columns 'left' and 'right'.
    columns : list of str or dict
        Columns to compare: names present in both ``left`` and ``right``, or a
        mapping ``{right_column: left_column}``.
    min_count : int, default=1
        Keep a pair of values only if it occurs in at least this many matches.

    Returns
    -------
    dict
        ``{right_column: {value: set_of_variants}}``, symmetric (if b is a variant
        of a, a is a variant of b).
    """
    pairs = columns if isinstance(columns, dict) else {c: c for c in columns}
    l_rows = left.loc[matches["left"]]
    r_rows = right.loc[matches["right"]]
    out = {}
    for r_col, l_col in pairs.items():
        counts = {}
        for a, b in zip(l_rows[l_col].to_numpy(), r_rows[r_col].to_numpy()):
            if pd.isna(a) or pd.isna(b) or a == b or a == "" or b == "":
                continue
            key = (a, b) if str(a) <= str(b) else (b, a)
            counts[key] = counts.get(key, 0) + 1
        table = {}
        for (a, b), n in counts.items():
            if n >= min_count:
                table.setdefault(a, set()).add(b)
                table.setdefault(b, set()).add(a)
        out[r_col] = table
    return out


def synth_matches(
    right: pd.DataFrame,
    matches: pd.DataFrame,
    columns_change: List[str],
    n_cols: int = 1,
    n_per_match: int = 1,
    sample_share: float = 1.0,
    corruptions: Tuple[str, ...] = ("ocr", "typo", "variant"),
    variants: Optional[dict] = None,
    ocr_confusions: Tuple[Tuple[str, str], ...] = OCR_CONFUSIONS,
    max_shift: int = 1,
    id_right: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic matches: copies of matched right records with errors that
    keep their identity.

    For a share of the matches (left, right), a copy of the right record is made
    in which ``n_cols`` columns of ``columns_change`` are corrupted, and the pair
    (left, copy) is added to the matches. The corruptions imitate how one entity
    is recorded differently in two sources:

    - ``"ocr"``: one OCR confusion, e.g. "rn" read as "m" or "ü" as "u"
      (``ocr_confusions``);
    - ``"typo"``: one character dropped, doubled, or swapped with its neighbour;
    - ``"variant"``: the value replaced by a variant recorded for it in
      ``variants`` (e.g. from :func:`learn_variants`), e.g. "JOHANNES" for "JOHANN";
    - ``"shift"``: a numeric value moved by up to ``max_shift`` (e.g. a birth
      year misreported by one year);
    - ``"missing"``: the value not recorded.

    The label "match" is only right if the corruption keeps the identity, so
    choose ``columns_change`` and ``corruptions`` accordingly (e.g. "shift" for a
    birth year, not for a parish code). Apply the function to the training
    subset after ``split_test_train``, never to the test subset.

    Parameters
    ----------
    right : pandas.DataFrame
        The right records; ``matches`` refers to their row labels.
    matches : pandas.DataFrame
        True pairs, with columns 'left' and 'right'.
    columns_change : list of str
        Columns that may be corrupted.
    n_cols : int, default=1
        Number of columns corrupted per synthetic record. Columns where no
        selected corruption applies (e.g. a missing value) are not counted; if
        fewer than ``n_cols`` qualify, all of them are corrupted.
    n_per_match : int, default=1
        Number of synthetic records per selected match.
    sample_share : float, default=1.0
        Share of the matches, drawn at random, for which synthetic records are made.
    corruptions : tuple of str, default=("ocr", "typo", "variant")
        Corruptions to draw from, among "ocr", "typo", "variant", "shift" and
        "missing". For each corrupted cell, one of those that apply is drawn.
    variants : dict or None, default=None
        ``{column: {value: set_of_variants}}`` for the corruption "variant".
    ocr_confusions : tuple of (str, str), default=OCR_CONFUSIONS
        Confusable character sequences, used in both directions and in lower and
        upper case.
    max_shift : int, default=1
        Largest change of a numeric value for the corruption "shift".
    id_right : str or None, default=None
        Name of a unique-ID column in ``right``; synthetic records get new UUID4 IDs.
    random_state : int or None, default=None
        Seed for all random draws of this call.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        ``right`` with the synthetic records appended, and ``matches`` with their
        pairs appended. The original rows keep their row labels; with an integer
        index the synthetic records are numbered after the largest label. A
        synthetic record that equals an existing record, or another synthetic
        one, is dropped. The column ``index_original`` added by
        ``split_test_train`` is missing in synthetic records.
    """
    known = {"ocr", "typo", "variant", "shift", "missing"}
    unknown = set(corruptions) - known
    if unknown:
        raise ValueError(f"Unknown corruptions {sorted(unknown)}; choose among {sorted(known)}.")
    if not (0 <= sample_share <= 1):
        raise ValueError("sample_share must be between 0 and 1.")
    if n_cols < 1 or n_per_match < 1:
        raise ValueError("n_cols and n_per_match must be at least 1.")
    missing_cols = [c for c in columns_change if c not in right.columns]
    if missing_cols:
        raise ValueError(f"Columns not in right: {missing_cols}")
    if "variant" in corruptions and not variants:
        warnings.warn("corruption 'variant' needs variants (e.g. from learn_variants); it is skipped.")
    variants = variants or {}
    confusions = _directed_confusions(ocr_confusions)
    rnd = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)

    def options(col, value) -> List[str]:
        """The corruptions that apply to this value."""
        if pd.isna(value) or (isinstance(value, str) and value == ""):
            return []
        out = []
        if isinstance(value, str):
            if "ocr" in corruptions and any(a in value for a, _ in confusions):
                out.append("ocr")
            if "typo" in corruptions and len(value) >= 3:
                out.append("typo")
        elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            if "shift" in corruptions and max_shift >= 1:
                out.append("shift")
        if "variant" in corruptions and _known_variants(variants, col, value):
            out.append("variant")
        if "missing" in corruptions:
            out.append("missing")
        return out

    def corrupt(col, value, how):
        if how == "ocr":
            return _corrupt_ocr(value, confusions, rnd)
        if how == "typo":
            return _corrupt_typo(value, rnd)
        if how == "variant":
            return rnd.choice(sorted(_known_variants(variants, col, value), key=str))
        if how == "shift":
            step = rnd.choice([s for s in range(-max_shift, max_shift + 1) if s != 0])
            return value + step
        return np.nan  # "missing"

    n_sample = int(np.floor(sample_share * len(matches)))
    chosen_matches = matches.sample(n=n_sample, random_state=np_rng) if n_sample < len(matches) else matches

    def key(row) -> tuple:
        return tuple(None if (not isinstance(v, str) and pd.isna(v)) else v for v in row)

    data_columns = [c for c in right.columns if c not in (id_right, ORIGINAL_INDEX)]
    seen = {key(r) for r in right[data_columns].itertuples(index=False, name=None)}
    existing_ids = set(right[id_right].astype(str)) if id_right else set()

    new_rows, new_left = [], []
    for l_id, r_id in zip(chosen_matches["left"], chosen_matches["right"]):
        source = right.loc[r_id]
        for _ in range(n_per_match):
            usable = {c: options(c, source[c]) for c in columns_change}
            usable = {c: o for c, o in usable.items() if o}
            if not usable:
                continue
            cols = rnd.sample(sorted(usable), min(n_cols, len(usable)))
            row = source.copy()
            for c in cols:
                new = corrupt(c, source[c], rnd.choice(usable[c]))
                if new is not None:
                    row[c] = new
            k = key(row[data_columns].tolist())
            if k in seen:
                continue
            seen.add(k)
            if id_right:
                new_id = str(uuid.uuid4())
                while new_id in existing_ids:
                    new_id = str(uuid.uuid4())
                existing_ids.add(new_id)
                row[id_right] = new_id
            new_rows.append(row)
            new_left.append(l_id)

    if not new_rows:
        return right.copy(), matches.copy()
    new = pd.DataFrame(new_rows, columns=right.columns)
    for c in right.columns:
        if c != ORIGINAL_INDEX and new[c].dtype != right[c].dtype:
            try:
                new[c] = new[c].astype(right[c].dtype)
            except (TypeError, ValueError):
                pass  # e.g. a missing value in an integer column: keep the wider type
    right_out = _append_synthetic(right, new)
    added = right_out.index[len(right):]
    pairs = pd.DataFrame({"left": new_left, "right": added})
    matches_out = pd.concat([matches, pairs], ignore_index=True)
    return right_out, matches_out
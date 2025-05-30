import re
import numpy as np
import pandas as pd
from collections import OrderedDict
from .base import SuperClass
import spacy

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

        Attributes:
        -----------
        text : str
            The input text to process.

        Returns:
        --------
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
            If True, fills missing numeric values with `0` before conversion to numeric dtype.
            Default is False.
        to_numeric : list, optional
            A list of column names to be converted to numeric dtype.
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
            - Columns specified in `to_numeric` are converted to numeric dtype after 
              removing non-numeric characters and optionally filling missing values.
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
                    # remove non-numeric characters
                    df[col] = df[col].astype(str).str.replace(r'[^\d\.]','', regex=True)
                    # fill NaNs with 0 if specified
                    if fill_numeric_na == True:
                        df[col] = df[col].replace(r'','0',regex=True)
                    # convert to numeric dtype
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
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

import re
import numpy as np
import pandas as pd
from .base import SuperClass

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
    """
    
    def format(self, fill_numeric_na: bool = False, to_numeric: list = [], fill_string_na: bool = False, capitalize: bool = False):
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

        # Function to clean and process each DataFrame
        def process_df(df, columns, id_column):
            # Select and rename relevant columns
            df = df[
                [
                re.sub(r'\s', '', col) for col in columns
                ] + [id_column]
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

            # Capitalize if wished
            if capitalize == True:
                for col in columns:
                    if not col in to_numeric:
                        df[col] = df[col].str.upper()

            return df

        # Prepare columns for both DataFrames
        columns_left = list(set([
            key.split('~')[0] 
                if re.search('~',key)!= None 
                else key 
            for key in self.similarity_map
        ]))

        columns_right = list(set([
            key.split('~')[1] 
                if re.search('~',key)!= None 
                else key 
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
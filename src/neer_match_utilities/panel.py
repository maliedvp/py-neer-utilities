import pandas as pd
import numpy as np
from itertools import combinations
import uuid
from typing import List, Dict, Tuple
from neer_match.matching_model import DLMatchingModel, NSMatchingModel


class SetupData:
    """
    A class for processing and preparing data with overlapping matches and panel relationships.

    Attributes
    ----------
    matches : list
        A list of tuples representing matches.
    """

    def __init__(self, matches: list = None):
        """
        Initialize the SetupData class.

        Parameters
        ----------
        matches : list, optional
            A list of tuples representing matches. Defaults to an empty list.
        """
        
        self.matches = matches if matches is not None else []
        self.dfm = pd.DataFrame(self.matches, columns=['left', 'right'])

    def adjust_overlap(self, dfm: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust the overlap between left and right columns in the DataFrame.

        Parameters
        ----------
        dfm : pd.DataFrame
            DataFrame containing 'left' and 'right' columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with overlaps adjusted and additional combinations added.
        """

        if not set(dfm['left']).isdisjoint(dfm['right']):
            overlapping_ids = set(dfm['left']).intersection(dfm['right'])

            for overlap_id in overlapping_ids:
                chain = set(
                    dfm[dfm['left'] == overlap_id]['right'].tolist() +
                    dfm[dfm['right'] == overlap_id]['left'].tolist() +
                    [overlap_id]
                )

                combinations_df = pd.DataFrame(
                    list(combinations(sorted(chain), 2)),
                    columns=['left', 'right']
                )
                dfm = pd.concat([dfm, combinations_df], ignore_index=True) \
                    .drop_duplicates(ignore_index=True) \
                    .sort_values(by='left') \
                    .reset_index(drop=True)

        return dfm

    @staticmethod
    def drop_repetitions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate pairs in the DataFrame irrespective of order.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'left' and 'right' columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with duplicates removed.
        """

        df['sorted_pair'] = df.apply(lambda row: tuple(sorted([row['left'], row['right']])), axis=1)
        df = df.drop_duplicates(subset=['sorted_pair']).drop(columns=['sorted_pair'])
        return df

    def create_connected_groups(self, df_dict: Dict, matches: List[Tuple[int, int]]) -> List[List[int]]:
        """
        Create a list of lists where sublists contain connected values as one group.

        Parameters
        ----------
        df_dict : dict
            A dictionary where keys are integers and values are lists of integers.
        matches : list of tuple of int
            A list of tuples representing connections between values.

        Returns
        -------
        list of list of int
            A list of lists with connected values grouped together.
        """

        value_to_key = {val: key for key, values in df_dict.items() for val in values}
        connections = {}
        for left, right in matches:
            key_left, key_right = value_to_key.get(left), value_to_key.get(right)
            if key_left and key_right:
                if key_left in connections:
                    connections[key_left].add(key_right)
                else:
                    connections[key_left] = {key_left, key_right}

                if key_right in connections:
                    connections[key_right].add(key_left)
                else:
                    connections[key_right] = {key_left, key_right}

        connected_groups = []
        visited = set()
        for key in df_dict.keys():
            if key in visited:
                continue
            if key in connections:
                group = set()
                stack = [key]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        group.add(current)
                        stack.extend(connections.get(current, []))
                connected_groups.append(group)
            else:
                connected_groups.append({key})

        result = []
        for group in connected_groups:
            combined_values = []
            for key in group:
                combined_values.extend(df_dict[key])
            result.append(combined_values)

        return result

    def panel_preparation(self, dfm: pd.DataFrame, df_panel: pd.DataFrame, unique_id: str, panel_id: str) -> pd.DataFrame:
        """
        Generate combinations of IDs for each panel and append them to the DataFrame.

        Parameters
        ----------
        dfm : pd.DataFrame
            DataFrame to append combinations to.
        df_panel : pd.DataFrame
            Panel DataFrame containing IDs and panel information.
        unique_id : str
            Column name of unique identifiers in df_panel.
        panel_id : str
            Column name of panel identifiers in df_panel.

        Returns
        -------
        pd.DataFrame
            Updated DataFrame with appended combinations.
        """

        df_dict = {}

        for pid in df_panel[panel_id].unique():
            unique_ids = sorted(df_panel[df_panel[panel_id] == pid][unique_id].tolist())
            df_dict[pid] = unique_ids

        groups = self.create_connected_groups(
            df_dict=df_dict,
            matches=self.matches
        )
        for g in groups:
            combinations_df = pd.DataFrame(list(combinations(g, 2)), columns=['left', 'right'])
            dfm = pd.concat([dfm, combinations_df], ignore_index=True).drop_duplicates(ignore_index=True)

        return dfm

    def data_preparation(self, df_panel: pd.DataFrame, unique_id: str, panel_id: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data by handling overlaps, panel combinations, and duplicates.

        Parameters
        ----------
        df_panel : pd.DataFrame
            Panel DataFrame containing IDs and panel information.
        unique_id : str
            Column name of unique identifiers in df_panel.
        panel_id : str, optional
            Column name of panel identifiers in df_panel.

        Returns
        -------
        tuple
            DataFrames for left, right, and the final matches.
        """

        try:
            df_panel[unique_id] = pd.to_numeric(df_panel[unique_id], errors='raise')
            stabile_dtype = df_panel[unique_id].dtype
        except ValueError:
            df_panel[unique_id] = df_panel[unique_id].astype(str)
            stabile_dtype = str

        dfm = self.dfm.copy()

        if panel_id:
            dfm = self.panel_preparation(dfm, df_panel, unique_id, panel_id)

        dfm = self.adjust_overlap(dfm)
        dfm = self.drop_repetitions(dfm)

        dfm['left'] = dfm['left'].astype(stabile_dtype)
        dfm['right'] = dfm['right'].astype(stabile_dtype)

        left = df_panel[df_panel[unique_id].isin(dfm['left'])].drop_duplicates(ignore_index=True)
        left = left[[c for c in left if c != panel_id]]
        left[unique_id] = left[unique_id].astype(stabile_dtype)

        right = df_panel[df_panel[unique_id].isin(dfm['right'])].drop_duplicates(ignore_index=True)
        right = right[[c for c in right if c != panel_id]]
        right[unique_id] = right[unique_id].astype(stabile_dtype)

        return left, right, dfm


class GenerateID:
    """
    A class to generate and harmonize unique IDs across time periods for panel data.

    Methods
    -------
    group_by_subgroups():
        Group the panel data into subgroups.
    generate_suggestions(df_slice):
        Generate ID suggestions for consecutive time periods.
    harmonize_ids(suggestions, periods, original_df):
        Harmonize IDs across time periods.
    assign_ids(id_mapping):
        Assign unique IDs to the harmonized IDs.
    execute():
        Execute the full ID generation and harmonization process.
    """

    def __init__(
        self, 
        df_panel: pd.DataFrame, 
        panel_var: str, 
        time_var: str, 
        model, 
        similarity_map: Dict, 
        prediction_threshold: float = 0.9, 
        subgroups: List[str] = None,
        relation: str = 'm:m'
    ):
        """
        Initialize the GenerateID class.

        Parameters
        ----------
        df_panel : pd.DataFrame
            The panel dataset.
        panel_var : str
            The panel identifier variable that is supposed to be created.
        time_var : str
            The time period variable.
        subgroups : list, optional
            List of subgroup variables for slicing. Defaults to None.
        model : object
            A model object with a `suggest` method.
        similarity_map : dict
            A dictionary of similarity functions for columns.
        prediction_threshold : float, optional
            Threshold for predictions. Defaults to 0.9.
        relation : str, optional
            Relationship between observations in cross sectional data. Default is 'm:m'
        """

        subgroups = subgroups or []

        if df_panel.index.duplicated().any():
            raise ValueError("The index of df_panel contains duplicate entries.")

        self.panel_var = panel_var
        self.time_var = time_var
        self.subgroups = subgroups
        self.model = model
        self.similarity_map = similarity_map
        self.prediction_threshold = prediction_threshold
        self.relation = relation

        # Ensure df_panel only includes relevant columns
        self.df_panel = df_panel[
            [col for col in df_panel.columns if col in similarity_map.keys() or col in subgroups or col == time_var]
        ]

    def group_by_subgroups(self):
        """
        Group the panel data into subgroups.

        Returns
        -------
        pd.core.groupby.generic.DataFrameGroupBy
            Grouped dataframe by subgroups.
        """

        return self.df_panel.groupby(self.subgroups)

    def relations_left_right(self, df: pd.DataFrame, relation: str = None) -> pd.DataFrame:
        """
        Apply validation rules to enforce relationships between matched observations.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'left', 'right', and 'prediction' columns.
        relation : str, optional
            Validation mode for relationships. If None, defaults to `self.relation`.
            Options:
            - 'm:m' : Many-to-many, no duplicates removed.
            - '1:m' : Unique 'left' values.
            - 'm:1' : Unique 'right' values.
            - '1:1' : Unique 'left' and 'right' values.

        Returns
        -------
        pd.DataFrame
            A reduced DataFrame with relationships enforced based on `relation`.

        Raises
        ------
        ValueError
            If `relation` is not one of ['m:m', '1:m', 'm:1', '1:1'].
        """

        relation = relation if relation is not None else self.relation

        if relation == 'm:m':
            pass
        else:
            df = df.sort_values(
                by = 'prediction',
                ascending = False,
                ignore_index = True
            )

            if relation == '1:m':        
                df = df.drop_duplicates(
                    subset = 'left',
                    keep='first'
                )
            elif relation == 'm:1':
                df = df.drop_duplicates(
                    subset = 'right',
                    keep='first'
                )
            elif relation == '1:1':
                df = df.drop_duplicates(
                    subset = 'left',
                    keep='first'
                )
                df = df.drop_duplicates(
                    subset = 'right',
                    keep='first'
                )
            else:
                raise ValueError(
                    f"Invalid value for `relation`: '{relation}'. "
                    "It must be one of ['m:m', '1:m', 'm:1', '1:1']."
                )

            df = df.reset_index(drop=False)

        return df

    def generate_suggestions(self, df_slice: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
        """
        Generate ID suggestions for consecutive time periods.

        Parameters
        ----------
        df_slice : pd.DataFrame
            A dataframe slice containing data to process.

        Returns
        -------
        tuple
            A tuple containing:
            - pd.DataFrame: A concatenated dataframe of suggestions.
            - list of int: A list of periods.
        """

        periods = sorted(df_slice[self.time_var].unique())
        suggestions_dict = {}

        for idx, period in enumerate(periods[:-1]):
            print(f"Processing periods {period}-{periods[idx + 1]} at {pd.Timestamp.now()}")

            left = df_slice[df_slice[self.time_var] == period].reset_index(drop=False)
            right = df_slice[df_slice[self.time_var] == periods[idx + 1]].reset_index(drop=False)

            suggestions = self.model.suggest(left, right, count=1)
            suggestions = suggestions[suggestions['prediction'] >= self.prediction_threshold]

            suggestions = self.relations_left_right(
                  df = suggestions,
                  relation = self.relation
            )


            suggestions = pd.merge(
                left[['index']], suggestions, left_index=True, right_on='left'
            )
            suggestions = pd.merge(
                suggestions, right[['index']], left_on='right', right_index=True, suffixes=('_left', '_right')
            )

            suggestions['period_left'] = period
            suggestions['period_right'] = periods[idx + 1]
            suggestions['periods_compared'] = f"{period}-{periods[idx + 1]}"

            suggestions.drop(columns=['left', 'right'], inplace=True)
            suggestions_dict[idx] = suggestions

        if suggestions_dict:
            suggestions_df = pd.concat(suggestions_dict.values(), ignore_index=True)
        else:
            suggestions_df = pd.DataFrame(columns=[
                'index_left',
                'prediction',
                'index_right',
                'period_left',
                'period_right',
                'periods_compared'
            ])

        return suggestions_df, periods

    def harmonize_ids(self, suggestions: pd.DataFrame, periods: List[int], original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonize IDs across time periods.

        Parameters
        ----------
        suggestions : pd.DataFrame
            The dataframe with suggestions.
        periods : list of int
            List of periods.
        original_df : pd.DataFrame
            The original dataframe.

        Returns
        -------
        pd.DataFrame
            Harmonized ID mapping.
        """

        unique_ids = list(original_df.index)

        id_mapping = pd.DataFrame({
            'index': unique_ids,
            'index_harm': unique_ids
        })

        for period in sorted(periods, reverse=True):
            replacement_map = dict(zip(id_mapping['index'], id_mapping['index_harm']))
            temp_df = suggestions[suggestions['period_right'] == period].copy()

            temp_df['id_left_harm'] = temp_df['index_left'].map(replacement_map)
            temp_df['id_right_harm'] = temp_df['index_right'].map(replacement_map)

            update_map = dict(zip(temp_df['id_right_harm'], temp_df['id_left_harm']))
            id_mapping['index_harm'] = id_mapping['index_harm'].map(update_map).fillna(id_mapping['index_harm']).astype(int)

        return id_mapping

    def assign_ids(self, id_mapping: pd.DataFrame) -> pd.DataFrame:
        """
        Assign unique IDs to the harmonized IDs.

        Parameters
        ----------
        id_mapping : pd.DataFrame
            The harmonized ID mapping dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe with assigned unique IDs.
        """

        unique_indices = id_mapping['index_harm'].unique()
        id_map = {idx: uuid.uuid4() for idx in unique_indices}

        id_mapping[self.panel_var] = id_mapping['index_harm'].map(id_map)
        id_mapping.drop(columns=['index_harm'], inplace=True)
        return id_mapping.reset_index(drop=True)

    def execute(self) -> pd.DataFrame:
        """
        Execute the full ID generation and harmonization process.

        Returns
        -------
        pd.DataFrame
            The final ID mapping.
        """

        if self.subgroups:
            harmonized_dict = {}
            for subgroup, group_df in self.group_by_subgroups():
                suggestions, periods = self.generate_suggestions(group_df)
                harmonized_dict[subgroup] = self.harmonize_ids(suggestions, periods, group_df)

            id_mapping = pd.concat(harmonized_dict.values(), ignore_index=False)
        else:
            suggestions, periods = self.generate_suggestions(self.df_panel)
            id_mapping = self.harmonize_ids(suggestions, periods, self.df_panel)

        return self.assign_ids(id_mapping)

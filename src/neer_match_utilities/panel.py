import pandas as pd
import numpy as np
from itertools import combinations, permutations
import uuid
from typing import List, Dict, Tuple, Union
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
from neer_match.similarity_map import SimilarityMap


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
        if matches is None:
            self.matches = []
        elif isinstance(matches, list):
            self.matches = matches
        elif isinstance(matches, pd.DataFrame):
            self.matches = list(matches.itertuples(index=False, name=None))

        self.dfm = pd.DataFrame(self.matches, columns=['left', 'right'])

    def adjust_overlap(self, dfm: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusts the overlap in the matches DataFrame by generating additional
        ordered pairs from connected components in the match graph.

        This function takes a DataFrame containing match pairs in the 'left' and
        'right' columns. It constructs a full connection graph from these matches,
        computes connected components using depth-first search (DFS), and then,
        for each connected component with more than one element, generates all
        ordered pairs (i.e. permutations) of distinct IDs. These new pairs are
        appended to the original DataFrame.

        Parameters
        ----------
        dfm : pd.DataFrame
            A DataFrame with columns 'left' and 'right' representing the matching
            pairs, where each entry is a unique identifier.

        Returns
        -------
        pd.DataFrame
            A combined DataFrame that includes both the original match pairs and
            the newly generated pairs from connected components. Note that pairs
            of the form (A, A) are not generated, and for any distinct IDs A and B,
            both (A, B) and (B, A) may appear. Duplicates are not removed.

        Notes
        -----
        - Connected components are determined by treating the match pairs as
          edges in an undirected graph.
        - The function uses permutations of length 2 to generate ordered pairs,
          ensuring that only pairs of distinct IDs are created.
        - The new pairs are simply appended to the original DataFrame without
          dropping duplicates.
        """

        # Build the full connection graph from all matches
        all_ids = set(dfm['left']).union(set(dfm['right']))
        graph = {id_: set() for id_ in all_ids}
        for _, row in dfm.iterrows():
            l, r = row['left'], row['right']
            graph[l].add(r)
            graph[r].add(l)
        
        # Compute connected components using a depth-first search (DFS)
        seen = set()
        components = []
        for id_ in all_ids:
            if id_ not in seen:
                component = set()
                stack = [id_]
                while stack:
                    node = stack.pop()
                    if node not in component:
                        component.add(node)
                        stack.extend(graph[node] - component)
                seen.update(component)
                components.append(component)
        
        # For each connected component, generate all ordered pairs (permutations)
        new_pairs = []
        for comp in components:
            if len(comp) > 1:
                # This generates both (A,B) and (B,A) for all distinct A and B.
                new_pairs.extend(list(permutations(comp, 2)))
        
        new_df = pd.DataFrame(new_pairs, columns=['left', 'right'])

        # Append new rows to the original dfm without resetting its order:
        df_combi = pd.concat([dfm.copy(), new_df])

        return df_combi

    @staticmethod
    def drop_repetitions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate pairs in the DataFrame irrespective of the order of elements.

        This function treats each pair in the 'left' and 'right' columns as unordered.
        It creates a temporary column 'sorted_pair' that contains a sorted tuple of the 
        'left' and 'right' values for each row. For example, the pairs (A, B) and (B, A)
        will both be transformed into (A, B), and only one instance will be retained.
        The function then drops duplicate rows based on this sorted pair and removes 
        the temporary column before returning the result.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing at least the columns 'left' and 'right', which represent
            paired elements.

        Returns
        -------
        pd.DataFrame
            A DataFrame in which duplicate pairs (ignoring order) have been removed.
        """

        df['sorted_pair'] = df.apply(lambda row: tuple(sorted([row['left'], row['right']])), axis=1)
        df = df.sort_values(['left','right'])
        df = df.drop_duplicates(subset=['sorted_pair']).drop(columns=['sorted_pair'])
        df = df.reset_index(drop=True)

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
            sorted_group = sorted(g)
            combinations_df = pd.DataFrame(list(combinations(g, 2)), columns=['left', 'right'])
            dfm = pd.concat([dfm, combinations_df], ignore_index=True).drop_duplicates(ignore_index=True)

        return dfm

    def data_preparation_cs(self, df_left: pd.DataFrame, df_right: pd.DataFrame, unique_id:str):
        df_panel = pd.concat([df_left, df_right], ignore_index=True, axis=0)
        return self.data_preparation_panel(df_panel=df_panel, unique_id=unique_id)

    def data_preparation_panel(self, df_panel: pd.DataFrame, unique_id: str, panel_id: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data by handling overlaps, panel combinations, and duplicates.

        This function converts the unique identifier column to a numeric type (if possible),
        then prepares match pairs by adjusting overlaps and dropping duplicate pairs.
        It then extracts the subset of the panel data corresponding to the matched IDs,
        and finally sorts the left and right DataFrames according to the unique identifier.

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
            A tuple of three DataFrames: left, right, and the final matches.
        """
        # Convert unique_id to numeric if possible, else to string.
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

        # Define left and right directly from the full set of matches.
        unique_ids = set(dfm['left']).union(set(dfm['right']))
        combined = df_panel[df_panel[unique_id].isin(unique_ids)].drop_duplicates()

        # Prepare left DataFrame, drop panel_id column if provided, convert type, sort by unique_id.
        left = combined[combined[unique_id].isin(dfm['left'])].copy()
        if panel_id:
            left = left[[c for c in left.columns if c != panel_id]]
        left.loc[:, unique_id] = left[unique_id].astype(stabile_dtype)
        left = left.sort_values(by=unique_id).reset_index(drop=True)

        # Prepare right DataFrame similarly.
        right = combined[combined[unique_id].isin(dfm['right'])].copy()
        if panel_id:
            right = right[[c for c in right.columns if c != panel_id]]
        right.loc[:, unique_id] = right[unique_id].astype(stabile_dtype)
        right = right.sort_values(by=unique_id).reset_index(drop=True)

        """
        it is (correctly) possible that some observations now are included in left and right. (see test/test_setup.py)
        identify them and add them to the dfm
        """

        lr_list = sorted(
            list(
                set(
                    left[unique_id]).intersection(set(right[unique_id])
                )
            )
        )

        dfm = pd.concat(
            [
                dfm,
                pd.DataFrame({
                    'left' : lr_list,
                    'right' : lr_list
                })
            ],
            axis=0,
            ignore_index=True
        )

        dfm['left'] = dfm['left'].astype(stabile_dtype)
        dfm['right'] = dfm['right'].astype(stabile_dtype)


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
        similarity_map: Union[dict, "SimilarityMap", None] = None, 
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
        # Use the provided similarity_map, or if not provided, take the model's own similarity_map.
        self.similarity_map = similarity_map if similarity_map is not None else model.similarity_map
        self.prediction_threshold = prediction_threshold
        self.relation = relation

        # Extract the top-level field keys from the similarity map.
        if isinstance(self.similarity_map, dict):
            field_keys = list(self.similarity_map.keys())
        elif hasattr(self.similarity_map, "instructions"):
            field_keys = list(self.similarity_map.instructions.keys())
        else:
            field_keys = []
        
        # Ensure df_panel only includes columns corresponding to these field keys, subgroups, or the time variable.
        self.df_panel = df_panel[
            [col for col in df_panel.columns if col in field_keys or col in subgroups or col == time_var]
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
                left[['index']].rename(columns={'index': 'index_left'}),
                suggestions, left_index=True, right_on='left'
            )
            suggestions = pd.merge(
                suggestions,
                right[['index']].rename(columns={'index': 'index_right'}),
                left_on='right', right_index=True
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

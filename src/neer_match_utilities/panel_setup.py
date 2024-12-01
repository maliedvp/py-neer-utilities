import pandas as pd
import numpy as np
from itertools import combinations


class SetupData:
    """
    A class for processing and preparing data with overlapping matches and panel relationships.

    Attributes:
        matches (list): A list of tuples representing matches.
    """

    def __init__(self, matches: list = None):
        """
        Initialize the SetupData class.

        Args:
            matches (list): A list of tuples representing matches. Default is an empty list.
        """
        self.matches = matches if matches is not None else []
        self.dfm = pd.DataFrame(self.matches, columns=['left', 'right'])

    def adjust_overlap(self, dfm: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust the overlap between left and right columns in the DataFrame.

        Args:
            dfm (pd.DataFrame): DataFrame containing 'left' and 'right' columns.

        Returns:
            pd.DataFrame: DataFrame with overlaps adjusted and additional combinations added.
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

        Args:
            df (pd.DataFrame): DataFrame containing 'left' and 'right' columns.

        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        df['sorted_pair'] = df.apply(lambda row: tuple(sorted([row['left'], row['right']])), axis=1)
        df = df.drop_duplicates(subset=['sorted_pair']).drop(columns=['sorted_pair'])
        return df


    def create_connected_groups(self, df_dict, matches):
        """
        Create a list of lists where sublists contain connected values as one (if a connection exists)
        and the normal values for keys without connections.

        Args:
            df_dict (dict): A dictionary where keys are numpy integers and values are lists of integers.
            matches (list): A list of tuples representing connections between values.

        Returns:
            list: A list of lists with connected values grouped together.
        """
        # Flatten df_dict into a value-to-key mapping
        value_to_key = {val: key for key, values in df_dict.items() for val in values}

        # Build a connection map from matches
        connections = {}
        for left, right in matches:
            key_left, key_right = value_to_key.get(left), value_to_key.get(right)
            if key_left and key_right:
                # Merge sets of connected keys
                if key_left in connections:
                    connections[key_left].add(key_right)
                else:
                    connections[key_left] = {key_left, key_right}

                if key_right in connections:
                    connections[key_right].add(key_left)
                else:
                    connections[key_right] = {key_left, key_right}

        # Combine connected keys into groups
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

        # Map connected groups back to values
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

        Args:
            dfm (pd.DataFrame): DataFrame to append combinations to.
            df_panel (pd.DataFrame): Panel DataFrame containing IDs and panel information.
            unique_id (str): Column name of unique identifiers in df_panel.
            panel_id (str): Column name of panel identifiers in df_panel.

        Returns:
            pd.DataFrame: Updated DataFrame with appended combinations.
        """
        df_dict = {}

        for pid in df_panel[panel_id].unique():
            unique_ids = sorted(df_panel[df_panel[panel_id] == pid][unique_id].tolist())
            df_dict[pid] = unique_ids

        groups = self.create_connected_groups(
            df_dict = df_dict,
            matches = self.matches
        )
        for g in groups:
            # print(unique_ids)
            combinations_df = pd.DataFrame(list(combinations(g, 2)), columns=['left', 'right'])
            dfm = pd.concat([dfm, combinations_df], ignore_index=True).drop_duplicates(ignore_index=True)
        return dfm

    def data_preparation(self, df_panel: pd.DataFrame, unique_id: str, panel_id: str = None):
        """
        Prepare data by handling overlaps, panel combinations, and duplicates.

        Args:
            df_panel (pd.DataFrame): Panel DataFrame containing IDs and panel information.
            unique_id (str): Column name of unique identifiers in df_panel.
            panel_id (str): Column name of panel identifiers in df_panel (optional).

        Returns:
            tuple: DataFrames for left, right, and the final matches.
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

        # Cast the `left` and `right` columns to the stable dtype
        dfm['left'] = dfm['left'].astype(stabile_dtype)
        dfm['right'] = dfm['right'].astype(stabile_dtype)

        left = df_panel[df_panel[unique_id].isin(dfm['left'])].drop_duplicates(ignore_index=True)
        left = left[[c for c in left if c != panel_id]]
        left[unique_id] = left[unique_id].astype(stabile_dtype)

        right = df_panel[df_panel[unique_id].isin(dfm['right'])].drop_duplicates(ignore_index=True)
        right = right[[c for c in right if c != panel_id]]
        right[unique_id] = right[unique_id].astype(stabile_dtype)

        return left, right, dfm
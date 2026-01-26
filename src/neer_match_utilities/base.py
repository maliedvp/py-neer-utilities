from pandas import DataFrame
from neer_match.similarity_map import SimilarityMap

class SuperClass:
    def __init__(self, similarity_map: dict | SimilarityMap, df_left: DataFrame = DataFrame(), df_right: DataFrame = DataFrame(), id_left: str = "id", id_right: str = "id"):
        # Convert SimilarityMap to dict if needed
        if isinstance(similarity_map, SimilarityMap):
            self.similarity_map = similarity_map.instructions
        else:
            self.similarity_map = similarity_map
        self.df_left = df_left
        self.df_right = df_right
        self.id_left = id_left
        self.id_right = id_right

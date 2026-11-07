import numpy as np
import pandas as pd
from neer_match.similarity_map import SimilarityMap

from neer_match_utilities.feature_selection import FeatureSelector


def columns_of(similarity_map):
    """The feature columns a similarity map produces."""
    return {f"col_{lcol}_{rcol}_{sim}" for lcol, rcol, sim in SimilarityMap(similarity_map)}


def make_selector(similarity_map, **kwargs):
    """Left and right name their columns differently; the right file has the name twice."""
    rng = np.random.default_rng(0)
    names = ["".join(rng.choice(list("abcdefgh"), 6)) for _ in range(30)]
    left = pd.DataFrame({"id": range(30), "name": names, "city": rng.choice(["x", "y"], 30)})
    right = pd.DataFrame({
        "id": range(30),
        "full_name": [n[:-1] + "q" for n in names],
        "alias": [n[:4] for n in names],
        "city": left["city"],
    })
    matches = pd.DataFrame({"left": range(30), "right": range(30)})
    settings = {"cv": 2, "Cs": 3, "l1_ratios": (0.9,), "n_jobs": 1, **kwargs}
    return FeatureSelector(similarity_map, (left, right, matches), **settings)


MAP = {
    "name~full_name": ["levenshtein", "jaro_winkler"],
    "name~alias": ["levenshtein", "prefix"],
    "city": ["discrete"],
}


def test_selected_columns_of_left_right_pairs_stay_in_the_map():
    selector = make_selector(MAP)
    selected = ["col_name_full_name_jaro_winkler", "col_name_alias_prefix", "col_city_city_discrete"]

    updated = selector._selected_cols_to_similarity_map(selected)

    assert updated == {
        "name~full_name": ["jaro_winkler"],
        "name~alias": ["prefix"],
        "city": ["discrete"],
    }


def test_pairs_on_the_same_left_column_are_kept_apart():
    selector = make_selector(MAP)

    updated = selector._selected_cols_to_similarity_map(["col_name_alias_levenshtein"])

    assert updated == {"name~alias": ["levenshtein"]}


def test_the_updated_map_produces_exactly_the_selected_columns():
    result = make_selector(MAP).execute()

    assert result.selected_feature_columns
    assert columns_of(result.updated_similarity_map) == set(result.selected_feature_columns)

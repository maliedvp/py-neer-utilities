import pandas as pd
import pytest
from neer_match import similarity_encoding as _enc
from neer_match import similarity_map as _sim

from neer_match_utilities import custom_similarities
from neer_match_utilities.custom_similarities import CustomSimilarities
from neer_match_utilities.prepare import Commonness


@pytest.fixture(autouse=True)
def restore_registry():
    """The registry is process-wide; keep one test's registrations out of the next."""
    saved = dict(custom_similarities._registered)
    yield
    custom_similarities._registered.clear()
    custom_similarities._registered.update(saved)


def make_commonness():
    df = pd.DataFrame({"name": ["a", "b", "b"]})
    return Commonness(["name"], df, df, df, df)


def test_commonness_keeps_custom_similarities():
    make_commonness()

    sims = _sim.available_similarities()
    assert "commonness_score" in sims
    assert {"notmissing", "notzero", "within_km_lat", "within_km_lon"} <= set(sims)
    assert sims["token_set_ratio"]("ab", "ab") == pytest.approx(1.0)


def test_custom_similarities_keeps_commonness():
    make_commonness()
    CustomSimilarities()

    assert "commonness_score" in _sim.available_similarities()


def test_similarity_encoding_sees_the_same_registry():
    make_commonness()

    # similarity_encoding binds the name at import, so it is patched separately
    assert "commonness_score" in _enc.available_similarities()
    assert _enc.available_similarities()["token_set_ratio"]("ab", "ab") == pytest.approx(1.0)


def test_reconfiguring_replaces_only_its_own_similarities():
    make_commonness()
    within_10km = _sim.available_similarities()["within_km_lat"]

    CustomSimilarities(lat_radius_km=1.0)

    within_1km = _sim.available_similarities()["within_km_lat"]
    # 0.05 degrees of latitude is about 5.6 km
    assert within_10km(54.0, 54.05) == 1.0
    assert within_1km(54.0, 54.05) == 0.0
    assert "commonness_score" in _sim.available_similarities()

import numpy as np
from neer_match import similarity_map as _sim
from neer_match import similarity_encoding as _enc

# Names of similarities that need scaling (because they come from rapidfuzz.fuzz)
_FUZZ_KEYS = [
    "basic_ratio",
    "partial_ratio",
    "partial_ratio_alignment",
    "partial_token_ratio",
    "partial_token_set_ratio",
    "partial_token_sort_ratio",
    "token_ratio",
    "token_set_ratio",
    "token_sort_ratio",
]

# Similarities added on top of neer_match's own, by name. CustomSimilarities and
# Commonness both register here. Each used to install its own wrapper around
# neer_match's original function, so whichever was constructed last removed what
# the other had added -- including the rescaling of the fuzz-based similarities.
_registered = {}


def _norm(f):
    # Wrap a 0–100 similarity into 0–1
    def wrapper(x, y):
        return f(x, y) / 100.0
    return wrapper


def _extended_available():
    sims = _sim._original_available_similarities().copy()

    # --- 1) Rescale fuzz-based similarities from [0, 100] to [0, 1] ---
    for key in _FUZZ_KEYS:
        if key in sims:
            sims[key] = _norm(sims[key])

    # --- 2) Add registered similarities ---
    sims.update(_registered)
    return sims


def register_similarities(similarities: dict) -> None:
    """
    Make similarity functions available to neer_match by name.

    Registered functions are added to neer_match's own, alongside the rescaling of
    the fuzz-based similarities from [0, 100] to [0, 1]. A name that is registered
    again is replaced; all other registered names are kept.

    Parameters
    ----------
    similarities : dict
        Mapping ``{name: callable}``, where each callable takes two values and
        returns their similarity.
    """
    _registered.update(similarities)

    # Store original only once
    if not hasattr(_sim, "_original_available_similarities"):
        _sim._original_available_similarities = _sim.available_similarities

    # similarity_encoding imported the name, so its binding is patched as well
    _sim.available_similarities = _extended_available
    _enc.available_similarities = _extended_available


class CustomSimilarities:
    """
    Monkey-patch neer_match to

    - add custom similarities (notmissing, notzero)
    - rescale fuzz-based similarities from [0, 100] to [0, 1]
    """

    def __init__(self, lat_radius_km: float = 10.0, lon_radius_km: float = 10.0, lat_ref: float = 54.0):
        """
        lat_radius_km: radius in km for within_km_lat similarity
        lon_radius_km: radius in km for within_km_lon similarity
        lat_ref: reference latitude (degrees) for longitude distance (UK ~54°)
        """
        self.lat_radius_km = lat_radius_km
        self.lon_radius_km = lon_radius_km
        self.lat_ref = lat_ref

        register_similarities({
            "notmissing": self.notmissing,
            "notzero": self.notzero,
            # Distance-based similarities with the configured radius
            "within_km_lat": self._make_within_km_lat(self.lat_radius_km),
            "within_km_lon": self._make_within_km_lon(self.lon_radius_km, self.lat_ref),
        })

    @staticmethod
    def notmissing(x: float, y: float) -> float:
        """Return 1.0 if both values are not missing (None, '', or NaN)."""
        if x in [None, ''] or y in [None, '']:
            return 0.0
        if isinstance(x, float) and np.isnan(x):
            return 0.0
        if isinstance(y, float) and np.isnan(y):
            return 0.0
        return 1.0

    @staticmethod
    def notzero(x: float, y: float) -> float:
        """Return 1.0 if both values are non-zero."""
        if x in [None, '', 0] or y in [None, '', 0]:
            return 0.0
        if isinstance(x, float) and np.isnan(x):
            return 0.0
        if isinstance(y, float) and np.isnan(y):
            return 0.0
        return 1.0
    
    @staticmethod
    def _make_within_km_lat(radius_km: float):
        """
        Create a similarity function sim(lat1, lat2) that returns 1.0
        if |lat1 - lat2| corresponds to <= radius_km, else 0.0.
        """
        km_per_deg_lat = 111.32
        threshold_deg = radius_km / km_per_deg_lat

        def sim(lat1: float, lat2: float) -> float:
            if lat1 in [None, ""] or lat2 in [None, ""]:
                return 0.0
            if isinstance(lat1, float) and np.isnan(lat1):
                return 0.0
            if isinstance(lat2, float) and np.isnan(lat2):
                return 0.0
            return 1.0 if abs(lat1 - lat2) <= threshold_deg else 0.0

        return sim

    @staticmethod
    def _make_within_km_lon(radius_km: float, lat_ref: float):
        """
        Create a similarity function sim(lon1, lon2) that returns 1.0
        if |lon1 - lon2| corresponds to <= radius_km, else 0.0.

        Uses km_per_deg_lon = 111.32 * cos(lat_ref).
        """
        km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat_ref))
        threshold_deg = radius_km / km_per_deg_lon

        def sim(lon1: float, lon2: float) -> float:
            if lon1 in [None, ""] or lon2 in [None, ""]:
                return 0.0
            if isinstance(lon1, float) and np.isnan(lon1):
                return 0.0
            if isinstance(lon2, float) and np.isnan(lon2):
                return 0.0
            return 1.0 if abs(lon1 - lon2) <= threshold_deg else 0.0

        return sim
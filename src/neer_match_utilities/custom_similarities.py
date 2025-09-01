import numpy as np
from neer_match import similarity_map as _sim
from neer_match import similarity_encoding as _enc

class CustomSimilarities:
    def __init__(self):
        # Store original only once
        if not hasattr(_sim, "_original_available_similarities"):
            _sim._original_available_similarities = _sim.available_similarities

        orig = _sim._original_available_similarities

        def _extended_available():
            sims = orig().copy()
            sims["notmissing"] = self.notmissing
            sims["notzero"] = self.notzero
            return sims

        # Monkey-patch
        _sim.available_similarities = _extended_available
        _enc.available_similarities = _extended_available

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
import pandas as pd
import pytest

from neer_match_utilities.panel import GenerateID


class LinkByPosition:
    """Stand-in model: links the n-th record of one period to the n-th of the next."""

    def suggest(self, left, right, count=1, verbose=0):
        n = min(len(left), len(right))
        return pd.DataFrame({"left": range(n), "right": range(n), "prediction": [1.0] * n})


# Firm A in 1900, 1910 and 1920; firm B in 1900 and 1910
PANEL = pd.DataFrame({
    "year": [1900, 1900, 1910, 1910, 1920],
    "name": ["A", "B", "A'", "B'", "A''"],
})


@pytest.mark.parametrize(
    "reindex",
    [
        lambda df: df,
        lambda df: df.set_axis([40, 30, 20, 10, 0]),
        lambda df: df.rename_axis("row"),
        lambda df: df.set_axis([f"r{i}" for i in range(5)]),
        # truncated to integers, 0.1 and 0.2 used to become one ID
        lambda df: df.set_axis([0.1, 0.2, 1.1, 1.2, 2.1]),
    ],
    ids=["default", "reordered", "named", "strings", "floats"],
)
def test_ids_follow_the_links_whatever_the_index(reindex):
    panel = reindex(PANEL)

    ids = GenerateID(
        panel, panel_var="pid", time_var="year", model=LinkByPosition(),
        similarity_map={"name": ["levenshtein"]}, prediction_threshold=0.5,
    ).execute()

    assert sorted(ids["index"]) == sorted(panel.index)
    assert ids["index"].dtype == panel.index.dtype
    groups = sorted(sorted(panel.loc[ids.loc[ids["pid"] == pid, "index"], "name"]) for pid in ids["pid"].unique())
    assert groups == [["A", "A'", "A''"], ["B", "B'"]]

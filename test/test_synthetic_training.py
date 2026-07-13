import pytest
import pandas as pd
import numpy as np
import random
from rapidfuzz import fuzz, distance
from neer_match_utilities.prepare import synth_mismatches, available_similarities

# Helper to compute percentage difference
def pct_diff(orig: float, candidate: float) -> float:
    if pd.isna(orig) or pd.isna(candidate):
        return np.nan
    if orig == 0:
        return 0.0 if candidate == 0 else 1.0
    return abs(orig - candidate) / abs(orig)

# Fixture: base 'right' DataFrame with 10 rows and 7 columns
@pytest.fixture
def right_df():
    return pd.DataFrame({
        "rid": range(1001, 1011),
        "name": [
            "Alice", "Bob", "Carol", "David", "Eva",
            "Frank", "Grace", "Hector", "Isabella", "Jack"
        ],
        "age": [25, 30, 22, 45, 28, 35, 40, 27, 33, 29],
        "city": [
            "Stockholm", "Gothenburg", "Malmö", "Uppsala", "Linköping",
            "Örebro", "Västerås", "Helsinki", "Oslo", "Copenhagen"
        ],
        "salary": [45000.0, 52000.5, 48000.0, 61000.0, 50000.0,
                   57000.0, 63000.5, 49000.0, 55000.0, 58000.0],
        "department": [
            "HR", "Engineering", "Marketing", "Sales", "Finance",
            "Engineering", "HR", "Research", "Marketing", "Sales"
        ],
        "score": [88, 92, 75, 81, 95, 78, 85, 90, 80, 87]
    })

# Use a fixed random seed to make sampling deterministic in tests
@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(0)
    random.seed(0)
    yield

def test_synth_mismatches_basic(right_df):
    # Call synth_mismatches with sample_share=0.5, n_mismatches=1
    expanded = synth_mismatches(
        right=right_df,
        id_right="rid",
        columns_fix=["name"],
        columns_change=["age", "city", "salary", "department", "score"],
        str_metric="levenshtein",
        str_similarity_range=(0.0, 0.3),
        pct_diff_range=(0.1, 0.2),
        n_cols=5,
        n_mismatches=1,
        keep_missing=True,
        nan_share=0.0,
        empty_share=0.0,
        sample_share=0.5
    )

    # 1. Original rows should still be present
    pd.testing.assert_frame_equal(
        expanded.iloc[:len(right_df)].reset_index(drop=True),
        right_df.reset_index(drop=True),
        check_dtype=False
    )

    # 2. Check number of synthetic rows: floor(0.5 * 10) = 5
    assert len(expanded) == 15

    # Identify synthetic portion
    synthetic_rows = expanded.iloc[len(right_df):].reset_index(drop=True)

    # 3. Since id_right=None, synthetic rows will have NaN in "rid"
    assert synthetic_rows["rid"].notna().all()

    # 4. For each synthetic row:
    #    - 'name' must equal original (columns_fix)
    #    - numeric columns must differ by 10%-20%
    #    - string columns must have similarity between 0.0 and 0.3
    sim_funcs = available_similarities()
    str_sim = sim_funcs["levenshtein"]

    # Map from original name to original row for easy lookup
    orig_by_name = right_df.set_index("name")

    for _, syn in synthetic_rows.iterrows():
        orig_name = syn["name"]
        assert orig_name in orig_by_name.index

        orig_row = orig_by_name.loc[orig_name]

        # age
        orig_age = orig_row["age"]
        new_age = syn["age"]
        assert not pd.isna(new_age)
        pdiff_age = pct_diff(orig_age, new_age)
        assert 0.1 <= pdiff_age <= 0.2

        # salary
        orig_sal = orig_row["salary"]
        new_sal = syn["salary"]
        assert not pd.isna(new_sal)
        pdiff_sal = pct_diff(orig_sal, new_sal)
        assert 0.1 <= pdiff_sal <= 0.2

        # score
        orig_score = orig_row["score"]
        new_score = syn["score"]
        assert not pd.isna(new_score)
        pdiff_score = pct_diff(orig_score, new_score)
        assert 0.1 <= pdiff_score <= 0.2

        # city
        orig_city = orig_row["city"]
        new_city = syn["city"]
        assert isinstance(new_city, str) and new_city != ""
        sim_city = str_sim(orig_city, new_city)
        assert 0.0 <= sim_city <= 0.3

        # department
        orig_dept = orig_row["department"]
        new_dept = syn["department"]
        assert isinstance(new_dept, str) and new_dept != ""
        sim_dept = str_sim(orig_dept, new_dept)
        assert 0.0 <= sim_dept <= 0.3

def test_synth_mismatches_sample_share_zero(right_df):
    # If sample_share=0.0, no synthetic rows should be created
    expanded = synth_mismatches(
        right=right_df,
        id_right=None,
        columns_fix=["name"],
        columns_change=["age", "city"],
        str_metric="levenshtein",
        str_similarity_range=(0.0, 0.3),
        pct_diff_range=(0.1, 0.2),
        n_cols=2,
        n_mismatches=2,
        keep_missing=True,
        nan_share=0.0,
        empty_share=0.0,
        sample_share=0.0
    )
    # Should equal original
    pd.testing.assert_frame_equal(
        expanded.reset_index(drop=True),
        right_df.reset_index(drop=True),
        check_dtype=False
    )

def test_nan_empty_injection_forced(right_df):
    # Force injection by setting nan_share=1.0, empty_share=0.0
    # and sample_share=0.5, n_mismatches=2 so that 5 sampled rows * 2 = 10 synthetics
    expanded = synth_mismatches(
        right=right_df,
        id_right=None,
        columns_fix=["name"],
        columns_change=["city", "age"],
        str_metric="levenshtein",
        str_similarity_range=(0.0, 0.3),
        pct_diff_range=(0.1, 0.2),
        n_cols=2,
        n_mismatches=2,
        keep_missing=True,
        nan_share=1.0,
        empty_share=0.0,
        sample_share=0.5
    )

    # Original rows
    assert len(expanded) >= len(right_df)

    # Synthetic rows are those after index 10
    synth = expanded.iloc[len(right_df):].reset_index(drop=True)

    # With nan_share=1.0, every synthetic 'city' and 'age' must be NaN
    assert synth["city"].isna().all()
    assert synth["age"].isna().all()

    # Now force empty_share=1.0, nan_share=0.0
    expanded2 = synth_mismatches(
        right=right_df,
        id_right=None,
        columns_fix=["name"],
        columns_change=["city", "age"],
        str_metric="levenshtein",
        str_similarity_range=(0.0, 0.3),
        pct_diff_range=(0.1, 0.2),
        n_cols=2,
        n_mismatches=2,
        keep_missing=True,
        nan_share=0.0,
        empty_share=1.0,
        sample_share=0.5
    )

    synth2 = expanded2.iloc[len(right_df):].reset_index(drop=True)
    # With empty_share=1.0, every synthetic 'city' and 'age' must be ""
    assert (synth2["city"] == "").all()
    assert (synth2["age"] == "").all()


# Fuzz-based metrics come out of rapidfuzz on a 0-100 scale and are rescaled to
# 0-1 by CustomSimilarities. synth_mismatches validates str_similarity_range
# against [0, 1], so it must see the rescaled versions.
FUZZ_KEYS = [
    "basic_ratio",
    "partial_ratio",
    "partial_token_ratio",
    "partial_token_set_ratio",
    "partial_token_sort_ratio",
    "token_ratio",
    "token_set_ratio",
    "token_sort_ratio",
]

def test_available_similarities_rescales_fuzz_metrics():
    sim_funcs = available_similarities()

    for key in FUZZ_KEYS:
        assert key in sim_funcs
        # Identical strings sit at the top of the scale, not at 100
        assert sim_funcs[key]("Stockholm", "Stockholm") == pytest.approx(1.0)
        # Partial overlap stays inside the unit interval
        assert 0.0 <= sim_funcs[key]("Stockholm", "Gothenburg") <= 1.0

    # A partial match is genuinely partial, i.e. the rescaling is not clipping
    assert 0.0 < sim_funcs["basic_ratio"]("alice", "alicia") < 1.0

def test_available_similarities_resolves_at_call_time(monkeypatch):
    # The similarity registry is monkey-patched at runtime (CustomSimilarities,
    # Commonness). Resolving it at import time would freeze the unpatched
    # version and silently bypass the rescaling above.
    from neer_match import similarity_map as _sim

    monkeypatch.setattr(
        _sim,
        "available_similarities",
        lambda: {"sentinel_metric": lambda x, y: 1.0},
    )

    assert "sentinel_metric" in available_similarities()

def test_synth_mismatches_with_fuzz_metric(right_df):
    # End-to-end guard: every pair of fixture cities scores <= 0.47 once
    # rescaled, so a range of (0.0, 0.5) admits a real city as a candidate for
    # every row. On the raw 0-100 scale nothing would qualify and the fallback
    # would emit random ASCII instead.
    expanded = synth_mismatches(
        right=right_df,
        id_right="rid",
        columns_fix=["name"],
        columns_change=["city"],
        str_metric="basic_ratio",
        str_similarity_range=(0.0, 0.5),
        pct_diff_range=(0.1, 0.2),
        n_cols=1,
        n_mismatches=1,
        keep_missing=True,
        nan_share=0.0,
        empty_share=0.0,
        sample_share=1.0
    )

    synth = expanded.iloc[len(right_df):].reset_index(drop=True)
    assert len(synth) > 0

    str_sim = available_similarities()["basic_ratio"]
    known_cities = set(right_df["city"])
    orig_by_name = right_df.set_index("name")

    for _, syn in synth.iterrows():
        new_city = syn["city"]
        # Drawn from the candidate pool, not fabricated by the perturbation fallback
        assert new_city in known_cities

        orig_city = orig_by_name.loc[syn["name"], "city"]
        assert new_city != orig_city
        assert 0.0 <= str_sim(orig_city, new_city) <= 0.5

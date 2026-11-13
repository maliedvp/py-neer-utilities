"""learn_variants and synth_matches: synthetic matches with errors that keep the identity."""
import numpy as np
import pandas as pd
import pytest

from neer_match_utilities.prepare import (
    OCR_CONFUSIONS, _directed_confusions, learn_variants, synth_matches,
)
from neer_match_utilities.split import pairs_in_subset, split_test_train

COLS = ["first_name", "last_name", "birth_year"]


@pytest.fixture
def persons():
    """Twelve persons; the right source records some first names differently."""
    first_l = ["JOHANN", "CARL", "MARIA", "ANNA", "JOHANN", "PETER", "ERIK", "KARIN", "OLOF", "ANDERS", "LARS", "BRITA"]
    first_r = ["JOHANNES", "KARL", "MARIA", "ANNA", "JOHANNES", "PETER", "ERIC", "KARIN", "OLOF", "ANDERS", "LARS", "BRITA"]
    last = ["FISCHER", "SCHMIDT", "MÜLLER", "BERG", "HORN", "LUND", "STRÖM", "NILSSON", "ANDERSSON", "BROMAN", "MORN", "KLINT"]
    years = [1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861]
    left = pd.DataFrame({"first_name": first_l, "last_name": last, "birth_year": years}, index=range(10, 22))
    right = pd.DataFrame({"first_name": first_r, "last_name": last, "birth_year": years}, index=range(40, 52))
    matches = pd.DataFrame({"left": range(10, 22), "right": range(40, 52)})
    return left, right, matches


def test_learn_variants_is_symmetric(persons):
    left, right, matches = persons
    v = learn_variants(left, right, matches, ["first_name", "last_name"])
    assert v["first_name"] == {
        "JOHANN": {"JOHANNES"}, "JOHANNES": {"JOHANN"},
        "CARL": {"KARL"}, "KARL": {"CARL"},
        "ERIK": {"ERIC"}, "ERIC": {"ERIK"},
    }
    assert v["last_name"] == {}


def test_learn_variants_min_count_and_column_map(persons):
    left, right, matches = persons
    v = learn_variants(left, right, matches, ["first_name"], min_count=2)
    assert v["first_name"] == {"JOHANN": {"JOHANNES"}, "JOHANNES": {"JOHANN"}}
    renamed = left.rename(columns={"first_name": "fn"})
    w = learn_variants(renamed, right, matches, {"first_name": "fn"})
    assert w["first_name"]["CARL"] == {"KARL"}


def check_basics(right, matches, r2, m2, n_cols):
    """Originals unchanged; new rows after the largest label; one new pair per new row."""
    pd.testing.assert_frame_equal(r2.loc[right.index], right, check_dtype=False)
    new = r2.index[len(right):]
    assert (new > right.index.max()).all()
    pd.testing.assert_frame_equal(m2.iloc[:len(matches)].reset_index(drop=True), matches.reset_index(drop=True))
    added = m2.iloc[len(matches):]
    assert list(added["right"]) == list(new)
    source = dict(zip(matches["left"], matches["right"]))
    for l, r in zip(added["left"], added["right"]):
        orig = right.loc[source[l]]
        changed = [c for c in right.columns if not (r2.at[r, c] == orig[c] or (pd.isna(r2.at[r, c]) and pd.isna(orig[c])))]
        assert 1 <= len(changed) <= n_cols
        yield l, r, orig, changed


def test_variant_corruption_uses_variants_only(persons):
    left, right, matches = persons
    v = learn_variants(left, right, matches, ["first_name"])
    r2, m2 = synth_matches(right, matches, ["first_name"], corruptions=("variant",), variants=v, random_state=0)
    rows = list(check_basics(right, matches, r2, m2, 1))
    assert len(rows) == 4  # only JOHANNES (twice), KARL and ERIC have variants
    for _, r, orig, changed in rows:
        assert changed == ["first_name"]
        assert r2.at[r, "first_name"] in v["first_name"][orig["first_name"]]


def test_ocr_corruption_is_one_confusion(persons):
    _, right, matches = persons
    r2, m2 = synth_matches(right, matches, ["first_name", "last_name"], corruptions=("ocr",), random_state=1)
    confusions = _directed_confusions(OCR_CONFUSIONS)
    rows = list(check_basics(right, matches, r2, m2, 1))
    assert rows
    for _, r, orig, changed in rows:
        (c,) = changed
        before, after = orig[c], r2.at[r, c]
        assert any(
            before[:i] + b + before[i + len(a):] == after
            for a, b in confusions for i in range(len(before)) if before.startswith(a, i)
        )


def test_typo_corruption_is_one_slip(persons):
    _, right, matches = persons
    r2, m2 = synth_matches(right, matches, ["last_name"], corruptions=("typo",), n_per_match=3, random_state=2)
    for _, r, orig, changed in check_basics(right, matches, r2, m2, 1):
        before, after = orig["last_name"], r2.at[r, "last_name"]
        assert abs(len(after) - len(before)) <= 1
        if len(after) == len(before):  # a swap of two neighbours
            assert sorted(after) == sorted(before)


def test_shift_and_missing(persons):
    _, right, matches = persons
    r2, m2 = synth_matches(right, matches, ["birth_year"], corruptions=("shift",), max_shift=2, random_state=3)
    for _, r, orig, _ in check_basics(right, matches, r2, m2, 1):
        assert 1 <= abs(r2.at[r, "birth_year"] - orig["birth_year"]) <= 2
    r3, m3 = synth_matches(right, matches, ["first_name"], corruptions=("missing",), random_state=3)
    # every copy loses its first name; the last names differ, so no copy is a duplicate
    assert r3["first_name"].iloc[len(right):].isna().all()
    assert len(m3) == len(matches) + len(right)


def test_n_cols_and_n_per_match(persons):
    left, right, matches = persons
    r2, m2 = synth_matches(right, matches, COLS, n_cols=2, n_per_match=2,
                           corruptions=("ocr", "typo", "shift"), random_state=4)
    rows = list(check_basics(right, matches, r2, m2, 2))
    assert len(rows) > len(matches)  # up to two per match, minus duplicates
    assert any(len(c) == 2 for *_, c in rows)


def test_reproducible_and_seed_dependent(persons):
    _, right, matches = persons
    a = synth_matches(right, matches, COLS, corruptions=("ocr", "typo", "shift"), random_state=5)
    b = synth_matches(right, matches, COLS, corruptions=("ocr", "typo", "shift"), random_state=5)
    c = synth_matches(right, matches, COLS, corruptions=("ocr", "typo", "shift"), random_state=6)
    assert a[0].equals(b[0]) and a[1].equals(b[1])
    assert not a[0].equals(c[0])


def test_no_duplicates_of_existing_rows(persons):
    _, right, matches = persons
    r2, _ = synth_matches(right, matches, COLS, corruptions=("ocr", "typo", "shift"), n_per_match=5, random_state=7)
    assert not r2[COLS].duplicated().any()


def test_sample_share_zero_adds_nothing(persons):
    _, right, matches = persons
    r2, m2 = synth_matches(right, matches, COLS, sample_share=0.0, random_state=0)
    assert r2.equals(right) and m2.equals(matches)


def test_sample_share_selects_matches(persons):
    _, right, matches = persons
    _, m2 = synth_matches(right, matches, ["last_name"], corruptions=("typo",), sample_share=0.5, random_state=0)
    assert len(m2) - len(matches) == 6


def test_errors(persons):
    _, right, matches = persons
    with pytest.raises(ValueError, match="Unknown corruptions"):
        synth_matches(right, matches, COLS, corruptions=("ocr", "blur"))
    with pytest.raises(ValueError, match="not in right"):
        synth_matches(right, matches, ["parish"])
    with pytest.warns(UserWarning, match="variant"):
        synth_matches(right, matches, COLS, corruptions=("variant",))


def test_new_ids(persons):
    _, right, matches = persons
    right = right.assign(pid=[f"p{i}" for i in range(len(right))])
    r2, _ = synth_matches(right, matches, ["last_name"], corruptions=("typo",), id_right="pid", random_state=0)
    assert r2["pid"].is_unique and r2["pid"].notna().all()


def test_after_split_on_the_training_subset(persons):
    left, right, matches = persons
    lt, rt, mt, *_ = split_test_train(left, right, matches, 0.25, 0.0, random_state=0)
    v = learn_variants(lt, rt, mt, ["first_name"])
    rt2, mt2 = synth_matches(rt, mt, COLS, corruptions=("ocr", "typo", "variant", "shift"),
                             variants=v, random_state=0)
    new = rt2.index[len(rt):]
    assert len(new) > 0
    # synthetic rows have no original label, so pairs in original labels never point to them
    assert rt2.loc[new, "index_original"].isna().all()
    assert rt2["index_original"].dtype == "Int64"
    inside = pairs_in_subset(matches, lt, rt2)
    assert len(inside) == len(mt) and not set(inside["right"]) & set(new)
    # each new pair links a training left record to its source's corrupted copy
    source_of = dict(zip(mt["left"], mt["right"]))
    for l, r in zip(mt2["left"].iloc[len(mt):], mt2["right"].iloc[len(mt):]):
        differ = [c for c in COLS if rt2.at[r, c] != rt.at[source_of[l], c]]
        assert len(differ) == 1  # n_cols=1

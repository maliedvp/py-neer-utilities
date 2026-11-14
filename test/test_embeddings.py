import json
import os
import pickle
import zlib

import numpy as np
import pandas as pd
import pytest
from neer_match import similarity_map as nm_sim
from neer_match.similarity_encoding import SimilarityEncoder
from neer_match.similarity_map import SimilarityMap

from neer_match_utilities.custom_similarities import CustomSimilarities
from neer_match_utilities.embeddings import EmbeddingSimilarity
from neer_match_utilities.similarity_features import SimilarityFeatures


class TrigramEncoder:
    """Deterministic stand-in for a sentence encoder: hashed character trigrams.

    Records every call, so that tests can check what was encoded.
    """

    def __init__(self, dim=64):
        self.dim = dim
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            padded = f"  {text.lower()} "
            for k in range(len(padded) - 2):
                out[i, zlib.crc32(padded[k:k + 3].encode()) % self.dim] += 1.0
        return out


def refuse(texts):
    raise AssertionError(f"nothing should be encoded, got {texts}")


@pytest.fixture
def data():
    left = pd.DataFrame({
        "id": ["L0", "L1", "L2"],
        "name": ["Gustaf Andersson", "Johann Fischer", None],
        "city": ["Uppsala", "Berlin", ""],
    })
    right = pd.DataFrame({
        "id": ["R0", "R1", "R2"],
        "name": ["Gustav Andersson", "Johannes Fischer", "Maria Lind"],
        "city": ["Upsala", "Berlin", np.nan],
    })
    matches = pd.DataFrame({"left": [0, 1], "right": [0, 1]})
    return left, right, matches


def test_identical_values_score_one_and_missing_values_zero(tmp_path, data):
    left, right, _ = data
    emb = EmbeddingSimilarity(tmp_path, encoder=TrigramEncoder()).encode(left, right, columns=["name", "city"])

    assert emb.similarity("Berlin", "Berlin") == pytest.approx(1.0)
    assert 0.0 < emb.similarity("Gustaf Andersson", "Gustav Andersson") < 1.0
    assert emb.similarity("Gustaf Andersson", "Gustav Andersson") > emb.similarity("Gustaf Andersson", "Maria Lind")
    for missing in (None, np.nan, pd.NA, "", "  "):
        assert emb.similarity(missing, "Berlin") == 0.0
        assert emb.similarity("Berlin", missing) == 0.0


def test_missing_and_empty_values_are_not_encoded(tmp_path, data):
    left, right, _ = data
    enc = TrigramEncoder()
    emb = EmbeddingSimilarity(tmp_path, encoder=enc).encode(left, right, columns=["name", "city"])

    encoded = [t for call in enc.calls for t in call]
    assert sorted(encoded) == sorted(set(encoded))
    assert set(emb.values) == {
        "Gustaf Andersson", "Johann Fischer", "Gustav Andersson", "Johannes Fischer",
        "Maria Lind", "Uppsala", "Berlin", "Upsala",
    }


def test_cache_is_read_again_and_only_new_values_are_encoded(tmp_path, data):
    left, right, _ = data
    first = EmbeddingSimilarity(tmp_path, model_name="m", encoder=TrigramEncoder()).encode(
        left, right, columns=["name"]
    )
    expected = first.similarity("Johann Fischer", "Johannes Fischer")

    again = EmbeddingSimilarity(tmp_path, model_name="m", encoder=refuse)
    assert len(again) == 5
    assert again.similarity("Johann Fischer", "Johannes Fischer") == pytest.approx(expected)
    again.encode(left, right, columns=["name"])          # nothing new

    enc = TrigramEncoder()
    more = EmbeddingSimilarity(tmp_path, model_name="m", encoder=enc)
    more.encode(pd.DataFrame({"name": ["Johann Fischer", "Johan Fiscer"]}), columns=["name"])
    assert enc.calls == [["Johan Fiscer"]]
    assert len(EmbeddingSimilarity(tmp_path, model_name="m", encoder=refuse)) == 6

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["model_name"] == "m" and meta["n_values"] == 6 and meta["dim"] == 64


def test_a_value_that_was_not_encoded_raises(tmp_path, data):
    left, right, _ = data
    emb = EmbeddingSimilarity(tmp_path, encoder=TrigramEncoder()).encode(left, right, columns=["name"])

    with pytest.raises(KeyError, match="encode the data"):
        emb.similarity("Johann Fischer", "Hans Fischer")


def test_cache_of_another_model_or_revision_raises(tmp_path, data):
    left, right, _ = data
    EmbeddingSimilarity(tmp_path, model_name="a", revision="r1", encoder=TrigramEncoder()).encode(
        left, columns=["name"]
    )

    with pytest.raises(ValueError, match="made with 'a'"):
        EmbeddingSimilarity(tmp_path, model_name="b", encoder=refuse)
    with pytest.raises(ValueError, match="revision 'r1'"):
        EmbeddingSimilarity(tmp_path, model_name="a", revision="r2", encoder=refuse)
    assert EmbeddingSimilarity(tmp_path, model_name="a", encoder=refuse).revision == "r1"


def test_inconsistent_cache_raises(tmp_path, data):
    left, right, _ = data
    EmbeddingSimilarity(tmp_path, encoder=TrigramEncoder()).encode(left, columns=["name"])
    (tmp_path / "values.json").write_text(json.dumps(["only one"]))

    with pytest.raises(ValueError, match="inconsistent"):
        EmbeddingSimilarity(tmp_path, encoder=refuse)


def test_missing_column_and_wrong_encoder_output_raise(tmp_path, data):
    left, _, _ = data
    with pytest.raises(KeyError, match="not in the data"):
        EmbeddingSimilarity(tmp_path, encoder=TrigramEncoder()).encode(left, columns=["surname"])
    with pytest.raises(ValueError, match="one vector per value"):
        EmbeddingSimilarity(tmp_path / "x", encoder=lambda t: np.ones((1, 4))).encode(left, columns=["name"])


def test_encoding_without_sentence_transformers_says_what_to_install(tmp_path, data, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_st(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_st)
    left, _, _ = data
    with pytest.raises(ImportError, match=r"neer-match-utilities\[embeddings\]"):
        EmbeddingSimilarity(tmp_path).encode(left, columns=["name"])


def test_similarity_features_use_it_by_name(tmp_path, data):
    left, right, matches = data
    emb = EmbeddingSimilarity(tmp_path, encoder=TrigramEncoder()).encode(left, right, columns=["name", "city"])

    frame = SimilarityFeatures(SimilarityMap({
        "name": ["jaro_winkler", "embedding_cosine"],
        "city": ["embedding_cosine"],
    })).pairwise_similarity_dataframe(left, right, matches, "id", "id")

    assert len(frame) == 9
    for _, row in frame.iterrows():
        lrow = left.set_index("id").loc[row["id_left"]]
        rrow = right.set_index("id").loc[row["id_right"]]
        assert row["col_name_name_embedding_cosine"] == pytest.approx(emb.similarity(lrow["name"], rrow["name"]))
        assert row["col_city_city_embedding_cosine"] == pytest.approx(emb.similarity(lrow["city"], rrow["city"]))


def test_neer_match_similarity_encoder_uses_it(tmp_path, data):
    """The neural network computes its similarities with neer_match's SimilarityEncoder."""
    left, right, _ = data
    emb = EmbeddingSimilarity(tmp_path, encoder=TrigramEncoder()).encode(left, right, columns=["name"])

    encoder = SimilarityEncoder(SimilarityMap({"name": ["embedding_cosine", "jaro_winkler"]}))
    matrix = encoder.encode_as_matrix(left.iloc[[0, 1]], right.iloc[[0, 1]])

    assert matrix[0, 0] == pytest.approx(emb.similarity("Gustaf Andersson", "Gustav Andersson"))
    assert matrix[1, 0] == pytest.approx(emb.similarity("Johann Fischer", "Johannes Fischer"))


def test_other_registered_similarities_are_kept(tmp_path, data):
    left, right, _ = data
    CustomSimilarities()
    EmbeddingSimilarity(tmp_path, encoder=TrigramEncoder(), similarity_name="emb_a").encode(left, columns=["name"])
    EmbeddingSimilarity(tmp_path / "b", encoder=TrigramEncoder(), similarity_name="emb_b").encode(left, columns=["name"])

    names = nm_sim.available_similarities()
    assert {"emb_a", "emb_b", "notmissing", "jaro_winkler"} <= set(names)
    assert names["basic_ratio"]("abc", "abc") == pytest.approx(1.0)   # fuzz rescaling kept


def test_pickled_object_gives_the_same_similarities(tmp_path, data):
    left, right, _ = data
    emb = EmbeddingSimilarity(tmp_path, encoder=TrigramEncoder()).encode(left, right, columns=["name"])

    copy = pickle.loads(pickle.dumps(emb))

    assert copy.similarity("Gustaf Andersson", "Gustav Andersson") == pytest.approx(
        emb.similarity("Gustaf Andersson", "Gustav Andersson")
    )


@pytest.mark.skipif(
    not os.environ.get("NEER_TEST_EMBEDDINGS"),
    reason="downloads BAAI/bge-m3 (2.3 GB); set NEER_TEST_EMBEDDINGS=1 to run",
)
def test_bge_m3(tmp_path):
    pytest.importorskip("sentence_transformers")
    emb = EmbeddingSimilarity(tmp_path, device="cpu")
    emb.encode(pd.DataFrame({"v": ["Aktiengesellschaft", "Act.-Ges.", "Braunkohle", "Gustaf", "Gustav"]}), columns=["v"])

    assert emb.dim == 1024 and emb.revision
    assert emb.similarity("Gustaf", "Gustav") > emb.similarity("Gustaf", "Braunkohle")
    assert EmbeddingSimilarity(tmp_path, encoder=refuse).similarity("Gustaf", "Gustaf") == pytest.approx(1.0)

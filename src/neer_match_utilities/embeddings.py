"""
Similarity of field values from a pretrained text encoder (sentence embeddings).

Each distinct value of the chosen columns is encoded once by a pretrained model
(by default ``BAAI/bge-m3``, multilingual); the similarity of two values is the
cosine of their vectors. It is registered like any other similarity, so it can
be named in the similarity map, and then the logit, probit and gradient-boosting
models (through :class:`~neer_match_utilities.similarity_features.SimilarityFeatures`),
the neural network of neer_match (through its ``SimilarityEncoder``) and the
feature selection all use it.

The vectors are kept in a cache folder. Encoding needs the optional packages
``sentence-transformers`` and ``torch`` (``pip install
"neer-match-utilities[embeddings]"``); reading a cache needs only numpy, so the
models can be trained in an environment without torch, and the cache is what a
replication package stores.

Encoding runs in a Python process of its own (``_encode_worker.py``), which
loads neither this package nor TensorFlow: on Linux with a GPU, TensorFlow and
triton (loaded by sentence-transformers) cannot be loaded into one process in
this order.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .custom_similarities import register_similarities

DEFAULT_MODEL = "BAAI/bge-m3"
_VALUES, _VECTORS, _META = "values.json", "vectors.npy", "meta.json"
_WORKER = Path(__file__).with_name("_encode_worker.py")


def _sentence_transformers_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def _is_missing(value) -> bool:
    """None, NaN, pandas.NA, or a string that is empty after stripping."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _atomic_write(path: Path, write: Callable) -> None:
    """Write to a temporary file in the same folder, then replace `path`."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _resolve_revision(model_name: str) -> str | None:
    """Commit of the model on the Hugging Face Hub, or None if it cannot be found."""
    if Path(model_name).exists():
        return None
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model_name).sha
    except Exception:
        warnings.warn(
            f"Could not look up the current commit of {model_name!r}; the cache records "
            "no revision. Pass revision= to pin the model.",
            stacklevel=3,
        )
        return None


class EmbeddingSimilarity:
    """
    Cosine similarity of sentence embeddings, available to neer_match by name.

    Parameters
    ----------
    cache_dir : str or Path
        Folder for the vectors (``vectors.npy``), the encoded values
        (``values.json``) and the model they come from (``meta.json``). If it
        holds a cache already, that cache is read, and only values not in it are
        encoded later. The values are the data themselves: store the folder
        under the same terms as the data.
    model_name : str, optional
        Model on the Hugging Face Hub, or a local path, loaded with
        ``sentence_transformers.SentenceTransformer``. Default ``"BAAI/bge-m3"``.
        A cache made with another model raises ValueError.
    revision : str or None, optional
        Commit of the model to use. Default None: the commit recorded in the
        cache, or else the current one on the Hub, which is then recorded.
    similarity_name : str, optional
        Name under which the similarity is registered and used in the similarity
        map. Default ``"embedding_cosine"``. Use one name per model if several
        are compared.
    batch_size : int, optional
        Values encoded at once. Default 64.
    max_seq_length : int, optional
        Tokens per value; longer values are cut. Default 512 (bge-m3 accepts
        up to 8192, which field values rarely need).
    device : str or None, optional
        Device for encoding, e.g. ``"cuda"`` or ``"cpu"``. Default None: the
        GPU if one is available.
    fp16 : bool, optional
        Encode in half precision (GPU only). Default False.
    cuda_visible_devices : str or None, optional
        GPUs the encoding process may use, as in the environment variable
        ``CUDA_VISIBLE_DEVICES`` (e.g. ``"1"``). Default None: the same GPUs as
        this process. Lets this process hide the GPUs from TensorFlow
        (``CUDA_VISIBLE_DEVICES=``), which otherwise may take a GPU's memory
        when the package is imported, while the model encodes on one.
    encoder : callable or None, optional
        Instead of the model: a function that takes a list of strings and returns
        one vector per string (e.g. for tests or another library). Default None.

    Notes
    -----
    - A value is looked up exactly as it appears in the data (as ``str(value)``),
      so encode the columns in the form they have when the similarities are
      computed, i.e. after ``Prepare.format``, and encode again after adding
      rows, e.g. with ``synth_matches`` or ``synth_mismatches``: only the new
      values are encoded. A value that was not encoded raises KeyError.
    - Missing values, and strings that are empty, have similarity 0, as in
      :class:`~neer_match_utilities.similarity_features.SimilarityFeatures`.
    - The cosine is cut to [0, 1], the range of the other similarities.
    - The vectors are normalized and kept as float32: 4 bytes times the
      dimension (1024 for bge-m3) per distinct value.
    - To use a saved model later, create the object again from the cache before
      predicting, so that the similarity is registered.
    - Each call of :meth:`encode` that finds new values starts a Python process
      that loads the model, encodes them and ends (see the module notes);
      encode all frames in one call where possible.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        model_name: str = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        similarity_name: str = "embedding_cosine",
        batch_size: int = 64,
        max_seq_length: int = 512,
        device: str | None = None,
        fp16: bool = False,
        cuda_visible_devices: str | None = None,
        encoder: Callable[[list], np.ndarray] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.revision = revision
        self.similarity_name = similarity_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = device
        self.fp16 = fp16
        self.cuda_visible_devices = cuda_visible_devices
        self._encoder = encoder

        self._values: list[str] = []
        self._index: dict[str, int] = {}
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._rows: list[np.ndarray] = []
        if (self.cache_dir / _META).exists():
            self._read_cache()

        register_similarities({similarity_name: self.similarity})

    # ------------------------------------------------------------------ data
    def __len__(self) -> int:
        return len(self._values)

    @property
    def dim(self) -> int:
        """Length of the vectors (0 before anything is encoded)."""
        return int(self._vectors.shape[1]) if len(self) else 0

    @property
    def values(self) -> list:
        """The encoded values, in the order of the rows of :attr:`vectors`."""
        return list(self._values)

    @property
    def vectors(self) -> np.ndarray:
        """The normalized vectors, one row per encoded value."""
        return self._vectors

    # ------------------------------------------------------------ similarity
    def similarity(self, x, y) -> float:
        """Cosine of the vectors of `x` and `y`, cut to [0, 1]; 0 if either is missing."""
        if _is_missing(x) or _is_missing(y):
            return 0.0
        try:
            i, j = self._index[str(x)], self._index[str(y)]
        except KeyError as e:
            raise KeyError(
                f"{e.args[0]!r} has no embedding in {self.cache_dir}: encode the data "
                "that contain it first (also after rows were added, e.g. by "
                "synth_matches or synth_mismatches)."
            ) from None
        return min(max(float(np.dot(self._rows[i], self._rows[j])), 0.0), 1.0)

    # -------------------------------------------------------------- encoding
    def encode(self, *frames: pd.DataFrame, columns: Sequence[str]) -> "EmbeddingSimilarity":
        """
        Encode the values of `columns` in `frames` that are not in the cache yet.

        Parameters
        ----------
        *frames : pandas.DataFrame
            Data whose values are encoded, e.g. ``left, right``. Each must have
            all `columns`; if the left and right names differ, call once per
            frame.
        columns : sequence of str
            Columns to encode.

        Returns
        -------
        EmbeddingSimilarity
            The object itself; the cache folder is updated if anything was new.
        """
        new: list[str] = []
        seen = set(self._index)
        for frame in frames:
            missing = [c for c in columns if c not in frame.columns]
            if missing:
                raise KeyError(f"Columns not in the data: {missing}")
            for col in columns:
                for value in pd.unique(frame[col].to_numpy(dtype=object)):
                    if _is_missing(value):
                        continue
                    key = str(value)
                    if key not in seen:
                        seen.add(key)
                        new.append(key)
        if new:
            self._append(new, self._encode_texts(new))
            self._write_cache()
        return self

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if self._encoder is not None:
            vectors = self._encoder(texts)
        else:
            vectors = self._encode_in_subprocess(texts)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError(
                f"The encoder returned shape {vectors.shape} for {len(texts)} values; "
                "expected one vector per value."
            )
        if len(self) and vectors.shape[1] != self.dim:
            raise ValueError(f"The encoder returned vectors of length {vectors.shape[1]}, the cache has {self.dim}.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms > 0, norms, 1.0)

    def _encode_in_subprocess(self, texts: list[str]) -> np.ndarray:
        """Encode `texts` with the model in a Python process of its own (see the module notes)."""
        if not _sentence_transformers_available():
            raise ImportError(
                "Encoding needs sentence-transformers and torch: "
                'pip install "neer-match-utilities[embeddings]". '
                "Reading an existing cache does not."
            )
        if self.revision is None:
            self.revision = _resolve_revision(self.model_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.cache_dir, prefix=".encode-") as tmp:
            job, out = Path(tmp) / "job.json", Path(tmp) / "vectors.npy"
            job.write_text(json.dumps({
                "model_name": self.model_name,
                "revision": self.revision,
                "device": self.device,
                "fp16": self.fp16,
                "max_seq_length": self.max_seq_length,
                "batch_size": self.batch_size,
                "texts": texts,
                "out": str(out),
            }, ensure_ascii=False), encoding="utf-8")
            # -I: the worker's folder (this package) is not put on sys.path
            env = os.environ.copy()
            if self.cuda_visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
            done = subprocess.run([sys.executable, "-I", str(_WORKER), str(job)], env=env)
            if done.returncode != 0 or not out.exists():
                raise RuntimeError(
                    f"Encoding {len(texts)} values with {self.model_name!r} failed "
                    f"(exit code {done.returncode}); see the output above."
                )
            return np.load(out)

    def _append(self, texts: list[str], vectors: np.ndarray) -> None:
        start = len(self._values)
        self._values.extend(texts)
        self._index.update({t: start + k for k, t in enumerate(texts)})
        self._vectors = vectors if start == 0 else np.vstack([self._vectors, vectors])
        self._rows = list(self._vectors)

    # ----------------------------------------------------------------- cache
    def _meta(self) -> dict:
        try:
            from importlib.metadata import version
            st_version = version("sentence-transformers")
        except Exception:
            st_version = None
        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "encoder": "sentence-transformers" if self._encoder is None else "custom",
            "sentence_transformers": st_version if self._encoder is None else None,
            "dim": self.dim,
            "max_seq_length": self.max_seq_length,
            "normalized": True,
            "n_values": len(self),
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _write_cache(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.cache_dir / _VECTORS, lambda f: np.save(f, self._vectors))
        _atomic_write(
            self.cache_dir / _VALUES,
            lambda f: f.write(json.dumps(self._values, ensure_ascii=False).encode("utf-8")),
        )
        _atomic_write(
            self.cache_dir / _META,
            lambda f: f.write(json.dumps(self._meta(), indent=2).encode("utf-8")),
        )

    def _read_cache(self) -> None:
        meta = json.loads((self.cache_dir / _META).read_text(encoding="utf-8"))
        if meta.get("model_name") != self.model_name:
            raise ValueError(
                f"The cache in {self.cache_dir} was made with {meta.get('model_name')!r}, "
                f"not {self.model_name!r}. Use another cache_dir per model."
            )
        cached_revision = meta.get("revision")
        if self.revision is not None and cached_revision is not None and self.revision != cached_revision:
            raise ValueError(
                f"The cache in {self.cache_dir} was made with revision {cached_revision!r} "
                f"of {self.model_name!r}, not {self.revision!r}."
            )
        self.revision = self.revision or cached_revision
        values = json.loads((self.cache_dir / _VALUES).read_text(encoding="utf-8"))
        vectors = np.load(self.cache_dir / _VECTORS)
        if vectors.ndim != 2 or len(values) != vectors.shape[0]:
            raise ValueError(
                f"The cache in {self.cache_dir} is inconsistent: {len(values)} values, "
                f"vectors of shape {vectors.shape}."
            )
        self._values, self._index = [], {}
        self._append(values, vectors.astype(np.float32, copy=False))

    # --------------------------------------------------------------- pickling
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_rows"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._rows = list(self._vectors)

"""
Encode values with sentence-transformers, in a process of its own.

:class:`~neer_match_utilities.embeddings.EmbeddingSimilarity` runs this file as
a script (``python -I _encode_worker.py job.json``), not as part of the
package, so that the process loads neither the package nor TensorFlow.
TensorFlow and triton, which sentence-transformers loads on Linux with a GPU,
each bring their own copy of LLVM; once TensorFlow is loaded, loading triton
aborts the process ("LLVM ERROR: inconsistency in registered CommandLine
options"). A process of its own also frees the GPU memory when it ends.

The job file is JSON with the keys model_name, revision, device, fp16,
max_seq_length, batch_size, texts and out (path of the .npy file to write).
"""
import json
import sys

import numpy as np


def main(job_path: str) -> None:
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(job["model_name"], revision=job["revision"], device=job["device"])
    model.max_seq_length = job["max_seq_length"]
    if job["fp16"] and model.device.type == "cuda":
        model.half()
    elif job["fp16"]:
        print("fp16 is used on a GPU only; encoding in float32.", file=sys.stderr)

    texts = job["texts"]
    vectors = model.encode(
        texts,
        batch_size=job["batch_size"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) >= 10_000,
    )
    np.save(job["out"], np.asarray(vectors, dtype=np.float32))


if __name__ == "__main__":
    main(sys.argv[1])

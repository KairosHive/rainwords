"""
Lay out every built-in passage on a 2D map of meaning, for the corpus atlas
in the map window (/map).

Run it again whenever poetry.index is rebuilt:

    python -m rainwords.build_corpus_atlas

Needs umap-learn, a build-time-only dependency (pip install umap-learn); the
server only reads the JSON it writes. Uploaded corpora are placed at runtime
by interpolating from their nearest built-in passages (see main.py).
"""
import json
import pickle
from pathlib import Path

import faiss
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "corpus_atlas.json"


def fit_unit_square(Y: np.ndarray, pad: float = 0.03) -> np.ndarray:
    """Scale a layout into [0, 1] with one scale for both axes, centred, with
    the long side horizontal (UMAP's orientation is arbitrary; the map window
    is landscape).

    Uses the 1-99 percentile box so a small far-off island (e.g. a running page
    header repeated as its own passage) doesn't squeeze everything else; points
    outside it are pinned to the edge of the map.
    """
    lo, hi = np.percentile(Y, 1, axis=0), np.percentile(Y, 99, axis=0)
    if (hi - lo)[1] > (hi - lo)[0]:
        Y, lo, hi = Y[:, ::-1], lo[::-1], hi[::-1]
    span = float((hi - lo).max()) * (1 + 2 * pad)
    out = (Y - lo) / span + (1 - (hi - lo) / span) / 2
    return np.clip(out, 0.0, 1.0)


def main():
    import umap

    index = faiss.read_index(str(BASE / "poetry.index"))
    X = index.reconstruct_n(0, index.ntotal).astype("float32")
    with open(BASE / "poetry_docs.pkl", "rb") as f:
        docs = pickle.load(f)
    assert len(docs) == X.shape[0], "poetry_docs.pkl and poetry.index disagree"

    print(f"UMAP over {X.shape[0]} passages ({X.shape[1]} dims)...")
    Y = umap.UMAP(
        n_neighbors=30, min_dist=0.15, metric="cosine", random_state=7
    ).fit_transform(X)

    Y = fit_unit_square(Y)

    points: dict = {}
    for d, (x, y) in zip(docs, Y):
        points.setdefault(d["source"], []).append([round(float(x), 4), round(float(y), 4)])

    OUT.write_text(
        json.dumps({"method": "umap cosine n_neighbors=30 min_dist=0.15", "points": points},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {OUT.name}: {sum(len(v) for v in points.values())} points in {len(points)} corpora")


if __name__ == "__main__":
    main()

"""
Face clustering module.

In plain English: once every photo has been scanned and every face turned
into a 512-dimensional embedding vector, we need to group the vectors so
that every photo of the same person ends up in one cluster. We use
DBSCAN with cosine distance: points within a small angle of each other
get grouped, isolated points stay un-clustered (label = -1, "unknown").

Optional HDBSCAN cross-verification (when the `hdbscan` package is
installed) runs a second clustering pass and flags clusters whose two
algorithms disagree — a useful sanity check on noisy data.

Clusters face embeddings into person groups using DBSCAN with optional
HDBSCAN cross-verification.

CLI usage:
    python -m backend.face_cluster --method dbscan
    python -m backend.face_cluster --method dbscan --verify hdbscan
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("memoria.ml.face_cluster")

try:
    import metrics as _metrics  # type: ignore
except ImportError:
    try:
        import sys as _sys
        _backend = Path(__file__).resolve().parent.parent
        if str(_backend) not in _sys.path:
            _sys.path.insert(0, str(_backend))
        import metrics as _metrics  # type: ignore
    except Exception:  # pragma: no cover — CLI fallback
        from contextlib import contextmanager as _ctx

        class _NoopMetrics:
            @staticmethod
            @_ctx
            def timer(*_a, **_k):
                yield {}

            @staticmethod
            def count(*_a, **_k):
                pass

            @staticmethod
            def record(*_a, **_k):
                pass

        _metrics = _NoopMetrics()


def cluster_dbscan(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 2,
) -> np.ndarray:
    """Run DBSCAN clustering on L2-normalized embeddings with cosine distance."""
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine",
        n_jobs=-1,
    ).fit(embeddings)
    return clustering.labels_


def cluster_dbscan_precomputed(
    distances_csr,
    eps: float = 0.5,
    min_samples: int = 2,
) -> np.ndarray:
    """DBSCAN over a precomputed sparse cosine-distance matrix.

    Edges absent from the CSR are treated as > eps. The diagonal must be 0;
    callers should pre-filter so all stored entries satisfy dist <= eps.
    """
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed",
        n_jobs=-1,
    ).fit(distances_csr)
    return clustering.labels_


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
) -> np.ndarray:
    """Run HDBSCAN on L2-normalized embeddings (euclidean ≈ cosine for unit vectors)."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )
    clusterer.fit(embeddings)
    return clusterer.labels_


def cross_verify(
    labels_primary: np.ndarray,
    labels_secondary: np.ndarray,
) -> np.ndarray:
    """Compare two clustering results. Return confidence array.

    'high' for faces where both methods agree, 'low' where they disagree.
    """
    n = len(labels_primary)
    confidence = np.array(["high"] * n, dtype=object)

    for i in range(n):
        if labels_primary[i] == -1:
            confidence[i] = "low"
            continue
        for j in range(i + 1, n):
            same_primary = labels_primary[i] == labels_primary[j]
            same_secondary = labels_secondary[i] == labels_secondary[j]
            if same_primary != same_secondary:
                confidence[i] = "low"
                confidence[j] = "low"

    return confidence


def run_clustering(
    embeddings: np.ndarray,
    method: str = "dbscan",
    verify: str | None = None,
    eps: float = 0.5,
    min_samples: int = 2,
    min_cluster_size: int = 3,
) -> dict[str, Any]:
    """Run clustering pipeline and return results.

    Returns:
        {
            "labels": np.ndarray of cluster IDs (-1 = noise),
            "confidence": np.ndarray of "high"/"low",
            "n_clusters": int,
            "n_noise": int,
        }
    """
    logger.info("Running %s clustering on %d embeddings", method, len(embeddings))
    _metrics.record("face.cluster.n_in", len(embeddings), method=method)

    with _metrics.timer("face.cluster", method=method):
        if method == "dbscan":
            labels = cluster_dbscan(embeddings, eps=eps, min_samples=min_samples)
        elif method == "hdbscan":
            labels = cluster_hdbscan(embeddings, min_cluster_size=min_cluster_size)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    confidence = np.array(["high"] * len(labels), dtype=object)

    if verify:
        logger.info("Cross-verifying with %s", verify)
        with _metrics.timer("face.cluster.verify", method=verify):
            if verify == "hdbscan":
                labels_v = cluster_hdbscan(embeddings, min_cluster_size=min_cluster_size)
            elif verify == "dbscan":
                labels_v = cluster_dbscan(embeddings, eps=eps, min_samples=min_samples)
            else:
                raise ValueError(f"Unknown verification method: {verify}")
            confidence = cross_verify(labels, labels_v)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    logger.info("Found %d clusters, %d noise points", n_clusters, n_noise)
    _metrics.record("face.cluster.n_out", n_clusters, method=method)
    if len(labels) > 0:
        _metrics.record(
            "face.cluster.noise_pct", 100.0 * n_noise / len(labels), method=method
        )

    return {
        "labels": labels,
        "confidence": confidence,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Face clustering")
    parser.add_argument("--method", default="dbscan", choices=["dbscan", "hdbscan"])
    parser.add_argument("--verify", default=None, choices=["dbscan", "hdbscan"])
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--min-cluster-size", type=int, default=3)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # In a real setup, embeddings would be loaded from the database
    logger.info("Clustering requires face embeddings from the database.")
    logger.info("Run face detection first: python -m backend.face_pipeline --reprocess")
    print("\nTo test clustering with random data:")
    print("  python -c \"")
    print("  import numpy as np")
    print("  from backend.face_cluster import run_clustering")
    print("  emb = np.random.randn(50, 512).astype(np.float32)")
    print("  emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)")
    print("  result = run_clustering(emb, method='dbscan')")
    print("  print(result['n_clusters'], 'clusters,', result['n_noise'], 'noise')\"")


if __name__ == "__main__":
    main()

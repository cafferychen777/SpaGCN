import warnings

import numpy as np
import scanpy as sc


def graph_cluster_labels(
    values: np.ndarray,
    *,
    method: str,
    n_neighbors: int,
    resolution: float,
) -> np.ndarray:
    """Initialize clusters with the maintained igraph Leiden implementation."""
    if method == "louvain":
        warnings.warn(
            "init='louvain' is deprecated and now uses Leiden initialization. "
            "Use init='leiden' explicitly.",
            FutureWarning,
            stacklevel=2,
        )
    elif method != "leiden":
        raise ValueError("init must be 'leiden', 'louvain', or 'kmeans'")

    adata = sc.AnnData(values)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.leiden(
        adata,
        resolution=resolution,
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )
    return adata.obs["leiden"].astype(int).to_numpy()

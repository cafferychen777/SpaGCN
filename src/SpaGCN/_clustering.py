import numpy as np
import scanpy as sc

from ._louvain import louvain_membership


def graph_cluster_labels(
    values: np.ndarray,
    *,
    method: str,
    n_neighbors: int,
    resolution: float,
) -> np.ndarray:
    """Cluster a kNN graph of ``values`` with Louvain or Leiden.

    ``louvain`` reproduces ``scanpy.tl.louvain(adata, resolution=resolution)``
    as called by SpaGCN 1.2.7 (unweighted directed graph, seed 0) without the
    unmaintained ``louvain`` extension. ``leiden`` uses igraph's Leiden.
    """
    if method not in {"leiden", "louvain"}:
        raise ValueError("init must be 'louvain', 'leiden', or 'kmeans'")

    adata = sc.AnnData(values)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    if method == "louvain":
        return louvain_membership(
            adata.obsp["connectivities"], resolution=resolution, seed=0
        )
    sc.tl.leiden(
        adata,
        resolution=resolution,
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )
    return adata.obs["leiden"].astype(int).to_numpy()

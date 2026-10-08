"""Regenerate spagcn_127_reference.npz with the original SpaGCN 1.2.7.

Run in an environment where ``import SpaGCN`` resolves to the original
SpaGCN 1.2.7 sdist and ``import louvain`` to louvain 0.8.2 (SciPy >= 1.14 also
needs the sparse ``.A`` alias that SpaGCN 1.2.7 uses):

    python tests/fixtures/generate_spagcn_127_reference.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compat_fixture import (  # noqa: E402
    LOUVAIN_RESOLUTIONS,
    N_DOMAINS,
    make_dataset,
    make_louvain_points,
)

import louvain  # noqa: E402
import SpaGCN as spg  # noqa: E402
from scanpy import _utils  # noqa: E402

assert not hasattr(spg, "__version__") or not spg.__version__.startswith("1.3")

counts, rows, cols, x_pixel, y_pixel = make_dataset()
adata = sc.AnnData(counts)
adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
spg.prefilter_genes(adata, min_cells=3)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
l = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
res = spg.search_res(
    adata,
    adj,
    l,
    N_DOMAINS,
    start=0.7,
    step=0.1,
    tol=5e-3,
    lr=0.05,
    max_epochs=20,
    r_seed=100,
    t_seed=100,
    n_seed=100,
)
clf = spg.SpaGCN()
clf.set_l(l)
random.seed(100)
torch.manual_seed(100)
np.random.seed(100)
clf.train(
    adata,
    adj,
    init_spa=True,
    init="louvain",
    res=res,
    tol=5e-3,
    lr=0.05,
    max_epochs=200,
)
labels, prob = clf.predict()

points = sc.AnnData(make_louvain_points())
sc.pp.neighbors(points, n_neighbors=10)
connectivities = points.obsp["connectivities"].tocoo()
graph = _utils.get_igraph_from_adjacency(points.obsp["connectivities"], directed=True)
memberships = [
    louvain.find_partition(
        graph,
        louvain.RBConfigurationVertexPartition,
        resolution_parameter=r,
        seed=0,
    ).membership
    for r in LOUVAIN_RESOLUTIONS
]

np.savez_compressed(
    Path(__file__).with_name("spagcn_127_reference.npz"),
    length_scale=l,
    resolution=res,
    labels=np.asarray(labels, dtype=np.int64),
    graph_row=connectivities.row.astype(np.int32),
    graph_col=connectivities.col.astype(np.int32),
    graph_n=np.int64(points.n_obs),
    louvain_memberships=np.asarray(memberships, dtype=np.int64),
)
print("l =", l, "res =", res, "domains =", np.bincount(labels))
print("louvain communities:", [len(set(m)) for m in memberships])

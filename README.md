# SpaGCN Modern

`spagcn-modern` is an independently maintained fork of
[SpaGCN](https://github.com/jianhuupenn/SpaGCN), a graph convolutional network
for identifying spatial domains and spatially variable genes from gene
expression, spatial coordinates, and histology.

The distribution keeps the original `SpaGCN` import package and API while
supporting current Python, SciPy, Scanpy, and AnnData releases. By default it
reproduces the published SpaGCN 1.2.7 results label for label, without the
unmaintained `louvain` extension.

## Installation

```bash
pip install spagcn-modern
```

The import name remains unchanged:

```python
import SpaGCN
```

## Compatibility modes

Every entry point that trains or tunes the model (`SpaGCN.train`,
`multiSpaGCN.train`, `search_res`, `search_l`, `calculate_p`,
`detect_subclusters`, and `detect_spatial_domains_ez_mode`) accepts
`compat`:

| `compat` | Behavior |
| --- | --- |
| `"spagcn-1.2.7"` (default) | Reproduces SpaGCN 1.2.7: cluster centers stay at their initial values, soft assignments use the original expression, Louvain initialization, dense PCA, and float64 preprocessing. |
| `"modern"` | The behavior of spagcn-modern 1.3.x: cluster centers are trained, soft assignments use the DEC Student's t kernel, Leiden initialization, sparse-aware PCA, and float32 preprocessing. |

```python
import SpaGCN as spg

clf = spg.SpaGCN()
clf.set_l(l)
clf.train(adata, adj, res=res)                    # SpaGCN 1.2.7 results
clf.train(adata, adj, res=res, compat="modern")   # corrected DEC model
```

`init` defaults to `"louvain"` in the 1.2.7 mode and to `"leiden"` in the
modern mode; either mode accepts `init="louvain"`, `"leiden"`, or `"kmeans"`
explicitly.

### Why the default is the 1.2.7 behavior

SpaGCN 1.2.7 differs from the method described in the paper in two places:

- **Cluster centers are not trained.** `simple_GC_DEC.fit` built its optimizer
  before creating the centers `mu`, so only the graph convolution was updated.
  The paper optimizes the centers jointly.
- **The soft-assignment kernel.** `q**(alpha+1.0)/2.0` evaluates as
  `(q**(alpha+1))/2` rather than the Student's t exponent `-(alpha+1)/2`
  ([upstream issue #83](https://github.com/jianhuupenn/SpaGCN/issues/83)).

The `"modern"` mode corrects both. However, the published hyperparameters
(Adam, `lr=0.05`, `alpha=0.2`, and the tutorial's resolution search) were tuned
on the 1.2.7 code. On DLPFC slices 151673, 151507, and 151669 with seven
domains, the 1.2.7 mode reproduces SpaGCN 1.2.7 exactly (mean ARI 0.374), while
the modern mode with the same hyperparameters reaches a mean ARI of 0.211.
Results reported with SpaGCN therefore correspond to the 1.2.7 mode, which is
the default. Use `compat="modern"` when you want the corrected model and are
prepared to tune it.

### Louvain without the `louvain` package

SpaGCN 1.2.7 initialized clusters with `scanpy.tl.louvain`, which requires the
`louvain` package. That package is unmaintained, has no wheels for Python 3.12
or later, and requires `igraph<0.12`. This distribution includes a pure-Python
Louvain implementation (`SpaGCN._louvain`) for the configuration Scanpy uses
(unweighted directed kNN graph, configuration-model quality with a resolution
parameter, seed 0). It follows the same node order, random number stream,
tie-breaking, and community numbering, so it returns the same membership as
`louvain` 0.8.2. A regression test compares both the Louvain memberships and
complete SpaGCN labels with a fixture produced by the original SpaGCN 1.2.7.

### Changes from 1.3.x

spagcn-modern 1.3.x always used the modern behavior. Pass `compat="modern"` to
keep 1.3.x results. In 1.3.x, `init="louvain"` was an alias for Leiden; it now
runs Louvain.

## Maintenance scope

This repository contains only the maintained Python distribution. Tutorials,
sample outputs, prebuilt distributions, and paper assets remain available in
the [original repository](https://github.com/jianhuupenn/SpaGCN) and are not
runtime dependencies.

Compatibility fixes in this fork include:

- current SciPy sparse-array conversion through `toarray()`;
- corrected `use_raw` behavior in `plot_log_exp`;
- the optional `compat="modern"` model with trained centers and the DEC
  Student's t kernel;
- native sparse PCA plus cached resolution-search preprocessing;
- bounded Gaussian-kernel workspaces and zero-copy float32 tensor conversion;
- minimal runtime dependency metadata derived from imports;
- a dependency-free Louvain implementation identical to the obsolete `louvain`
  extension, plus igraph Leiden initialization;
- zero-filled (rather than uninitialized) cross-sample adjacency blocks in
  `multiSpaGCN`;
- isolated wheel builds and sparse-data smoke tests in CI.

## Citation

Please cite the original SpaGCN publication:

> Hu J, et al. SpaGCN: Integrating gene expression, spatial location and
> histology to identify spatial domains and spatially variable genes by graph
> convolutional network. *Nature Methods* (2021).

## Attribution and license

SpaGCN was created by Jian Hu and collaborators. This maintained fork preserves
the original MIT license and records its upstream base in
[`NOTICE.md`](NOTICE.md).

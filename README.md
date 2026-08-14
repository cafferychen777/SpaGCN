# SpaGCN Modern

`spagcn-modern` is an independently maintained fork of
[SpaGCN](https://github.com/jianhuupenn/SpaGCN), a graph convolutional network
for identifying spatial domains and spatially variable genes from gene
expression, spatial coordinates, and histology.

The distribution keeps the original `SpaGCN` import package and numerical API
while supporting current Python, SciPy, Scanpy, and AnnData releases. Graph
initialization uses the maintained Leiden implementation built into igraph.

## Installation

```bash
pip install spagcn-modern
```

The import name remains unchanged:

```python
import SpaGCN
```

## Maintenance scope

This repository contains only the maintained Python distribution. Tutorials,
sample outputs, prebuilt distributions, and paper assets remain available in
the [original repository](https://github.com/jianhuupenn/SpaGCN) and are not
runtime dependencies.

Compatibility fixes in this fork include:

- current SciPy sparse-array conversion through `toarray()`;
- corrected `use_raw` behavior in `plot_log_exp`;
- mathematically correct DEC soft assignments using the Student's t kernel;
- native sparse PCA plus cached resolution-search preprocessing;
- bounded Gaussian-kernel workspaces and zero-copy float32 tensor conversion;
- minimal runtime dependency metadata derived from imports;
- maintained igraph Leiden initialization instead of the obsolete Louvain
  extension;
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

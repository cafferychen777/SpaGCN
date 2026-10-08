"""Deterministic inputs shared by the SpaGCN 1.2.7 regression fixture."""

import numpy as np

FIXTURE_SEED = 20240613
N_DOMAINS = 4
LOUVAIN_RESOLUTIONS = (0.2, 0.5, 1.0, 2.0)
MARKER_RATE = 1.8


def make_dataset():
    """Return counts, array coordinates, and pixel coordinates for 300 spots.

    Four horizontal bands with band-specific marker genes on a 20 x 15 grid,
    loosely mimicking cortical layers.
    """
    rng = np.random.default_rng(FIXTURE_SEED)
    rows, cols = np.meshgrid(np.arange(20), np.arange(15), indexing="ij")
    rows = rows.ravel()
    cols = cols.ravel()
    domain = np.minimum(rows // 5, N_DOMAINS - 1)
    n_genes = 120
    rates = np.full((N_DOMAINS, n_genes), 1.0)
    for d in range(N_DOMAINS):
        rates[d, d * 20 : (d + 1) * 20] = MARKER_RATE
    counts = rng.poisson(rates[domain]).astype(np.float64)
    x_pixel = (rows * 100 + 50).astype(int)
    y_pixel = (cols * 100 + 50).astype(int)
    return counts, rows, cols, x_pixel, y_pixel


def make_louvain_points():
    """Weakly clustered points whose kNN graph yields many communities."""
    rng = np.random.default_rng(FIXTURE_SEED + 1)
    centers = rng.normal(scale=2.0, size=(12, 8))
    labels = rng.integers(0, 12, size=500)
    return (centers[labels] + rng.normal(size=(500, 8))).astype(np.float32)

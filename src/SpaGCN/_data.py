import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import PCA


def pca_embedding(expression, n_components, max_dense_bytes=64 << 20):
    """Return a float32 PCA embedding with adaptive sparse-memory handling."""
    if issparse(expression):
        dense_dtype = np.float32 if expression.dtype == np.float32 else np.float64
        dense_bytes = (
            np.prod(expression.shape, dtype=np.int64) * np.dtype(dense_dtype).itemsize
        )
        if dense_bytes <= max_dense_bytes or n_components >= min(expression.shape):
            expression = expression.toarray()
            solver = "auto"
        else:
            solver = "arpack"
    else:
        solver = "auto"

    embedding = PCA(n_components=n_components, svd_solver=solver).fit_transform(
        expression
    )
    return np.ascontiguousarray(embedding, dtype=np.float32)


def gaussian_adjacency(adjacency, length_scale):
    """Build a float32 Gaussian adjacency with one output-sized allocation."""
    if length_scale <= 0:
        raise ValueError("length_scale must be greater than zero")

    adjacency = np.asarray(adjacency)
    result = np.empty(adjacency.shape, dtype=np.float32)
    np.square(adjacency, out=result)
    result *= -0.5 / (length_scale * length_scale)
    np.exp(result, out=result)
    return result


def mean_gaussian_neighborhood(adjacency, length_scale, max_workspace_bytes=64 << 20):
    """Compute the mean Gaussian row sum using bounded temporary memory."""
    if length_scale <= 0:
        raise ValueError("length_scale must be greater than zero")

    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2:
        raise ValueError("adjacency must be a two-dimensional array")
    if adjacency.shape[0] == 0:
        return np.nan

    bytes_per_row = max(1, adjacency.shape[1] * np.dtype(np.float32).itemsize)
    rows_per_block = max(1, max_workspace_bytes // bytes_per_row)
    total = 0.0
    scale = -0.5 / (length_scale * length_scale)
    for start in range(0, adjacency.shape[0], rows_per_block):
        stop = min(start + rows_per_block, adjacency.shape[0])
        workspace = np.empty((stop - start, adjacency.shape[1]), dtype=np.float32)
        np.square(adjacency[start:stop], out=workspace)
        workspace *= scale
        np.exp(workspace, out=workspace)
        total += workspace.sum(dtype=np.float64)
    return total / adjacency.shape[0]

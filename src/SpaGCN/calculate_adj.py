import numpy as np
import numba


@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i] - t2[i]) ** 2
    return np.sqrt(sum)


@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n = X.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = euclid_dist(X[i], X[j])
    return adj


def extract_color(x_pixel=None, y_pixel=None, image=None, beta=49):
    # beta to control the range of neighbourhood when calculate grey vale for one spot
    beta_half = round(beta / 2)
    g = []
    for i in range(len(x_pixel)):
        max_x = image.shape[0]
        max_y = image.shape[1]
        nbs = image[
            max(0, x_pixel[i] - beta_half) : min(max_x, x_pixel[i] + beta_half + 1),
            max(0, y_pixel[i] - beta_half) : min(max_y, y_pixel[i] + beta_half + 1),
        ]
        g.append(np.mean(np.mean(nbs, axis=0), axis=0))
    c0, c1, c2 = [], [], []
    for i in g:
        c0.append(i[0])
        c1.append(i[1])
        c2.append(i[2])
    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    variances = np.array([np.var(c0), np.var(c1), np.var(c2)])
    if variances.sum() == 0:
        return np.zeros_like(c0, dtype=float)
    c3 = (c0 * variances[0] + c1 * variances[1] + c2 * variances[2]) / variances.sum()
    return c3


def calculate_adj_matrix(
    x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True
):
    # x,y,x_pixel, y_pixel are lists
    if histology:
        if x_pixel is None or y_pixel is None or image is None:
            raise ValueError("x_pixel, y_pixel, and image are required with histology")
        if len(x) != len(x_pixel) or len(y) != len(y_pixel):
            raise ValueError("spatial and pixel coordinates must have matching lengths")
        print("Calculating adjacency matrix using histology image...")
        c3 = extract_color(x_pixel=x_pixel, y_pixel=y_pixel, image=image, beta=beta)
        color_std = np.std(c3)
        c4 = np.zeros_like(c3) if color_std == 0 else (c3 - np.mean(c3)) / color_std
        z_scale = np.max([np.std(x), np.std(y)]) * alpha
        z = c4 * z_scale
        z = z.tolist()
        print("Var of x,y,z = ", np.var(x), np.var(y), np.var(z))
        X = np.array([x, y, z]).T.astype(np.float32)
    else:
        print("Calculating adjacency matrix using coordinates only...")
        X = np.array([x, y]).T.astype(np.float32)
    return pairwise_distance(X)

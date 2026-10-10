"""detect_spatial_domains_ez_mode keeps every input spot.

The spatial graph is built from the coordinates of all spots, so a spot whose
expression sums to zero after gene filtering must stay in the expression matrix.
"""

import contextlib
import io
import unittest
import warnings

import numpy as np
import scanpy as sc

from SpaGCN.ez_mode import detect_spatial_domains_ez_mode

from compat_fixture import N_DOMAINS, make_dataset


def run_ez_mode(counts, var_names, x_pixel, y_pixel):
    adata = sc.AnnData(counts)
    adata.var_names = var_names
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(
            detect_spatial_domains_ez_mode(
                adata,
                None,
                x_pixel.tolist(),
                y_pixel.tolist(),
                x_pixel.tolist(),
                y_pixel.tolist(),
                n_clusters=N_DOMAINS,
                histology=False,
            )
        )


class ZeroExpressionSpotTests(unittest.TestCase):
    def test_spots_without_retained_expression_keep_their_place(self):
        counts, _, _, x_pixel, y_pixel = make_dataset()
        var_names = [f"gene{i}" for i in range(counts.shape[1])]
        var_names[-1] = "MT-CO1"  # removed by prefilter_specialgenes
        counts[7] = 0.0  # no counts at all
        counts[23] = 0.0
        counts[23, -1] = 5.0  # counts only in a gene that ez mode filters out

        pred = run_ez_mode(counts, var_names, x_pixel, y_pixel)

        self.assertEqual(pred.shape, (counts.shape[0],))
        self.assertTrue(np.all(np.isfinite(pred.astype(float))))


if __name__ == "__main__":
    unittest.main()

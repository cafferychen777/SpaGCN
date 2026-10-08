"""Regression tests: compat="spagcn-1.2.7" reproduces the original SpaGCN 1.2.7.

The reference in fixtures/spagcn_127_reference.npz was produced once by the
original SpaGCN 1.2.7 and louvain 0.8.2 (fixtures/generate_spagcn_127_reference.py).
"""

import contextlib
import io
import random
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scanpy as sc
import torch
from scipy import sparse

import SpaGCN as spg
from SpaGCN._louvain import louvain_membership
from SpaGCN.models import _initial_centers, simple_GC_DEC

from compat_fixture import LOUVAIN_RESOLUTIONS, N_DOMAINS, make_dataset

FIXTURE = Path(__file__).with_name("fixtures") / "spagcn_127_reference.npz"


def run_pipeline(**compat):
    """The SpaGCN tutorial pipeline used to create the fixture."""
    counts, _, _, x_pixel, y_pixel = make_dataset()
    adata = sc.AnnData(counts)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
        spg.prefilter_genes(adata, min_cells=3)
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        l = spg.search_l(
            0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100, **compat
        )
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
            **compat,
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
            res=res,
            tol=5e-3,
            lr=0.05,
            max_epochs=200,
            **compat,
        )
        labels, _ = clf.predict()
    return l, res, labels


class SpaGCN127CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = np.load(FIXTURE)

    def test_default_mode_is_spagcn_127(self):
        self.assertEqual(spg.DEFAULT_COMPAT, "spagcn-1.2.7")
        self.assertEqual(spg.COMPAT_MODES, ("spagcn-1.2.7", "modern"))

    def test_louvain_matches_louvain_package(self):
        n = int(self.reference["graph_n"])
        rows = self.reference["graph_row"]
        cols = self.reference["graph_col"]
        adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        for resolution, expected in zip(
            LOUVAIN_RESOLUTIONS, self.reference["louvain_memberships"]
        ):
            with self.subTest(resolution=resolution):
                np.testing.assert_array_equal(
                    louvain_membership(adjacency, resolution), expected
                )

    def test_compat_mode_reproduces_spagcn_127_labels(self):
        l, res, labels = run_pipeline(compat="spagcn-1.2.7")
        self.assertEqual(l, float(self.reference["length_scale"]))
        self.assertEqual(res, float(self.reference["resolution"]))
        np.testing.assert_array_equal(labels, self.reference["labels"])

    def test_modern_mode_is_a_different_model(self):
        _, _, labels = run_pipeline(compat="modern")
        self.assertFalse(np.array_equal(labels, self.reference["labels"]))

    def test_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "compat must be one of"):
            simple_GC_DEC(2, 2, compat="spagcn-1.3")


class FrozenCenterTests(unittest.TestCase):
    def fit(self, compat):
        rng = np.random.default_rng(3)
        features = rng.random((40, 4), dtype=np.float32)
        model = simple_GC_DEC(4, 4, compat=compat)
        captured = []

        def record(*args):
            captured.append(_initial_centers(*args))
            return captured[-1]

        with (
            patch("SpaGCN.models._initial_centers", side_effect=record),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            model.fit(
                features,
                np.eye(40, dtype=np.float32),
                lr=0.05,
                max_epochs=5,
                opt="admin",
                init="kmeans",
                n_clusters=3,
                tol=0,
            )
        initial = torch.as_tensor(captured[0], dtype=torch.float32)
        return model.mu.detach(), initial

    def test_spagcn_127_keeps_initial_centers(self):
        trained, initial = self.fit("spagcn-1.2.7")
        torch.testing.assert_close(trained, initial, rtol=0, atol=0)

    def test_modern_trains_centers(self):
        trained, initial = self.fit("modern")
        self.assertFalse(torch.equal(trained, initial))


if __name__ == "__main__":
    unittest.main()

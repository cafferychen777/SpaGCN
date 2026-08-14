import unittest
from unittest.mock import patch

import numpy as np
import scanpy as sc
import torch
from scipy import sparse

from SpaGCN._data import (
    gaussian_adjacency,
    mean_gaussian_neighborhood,
    pca_embedding,
)
from SpaGCN.models import _float_tensor, _make_optimizer, simple_GC_DEC
from SpaGCN.util import search_res


class DataPreparationTests(unittest.TestCase):
    def test_gaussian_adjacency_matches_reference_without_float64_output(self):
        adjacency = np.array(
            [[0.0, 1.5, 3.0], [1.5, 0.0, 2.0], [3.0, 2.0, 0.0]],
            dtype=np.float64,
        )
        length_scale = 1.7
        expected = np.exp(-(adjacency**2) / (2.0 * length_scale**2))

        actual = gaussian_adjacency(adjacency, length_scale)

        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_bounded_neighborhood_sum_matches_dense_reference(self):
        rng = np.random.default_rng(42)
        adjacency = rng.random((23, 23), dtype=np.float32)
        length_scale = 0.8
        expected = np.exp(-(adjacency**2) / (2.0 * length_scale**2)).sum(axis=1).mean()

        actual = mean_gaussian_neighborhood(
            adjacency,
            length_scale,
            max_workspace_bytes=5 * adjacency.shape[1] * np.dtype(np.float32).itemsize,
        )

        self.assertAlmostEqual(actual, expected, places=5)

    def test_sparse_pca_returns_contiguous_float32_embedding(self):
        rng = np.random.default_rng(42)
        expression = sparse.csr_matrix(rng.random((20, 8), dtype=np.float32))

        with patch.object(
            sparse.csr_matrix,
            "toarray",
            side_effect=AssertionError("sparse PCA must not densify its input"),
        ):
            embedding = pca_embedding(expression, 4, max_dense_bytes=0)

        self.assertEqual(embedding.shape, (20, 4))
        self.assertEqual(embedding.dtype, np.float32)
        self.assertTrue(embedding.flags.c_contiguous)

    def test_sparse_pca_preserves_the_centered_principal_subspace(self):
        rng = np.random.default_rng(7)
        expression = rng.random((80, 40), dtype=np.float32)

        dense_embedding = pca_embedding(expression, 5)
        sparse_embedding = pca_embedding(
            sparse.csr_matrix(expression),
            5,
            max_dense_bytes=0,
        )

        dense_gram = dense_embedding @ dense_embedding.T
        sparse_gram = sparse_embedding @ sparse_embedding.T
        np.testing.assert_allclose(sparse_gram, dense_gram, rtol=2e-3, atol=2e-3)

    def test_float32_arrays_are_shared_with_torch(self):
        values = np.arange(12, dtype=np.float32).reshape(3, 4)

        tensor = _float_tensor(values)
        values[0, 0] = 99.0

        self.assertEqual(tensor[0, 0].item(), 99.0)


class ResolutionSearchTests(unittest.TestCase):
    def test_preprocessing_is_reused_across_resolution_trials(self):
        rng = np.random.default_rng(42)
        adata = sc.AnnData(sparse.csr_matrix(rng.random((60, 60), dtype=np.float32)))
        adjacency = rng.random((60, 60), dtype=np.float32)
        first_labels = np.zeros(60, dtype=int)
        second_labels = np.arange(60, dtype=int) % 2

        with (
            patch("SpaGCN.util.pca_embedding", wraps=pca_embedding) as prepare_pca,
            patch(
                "SpaGCN.util.gaussian_adjacency",
                wraps=gaussian_adjacency,
            ) as prepare_adjacency,
            patch("SpaGCN.util.SpaGCN._train_prepared") as train_prepared,
            patch(
                "SpaGCN.util.SpaGCN.predict",
                side_effect=[
                    (first_labels, np.ones((60, 1), dtype=np.float32)),
                    (second_labels, np.ones((60, 2), dtype=np.float32)),
                ],
            ),
        ):
            result = search_res(
                adata,
                adjacency,
                l=1.0,
                target_num=2,
                max_epochs=1,
            )

        self.assertEqual(result, 0.5)
        self.assertEqual(prepare_pca.call_count, 1)
        self.assertEqual(prepare_adjacency.call_count, 1)
        self.assertEqual(train_prepared.call_count, 2)
        first_call, second_call = train_prepared.call_args_list
        self.assertIs(first_call.args[0], second_call.args[0])
        self.assertIs(first_call.args[1], second_call.args[1])


class TrainingMemoryTests(unittest.TestCase):
    def test_initialized_cluster_centers_are_registered_with_the_optimizer(self):
        rng = np.random.default_rng(5)
        features = rng.random((12, 3), dtype=np.float32)
        adjacency = np.eye(12, dtype=np.float32)
        optimized_parameters = []

        def capture_optimizer(parameters, opt, lr, weight_decay):
            parameters = list(parameters)
            optimized_parameters.extend(parameters)
            return _make_optimizer(parameters, opt, lr, weight_decay)

        model = simple_GC_DEC(3, 3)
        with patch("SpaGCN.models._make_optimizer", side_effect=capture_optimizer):
            model.fit(
                features,
                adjacency,
                max_epochs=1,
                init="kmeans",
                n_clusters=2,
            )

        self.assertTrue(
            any(parameter is model.mu for parameter in optimized_parameters)
        )

    def test_prediction_does_not_retain_an_autograd_graph(self):
        model = simple_GC_DEC(2, 2, alpha=0.2)
        model.mu = torch.nn.Parameter(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
        features = np.array([[0.2, 0.3], [0.8, 0.9]], dtype=np.float32)
        adjacency = np.eye(2, dtype=np.float32)

        embedding, assignments = model.predict(features, adjacency)

        self.assertFalse(embedding.requires_grad)
        self.assertFalse(assignments.requires_grad)


if __name__ == "__main__":
    unittest.main()

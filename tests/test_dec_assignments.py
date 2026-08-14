import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from SpaGCN.models import GC_DEC, _student_t_assignments, simple_GC_DEC


class IdentityGraphConvolution(nn.Module):
    def forward(self, features, adjacency):
        return features


class StudentTAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.alpha = 0.2
        self.embeddings = torch.tensor(
            [[0.0, 0.5], [1.0, 2.0], [3.0, 1.0]],
            dtype=torch.float64,
        )
        self.centers = torch.tensor(
            [[0.0, 0.0], [2.0, 1.0]],
            dtype=torch.float64,
        )

    def expected_assignments(self):
        embeddings = self.embeddings.numpy()
        centers = self.centers.numpy()
        squared_distances = np.sum(
            (embeddings[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        weights = np.power(
            1.0 + squared_distances / self.alpha,
            -((self.alpha + 1.0) / 2.0),
        )
        return weights / weights.sum(axis=1, keepdims=True)

    def test_matches_dec_student_t_formula(self):
        actual = _student_t_assignments(
            self.embeddings,
            self.centers,
            self.alpha,
        )

        np.testing.assert_allclose(
            actual.detach().numpy(),
            self.expected_assignments(),
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            actual.sum(dim=1),
            torch.ones(3, dtype=torch.float64),
        )

    def test_gradients_are_finite(self):
        embeddings = self.embeddings.clone().requires_grad_(True)
        assignments = _student_t_assignments(
            embeddings,
            self.centers,
            self.alpha,
        )

        assignments[:, 0].sum().backward()

        self.assertIsNotNone(embeddings.grad)
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertGreater(torch.linalg.vector_norm(embeddings.grad).item(), 0.0)

    def test_rejects_nonpositive_alpha(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            _student_t_assignments(self.embeddings, self.centers, 0.0)

    def test_both_dec_models_use_correct_assignments(self):
        adjacency = torch.eye(len(self.embeddings), dtype=torch.float64)
        expected = torch.from_numpy(self.expected_assignments())

        simple_model = simple_GC_DEC(2, 2, alpha=self.alpha).double()
        simple_model.gc = IdentityGraphConvolution()
        simple_model.mu = Parameter(self.centers.clone())
        _, simple_assignments = simple_model(self.embeddings, adjacency)

        model = GC_DEC(2, 2, 2, n_clusters=2, dropout=0.0, alpha=self.alpha).double()
        model.gc1 = IdentityGraphConvolution()
        model.gc2 = IdentityGraphConvolution()
        model.mu = Parameter(self.centers.clone())
        _, assignments = model(self.embeddings, adjacency)

        torch.testing.assert_close(simple_assignments, expected)
        torch.testing.assert_close(assignments, expected)


if __name__ == "__main__":
    unittest.main()

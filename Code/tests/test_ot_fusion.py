import unittest

import torch

from ot_fusion import FeatureTransport, sinkhorn_divergence, sinkhorn_log


class SinkhornTests(unittest.TestCase):
    def test_plan_matches_requested_marginals(self):
        torch.manual_seed(0)
        cost = torch.rand(2, 5, 7, dtype=torch.float64)
        row = torch.full((2, 5), 0.2, dtype=torch.float64)
        col = torch.full((2, 7), 1.0 / 7, dtype=torch.float64)
        plan = sinkhorn_log(cost, row, col, eps=0.1, n_iters=200)
        torch.testing.assert_close(plan.sum(2), row, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(plan.sum(1), col, atol=1e-7, rtol=1e-6)

    def test_divergence_is_symmetric_and_zero_on_identity(self):
        torch.manual_seed(1)
        x = torch.randn(2, 8, 4, dtype=torch.float64)
        y = torch.randn(2, 8, 4, dtype=torch.float64)
        dxy = sinkhorn_divergence(x, y, eps=0.1, n_iters=150)
        dyx = sinkhorn_divergence(y, x, eps=0.1, n_iters=150)
        dxx = sinkhorn_divergence(x, x, eps=0.1, n_iters=150)
        # Alternating finite-iteration scaling is only approximately symmetric.
        torch.testing.assert_close(dxy, dyx, atol=5e-4, rtol=1e-3)
        torch.testing.assert_close(dxx, torch.zeros_like(dxx), atol=1e-8, rtol=0)
        self.assertTrue(torch.all(dxy >= -1e-7))

    def test_divergence_has_finite_gradients(self):
        x = torch.randn(1, 6, 3, requires_grad=True)
        y = torch.randn(1, 7, 3)
        sinkhorn_divergence(x, y, eps=0.1, n_iters=50).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


class FeatureTransportTests(unittest.TestCase):
    def test_chunking_does_not_change_result(self):
        torch.manual_seed(2)
        vis = torch.randn(1, 3, 8, 8)
        ir = torch.randn(1, 3, 8, 8)
        common = dict(window_size=4, overlap=2, eps=0.1, n_iters=30,
                      spatial_weight=1.0, detail_kernel=0)
        full = FeatureTransport(window_chunk_size=1000, **common)
        chunked = FeatureTransport(window_chunk_size=2, **common)
        out_full, plan = full(vis, ir)
        out_chunked, _ = chunked(vis, ir)
        self.assertIsNone(plan)
        torch.testing.assert_close(out_full, out_chunked)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            FeatureTransport(window_size=8, overlap=0)
        with self.assertRaises(ValueError):
            FeatureTransport(window_size=8, overlap=3)
        with self.assertRaises(ValueError):
            FeatureTransport(detail_kernel=4)
        with self.assertRaises(ValueError):
            FeatureTransport(eps=0)

    def test_mismatched_shapes_are_rejected(self):
        model = FeatureTransport(window_size=4, overlap=2)
        with self.assertRaisesRegex(ValueError, 'identical shapes'):
            model(torch.randn(1, 3, 8, 8), torch.randn(1, 3, 9, 8))


if __name__ == '__main__':
    unittest.main()

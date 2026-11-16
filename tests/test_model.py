import importlib.util
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class ModelComponentTests(unittest.TestCase):
    def test_sinc_preserves_finite_gradient_and_expected_length(self):
        from infant_benchmark.model import SincConv
        layer = SincConv()
        x = torch.randn(2, 1, 32000)
        out = layer(x)
        self.assertEqual(tuple(out.shape), (2, 64, 4000))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(out.requires_grad)

    def test_wave_encoder_convolution_shape(self):
        from infant_benchmark.model import WaveConv
        out = WaveConv()(torch.zeros(2, 1, 32000))
        self.assertEqual(tuple(out.shape), (2, 62, 128))

    def test_sinc_gradients_are_finite_without_optimizer_step(self):
        from infant_benchmark.model import SincConv
        layer = SincConv()
        before = layer.low_hz.detach().clone()
        layer(torch.randn(1, 1, 2048)).square().mean().backward()
        self.assertTrue(torch.isfinite(layer.low_hz.grad).all())
        self.assertTrue(torch.equal(before, layer.low_hz.detach()))

    def test_kan_outputs_five_finite_logits(self):
        from infant_benchmark.model import GaussianKAN
        layer = GaussianKAN(192, 5)
        out = layer(torch.randn(3, 192))
        self.assertEqual(tuple(out.shape), (3, 5))
        self.assertTrue(torch.isfinite(out).all())

    @unittest.skipIf(importlib.util.find_spec('mamba_ssm') is not None, 'Only tests missing dependency')
    def test_main_model_fails_closed_when_mamba_missing(self):
        from infant_benchmark.model import SincMambaKAN
        with self.assertRaisesRegex(RuntimeError, 'mamba'):
            SincMambaKAN()

    @unittest.skipUnless(torch is not None and torch.cuda.is_available() and importlib.util.find_spec('mamba_ssm') is not None,
                         'Requires installed CUDA PyTorch and mamba_ssm')
    def test_complete_model_gpu_inference_on_synthetic_input(self):
        from infant_benchmark.model import SincMambaKAN
        model = SincMambaKAN().cuda().eval()
        with torch.no_grad():
            result = model(torch.zeros((1, 32000), device='cuda'), torch.zeros((1, 100), device='cuda'))
        self.assertEqual(tuple(result.shape), (1, 5))
        self.assertTrue(torch.isfinite(result).all())

    @unittest.skipUnless(torch is not None and torch.cuda.is_available() and importlib.util.find_spec('mamba_ssm') is not None,
                         'Requires installed CUDA PyTorch and mamba_ssm')
    def test_complete_model_gpu_gradient_without_optimizer(self):
        from infant_benchmark.model import SincMambaKAN
        model = SincMambaKAN().cuda().eval()
        initial = [p.detach().clone() for p in model.parameters()]
        result = model(torch.randn((1, 32000), device='cuda'), torch.randn((1, 100), device='cuda'))
        result.square().mean().backward()
        for before, param in zip(initial, model.parameters()):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertTrue(torch.equal(before, param.detach()))


if __name__ == '__main__':
    unittest.main()

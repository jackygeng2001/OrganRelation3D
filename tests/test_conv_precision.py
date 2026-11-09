"""Reduced-precision depth_sliced 3x3x3 arithmetic: accuracy, FP32 boundaries and training guards."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import support as fixtures
from test_brain_tumor import Cases, config
from organ_relation.losses import RegionJointLoss
from organ_relation.models import depth_sliced_conv as sliced
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.training.engine import Trainer, validate_options
from organ_relation.training.state import seed_all

TOLERANCE = {'bfloat16': 3e-2, 'float16': 5e-3}  # Max error relative to the reference magnitude.


def close(test, actual, expected, dtype):
    test.assertEqual(actual.dtype, torch.float32)
    scale = expected.abs().max().clamp_min(1e-12)
    test.assertLessEqual(((actual - expected).abs().max() / scale).item(), TOLERANCE[dtype])


def converted(conv, dtype):
    module = sliced.DepthSlicedConv3d.from_native(conv)
    if dtype is not None:
        module.compute_dtype = sliced.COMPUTE_DTYPES[dtype]
    return module


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(11)

    def test_output_and_all_gradients_match_fp32_conv3d(self):
        for dtype in ('bfloat16', 'float16'):
            for stride, dilation in ((1, 1), (2, 1), (1, 2)):
                with self.subTest(dtype=dtype, stride=stride, dilation=dilation):
                    conv = torch.nn.Conv3d(5, 6, 3, stride=stride, padding=dilation, dilation=dilation)
                    x = torch.randn(1, 5, 9, 10, 11, generator=self.generator).requires_grad_()
                    reference = conv(x)
                    grad = torch.randn(reference.shape, generator=self.generator)
                    expected = torch.autograd.grad(reference, (x, conv.weight, conv.bias), grad)
                    module = converted(conv, dtype)
                    y = x.detach().clone().requires_grad_()
                    actual_out = module(y)
                    close(self, actual_out, reference.detach(), dtype)
                    actual = torch.autograd.grad(actual_out, (y, module.weight, module.bias), grad)
                    for a, e in zip(actual, expected):
                        close(self, a, e, dtype)

    def test_concatenated_input_gradients_match_fp32(self):
        conv = torch.nn.Conv3d(7, 4, 3, padding=1)
        a = torch.randn(1, 3, 6, 7, 8, generator=self.generator).requires_grad_()
        b = torch.randn(1, 4, 6, 7, 8, generator=self.generator).requires_grad_()
        reference = conv(torch.cat((a, b), 1))
        grad = torch.randn(reference.shape, generator=self.generator)
        expected = torch.autograd.grad(reference, (a, b, conv.weight), grad)
        module = converted(conv, 'bfloat16')
        a2, b2 = a.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
        out = module(a2, b2)
        close(self, out, reference.detach(), 'bfloat16')
        for x, e in zip(torch.autograd.grad(out, (a2, b2, module.weight), grad), expected):
            close(self, x, e, 'bfloat16')

    def test_pointwise_projection_stays_exact_fp32(self):
        conv = torch.nn.Conv3d(6, 4, 1)
        x = torch.randn(1, 6, 3, 4, 5, generator=self.generator)
        plain, reduced = converted(conv, None), converted(conv, 'bfloat16')
        self.assertTrue(torch.equal(reduced(x), plain(x)))

    def test_default_path_calls_the_original_kernel_signature(self):
        calls = []
        original = sliced._DepthSlicedConv.apply

        def record(*args):
            calls.append(len(args))
            return original(*args)
        conv = converted(torch.nn.Conv3d(3, 4, 3, padding=1), None)
        x = torch.randn(1, 3, 4, 5, 6)
        with patch.object(sliced._DepthSlicedConv, 'apply', side_effect=record):
            conv(x)
            conv(x[:, :1], x[:, 1:])
        self.assertEqual(calls, [7, 8])

    def test_invalid_combinations_are_refused(self):
        cfg = BrainTumorConfig(**config()['model'])
        with self.assertRaisesRegex(ValueError, 'depth_sliced'):
            BrainTumorSegmentor(cfg, convolution_backend='native', convolution_compute_dtype='bfloat16')
        with self.assertRaisesRegex(ValueError, 'null, bfloat16 or float16'):
            sliced.enable_depth_sliced_convolutions(torch.nn.Sequential(torch.nn.Conv3d(1, 1, 3)), 'float64')
        module = converted(torch.nn.Conv3d(2, 2, 3, padding=1), 'bfloat16')
        with self.assertRaisesRegex(ValueError, 'FP32 volumes'):
            module(torch.randn(1, 2, 3, 3, 3, dtype=torch.float64))


class ModelTests(unittest.TestCase):
    """Whole-network rounding accumulates through ~20 backward convolutions.

    Measured on this input: bf16 logits relL2 ~0.005, all-parameter gradient
    relL2 ~0.05 (first stem layer ~0.13, head ~0.002), cosine ~0.999; fp16
    errors are 4-8x smaller, matching its 3 extra mantissa bits. An
    implementation error would not scale with the mantissa width.
    """
    def run_model(self, reference, dtype):
        model = BrainTumorSegmentor(BrainTumorConfig(**config()['model']), convolution_backend='depth_sliced',
                                    convolution_compute_dtype=dtype)
        model.load_state_dict(reference.state_dict())
        image = torch.randn(1, 4, 9, 10, 11, generator=torch.Generator().manual_seed(4))
        label = (torch.arange(9 * 10 * 11).reshape(1, 9, 10, 11) % 4).long()
        out = model(image)
        RegionJointLoss(**dict(config()['loss'], lambda_hier=1.0))(out.coarse_logits, out.final_logits, label).total.backward()
        self.assertTrue(all(p.dtype == torch.float32 and p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        return model, out.final_logits.detach(), torch.cat([p.grad.flatten() for p in model.parameters()])

    def test_full_model_keeps_fp32_parameters_and_tracks_fp32_outputs(self):
        seed_all(3)
        reference = BrainTumorSegmentor(BrainTumorConfig(**config()['model']), convolution_backend='depth_sliced')
        _, logits, grads = self.run_model(reference, None)
        errors = {}
        for dtype in ('float16', 'bfloat16'):
            model, out, g = self.run_model(reference, dtype)
            self.assertEqual(list(model.state_dict()), list(reference.state_dict()))
            errors[dtype] = ((out - logits).norm() / logits.norm()).item(), ((g - grads).norm() / grads.norm()).item()
            self.assertLessEqual(errors[dtype][0], 1e-2)
            self.assertLessEqual(errors[dtype][1], 0.1)
            self.assertGreaterEqual(F.cosine_similarity(g, grads, 0).item(), 0.995)
        self.assertLess(errors['float16'][0], errors['bfloat16'][0])
        self.assertLess(errors['float16'][1], errors['bfloat16'][1])


class TrainingGuardTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cfg = config()

    def options(self, dtype):
        return dict(self.cfg['training'], max_steps=2, validation_every=1, diagnostics_every=1,
                    checkpoint_every=1, validation_role='train_monitor', convolution_compute_dtype=dtype)

    def test_engine_accepts_bfloat16_only(self):
        validate_options(self.options('bfloat16'))
        for bad in ('float16', 'float32', 'half'):
            with self.assertRaisesRegex(ValueError, 'null or bfloat16'):
                validate_options(self.options(bad))

    def test_two_bfloat16_steps_train_and_record_identity(self):
        seed_all(171)
        model = BrainTumorSegmentor(BrainTumorConfig(**self.cfg['model']), convolution_backend='depth_sliced',
                                    convolution_compute_dtype='bfloat16')
        options = self.options('bfloat16')
        identity = dict(mode='brats_regions', model=self.cfg['model'], loss=self.cfg['loss'], threshold=.5,
                        training=options, preprocessing=self.cfg['preprocessing'], provenance='test',
                        data={'manifest_hash': 'synthetic'})
        before = copy.deepcopy(model.state_dict())
        trainer = Trainer(model, RegionJointLoss(**self.cfg['loss']),
                          torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False, fused=False),
                          Cases(), ['one', 'two'], device='cpu', options=options, identity=identity,
                          run_dir=self.fixture.root / 'run', validation_dataset=Cases(),
                          validation_case_ids=['one', 'two'], tensorboard=False)
        self.fixture.quiet_run(trainer)
        rows = [r for r in self.fixture.rows(trainer.run_dir) if r['phase'] == 'train']
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(r['total_loss'])) for r in rows))
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertEqual(trainer.identity['training']['convolution_compute_dtype'], 'bfloat16')


if __name__ == '__main__':
    unittest.main()

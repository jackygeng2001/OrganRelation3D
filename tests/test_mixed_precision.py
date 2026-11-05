"""Whole-network bf16 mixed precision: kernels, GroupNorm, model boundaries and training guards."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import test_training as fixtures
from test_brain_tumor import Cases, config
from organ_relation.losses import RegionJointLoss
from organ_relation.models import depth_sliced_conv as sliced
from organ_relation.models import group_norm
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.training.engine import Trainer, validate_options
from organ_relation.training.state import seed_all


def bf16_volume(*shape, generator, offset=0.0):
    """bf16 channels-last input and its exact FP32 copy (same values)."""
    x = (torch.randn(*shape, generator=generator) + offset).to(torch.bfloat16)
    x = x.contiguous(memory_format=torch.channels_last_3d)
    return x, x.float()


def relative(actual, expected):
    return ((actual.double() - expected.double()).abs().max() / expected.double().abs().max().clamp_min(1e-12)).item()


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(21)

    def test_bf16_volume_convolution_tracks_fp32_reference(self):
        for stride, dilation in ((1, 1), (2, 1), (1, 2)):
            with self.subTest(stride=stride, dilation=dilation):
                conv = torch.nn.Conv3d(5, 6, 3, stride=stride, padding=dilation, dilation=dilation)
                x, x32 = bf16_volume(1, 5, 9, 10, 11, generator=self.generator)
                x32.requires_grad_()
                reference = conv(x32)
                grad = torch.randn(reference.shape, generator=self.generator).to(torch.bfloat16)
                expected = torch.autograd.grad(reference, (x32, conv.weight, conv.bias), grad.float())
                module = sliced.DepthSlicedConv3d.from_native(conv)
                x.requires_grad_()
                out = module(x)
                self.assertEqual(out.dtype, torch.bfloat16)
                self.assertLessEqual(relative(out, reference.detach()), 1e-2)
                dx, dw, db = torch.autograd.grad(out, (x, module.weight, module.bias), grad)
                self.assertEqual((dx.dtype, dw.dtype, db.dtype), (torch.bfloat16, torch.float32, torch.float32))
                for actual, reference_grad in zip((dx, dw, db), expected):
                    self.assertLessEqual(relative(actual, reference_grad), 2e-2)

    def test_concatenated_bf16_inputs_receive_separate_gradients(self):
        conv = torch.nn.Conv3d(7, 4, 3, padding=1)
        a, a32 = bf16_volume(1, 3, 6, 7, 8, generator=self.generator)
        b, b32 = bf16_volume(1, 4, 6, 7, 8, generator=self.generator)
        a32.requires_grad_(); b32.requires_grad_()
        reference = conv(torch.cat((a32, b32), 1))
        grad = torch.randn(reference.shape, generator=self.generator).to(torch.bfloat16)
        expected = torch.autograd.grad(reference, (a32, b32, conv.weight), grad.float())
        module = sliced.DepthSlicedConv3d.from_native(conv)
        a.requires_grad_(); b.requires_grad_()
        out = module(a, b)
        self.assertLessEqual(relative(out, reference.detach()), 1e-2)
        for actual, reference_grad in zip(torch.autograd.grad(out, (a, b, module.weight), grad), expected):
            self.assertLessEqual(relative(actual, reference_grad), 2e-2)

    def test_pointwise_projection_rounds_weights_and_returns_fp32_gradients(self):
        conv = torch.nn.Conv3d(6, 4, 1)
        module = sliced.DepthSlicedConv3d.from_native(conv)
        x, x32 = bf16_volume(1, 6, 3, 4, 5, generator=self.generator)
        out = module(x)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertLessEqual(relative(out, conv(x32).detach()), 1e-2)
        out.float().sum().backward()
        self.assertEqual(module.weight.grad.dtype, torch.float32)

    def test_kernel_only_and_whole_network_modes_are_exclusive(self):
        module = sliced.DepthSlicedConv3d.from_native(torch.nn.Conv3d(2, 2, 3, padding=1))
        module.compute_dtype = torch.bfloat16
        with self.assertRaisesRegex(ValueError, 'either'):
            module(torch.randn(1, 2, 3, 3, 3).to(torch.bfloat16))


class GroupNormTests(unittest.TestCase):
    def run_pair(self, offset, chunk_elements=None):
        generator = torch.Generator().manual_seed(5)
        reference = torch.nn.GroupNorm(4, 8)
        with torch.no_grad():
            reference.weight.uniform_(0.5, 1.5, generator=generator)
            reference.bias.uniform_(-0.5, 0.5, generator=generator)
        module = group_norm.ChannelsLastGroupNorm(4, 8)
        module.load_state_dict(reference.state_dict())
        reference.double()  # Native FP32 GroupNorm itself errs by ~1e-2 at this mean.
        x, x32 = bf16_volume(1, 8, 6, 7, 9, generator=generator, offset=offset)
        x64 = x32.double().requires_grad_(); x.requires_grad_()
        grad = torch.randn(x.shape, generator=generator).to(torch.bfloat16)
        expected_out = reference(x64)
        expected = torch.autograd.grad(expected_out, (x64, reference.weight, reference.bias), grad.double())
        context = (patch.object(group_norm, '_BACKWARD_CHUNK_ELEMENTS', chunk_elements)
                   if chunk_elements else patch.object(group_norm, '_BACKWARD_CHUNK_ELEMENTS',
                                                       group_norm._BACKWARD_CHUNK_ELEMENTS))
        with context:
            out = module(x)
            actual = torch.autograd.grad(out, (x, module.weight, module.bias), grad)
        return out, expected_out.detach(), actual, expected

    def test_statistics_are_fp32_even_with_a_large_mean(self):
        # bf16 spacing at 64 is 0.5; a bf16 mean would shift every output by ~0.25/std.
        # FP32 statistics leave only the final rounding: at most one bf16 half-ulp.
        for chunk_elements in (None, 40):  # 40 forces dozens of merged FP32 chunks.
            with self.subTest(chunk_elements=chunk_elements):
                out, expected_out, actual, expected = self.run_pair(64.0, chunk_elements)
                self.assertEqual(out.dtype, torch.bfloat16)
                error = (out.double() - expected_out).abs()
                self.assertTrue(bool((error <= expected_out.abs() * 2**-8 + 1e-6).all()))  # 8 significant bits
                self.assertEqual([g.dtype for g in actual], [torch.bfloat16, torch.float32, torch.float32])
                for a, e in zip(actual, expected):
                    self.assertLessEqual(relative(a, e), 2e-2)

    def test_non_channels_last_bf16_uses_fp32_reference(self):
        module = group_norm.ChannelsLastGroupNorm(2, 4)
        x = torch.randn(1, 4, 3, 4, 5).to(torch.bfloat16)
        out = module(x)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out, torch.nn.functional.group_norm(
            x.float(), 2, module.weight, module.bias, module.eps).to(torch.bfloat16)))


class ModelTests(unittest.TestCase):
    """Measured on this synthetic input (CPU): final logits relL2 0.0067, coarse
    0.0063, all-parameter gradient relL2 0.053, cosine 0.9990 -- within ~5% of
    the kernel-only bf16 option (0.0054/0.0071/0.051/0.9990). An implementation
    error would not keep the gradient direction."""

    def build(self, **kwargs):
        return BrainTumorSegmentor(BrainTumorConfig(**config()['model']), convolution_backend='depth_sliced', **kwargs)

    def run_model(self, model):
        image = torch.randn(1, 4, 9, 10, 11, generator=torch.Generator().manual_seed(4))
        image = image.contiguous(memory_format=torch.channels_last_3d)
        label = (torch.arange(9 * 10 * 11).reshape(1, 9, 10, 11) % 4).long()
        out = model(image)
        loss = RegionJointLoss(**dict(config()['loss'], lambda_hier=1.0))(out.coarse_logits, out.final_logits, label)
        loss.total.backward()
        return out, torch.cat([p.grad.flatten() for p in model.parameters()])

    def test_mixed_model_keeps_fp32_boundaries_and_tracks_fp32(self):
        seed_all(3)
        reference = self.build()
        reference.to(memory_format=torch.channels_last_3d)
        expected, expected_grads = self.run_model(reference)
        model = self.build(mixed_precision='bfloat16')
        model.load_state_dict(reference.state_dict())
        model.to(memory_format=torch.channels_last_3d)
        out, grads = self.run_model(model)
        self.assertEqual(list(model.state_dict()), list(reference.state_dict()))
        self.assertTrue(all(p.dtype == torch.float32 and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual((out.coarse_logits.dtype, out.final_logits.dtype), (torch.float32, torch.float32))
        for actual, wanted in ((out.final_logits, expected.final_logits), (out.coarse_logits, expected.coarse_logits)):
            self.assertLessEqual(((actual - wanted).norm() / wanted.norm()).item(), 2e-2)
        self.assertLessEqual(((grads - expected_grads).norm() / expected_grads.norm()).item(), 0.1)
        self.assertGreaterEqual(F.cosine_similarity(grads, expected_grads, 0).item(), 0.995)

    def test_volumes_are_bf16_inside_and_fp32_at_the_relation_core(self):
        seen = {}
        model = self.build(mixed_precision='bfloat16')
        model.encoder.register_forward_hook(lambda m, i, o: seen.update(encoder=[f.dtype for f in o]))
        for name in ('coarse_head', 'space_to_node', 'node_to_space', 'fusion'):
            getattr(model, name).register_forward_hook(
                lambda m, i, o, name=name: seen.update({name: [t.dtype for t in i]}))
        model.decoder.head.register_forward_hook(lambda m, i, o: seen.update(head=o.dtype))
        with torch.no_grad():
            out = model(torch.randn(1, 4, 9, 10, 11))
        self.assertEqual(seen['encoder'], [torch.bfloat16] * 4)
        for name in ('coarse_head', 'space_to_node', 'node_to_space', 'fusion'):
            self.assertEqual(set(seen[name]), {torch.float32}, name)
        self.assertEqual((seen['head'], out.final_logits.dtype), (torch.bfloat16, torch.float32))

    def test_invalid_combinations_are_refused(self):
        cfg = BrainTumorConfig(**config()['model'])
        for kwargs in (dict(convolution_backend='native', mixed_precision='bfloat16'),
                       dict(convolution_backend='depth_sliced', mixed_precision='float16'),
                       dict(convolution_backend='depth_sliced', mixed_precision='bfloat16',
                            convolution_compute_dtype='bfloat16')):
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, 'mixed_precision'):
                BrainTumorSegmentor(cfg, **kwargs)
        with self.assertRaisesRegex(ValueError, 'FP32 image'):
            self.build(mixed_precision='bfloat16')(torch.randn(1, 4, 3, 3, 3, dtype=torch.float64))


class TrainingGuardTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cfg = config()

    def options(self, **extra):
        return dict(self.cfg['training'], max_steps=2, validation_every=1, diagnostics_every=1,
                    checkpoint_every=1, validation_role='train_monitor', **extra)

    def test_engine_accepts_bfloat16_only_and_one_mode(self):
        validate_options(self.options(mixed_precision='bfloat16'))
        for bad in ('float16', 'float32'):
            with self.assertRaisesRegex(ValueError, 'mixed_precision must be'):
                validate_options(self.options(mixed_precision=bad))
        with self.assertRaisesRegex(ValueError, 'either'):
            validate_options(self.options(mixed_precision='bfloat16', convolution_compute_dtype='bfloat16'))

    def test_two_mixed_precision_steps_train_and_record_identity(self):
        seed_all(171)
        model = BrainTumorSegmentor(BrainTumorConfig(**self.cfg['model']), convolution_backend='depth_sliced',
                                    mixed_precision='bfloat16')
        options = self.options(mixed_precision='bfloat16')
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
        self.assertTrue(all(v.dtype == before[k].dtype for k, v in model.state_dict().items()))
        self.assertEqual(trainer.identity['training']['mixed_precision'], 'bfloat16')


if __name__ == '__main__':
    unittest.main()

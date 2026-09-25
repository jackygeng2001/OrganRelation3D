"""CPU synthetic formula/gradient checks; no CT data or training objective."""
import inspect
import itertools
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
ROOT = Path(__file__).resolve().parents[1]
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True
    from organ_relation.backbone import Encoder3D
    from organ_relation.backbone_config import BackboneConfig
    from organ_relation.coarse_head import CoarseHead, CoarsePrediction
    from organ_relation.space_to_node import (
        CENTROID_AXES, ORGAN_LABEL_IDS, OrganNodes, SpaceToNode,
    )


def explicit_reference(features, probabilities, epsilon):
    """Independent scalar loops over voxels; preserves reference autograd."""
    shape = features.shape[2:]
    batches = []
    for b in range(features.shape[0]):
        nodes = []
        for label_id in range(1, 16):
            mass = features.new_zeros(())
            semantic = features.new_zeros(features.shape[1])
            centroid = features.new_zeros(3)
            confidence = features.new_zeros(())
            for d, h, w in itertools.product(*(range(n) for n in shape)):
                p = probabilities[b, label_id, d, h, w]
                mass = mass + p
                semantic = semantic + p * features[b, :, d, h, w]
                coordinate = features.new_tensor([
                    0 if shape[0] == 1 else d / (shape[0] - 1),
                    0 if shape[1] == 1 else h / (shape[1] - 1),
                    0 if shape[2] == 1 else w / (shape[2] - 1),
                ])
                centroid = centroid + p * coordinate
                confidence = confidence + p * p
            nodes.append((semantic / (mass + epsilon), mass.reshape(1),
                          centroid / (mass + epsilon),
                          (mass / (shape[0] * shape[1] * shape[2])).reshape(1),
                          (confidence / (mass + epsilon)).reshape(1)))
        batches.append(tuple(torch.stack([node[j] for node in nodes]) for j in range(5)))
    return tuple(torch.stack([batch[j] for batch in batches]) for j in range(5))


@unittest.skipUnless(TORCH_AVAILABLE, 'isolated PyTorch CPU environment required')
class CoarseNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(20260925)

    def inputs(self, shape=(2, 3, 2, 3, 4), dtype=None):
        dtype = torch.float64 if dtype is None else dtype
        f = torch.randn(shape, dtype=dtype)
        logits = torch.randn((shape[0], 16, *shape[2:]), dtype=dtype)
        return f, logits.softmax(dim=1)

    def assertOutputsClose(self, actual, expected, **kwargs):
        self.assertEqual(len(actual), 5)
        for name, a, e in zip(OrganNodes._fields, actual, expected):
            with self.subTest(field=name):
                torch.testing.assert_close(a, e, **kwargs)

    def test_coarse_convolution_and_sixteen_class_softmax(self):
        f, _ = self.inputs()
        head = CoarseHead(3, bias=True).double()
        out = head(f)
        expected = torch.einsum('bcxyz,kc->bkxyz', f, head.projection.weight[:, :, 0, 0, 0])
        expected = expected + head.projection.bias[None, :, None, None, None]
        self.assertIsInstance(out, CoarsePrediction)
        torch.testing.assert_close(out.logits, expected)
        torch.testing.assert_close(out.probabilities, expected.softmax(dim=1))
        torch.testing.assert_close(out.probabilities.sum(dim=1), torch.ones_like(f[:, 0]))
        self.assertEqual(head.projection.kernel_size, (1, 1, 1))
        self.assertEqual(head.projection.padding, (0, 0, 0))
        self.assertEqual(tuple(out.logits.shape), (2, 16, 2, 3, 4))

    def test_coarse_without_bias_and_all_parameters_registered(self):
        for bias in (False, True):
            head = CoarseHead(3, bias=bias).double()
            self.assertEqual(sum(p.numel() for p in head.parameters()), 16 * 3 + 16 * bias)
            f, _ = self.inputs()
            head(f).logits.square().mean().backward()
            for name, parameter in head.named_parameters():
                with self.subTest(bias=bias, parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_hand_calculation_weighted_once(self):
        f = torch.tensor([2., 8.], dtype=torch.float64).reshape(1, 1, 1, 1, 2)
        p = torch.zeros(1, 16, 1, 1, 2, dtype=f.dtype)
        p[:, 0] = torch.tensor([.8, .4], dtype=f.dtype)
        p[:, 1] = torch.tensor([.2, .6], dtype=f.dtype)
        out = SpaceToNode(epsilon=.1)(f, p)
        expected = ([5.2 / .9], [.8], [0., 0., .6 / .9], [.4], [.4 / .9])
        for a, e in zip(out, expected):
            torch.testing.assert_close(a[0, 0], f.new_tensor(e), rtol=1e-14, atol=1e-14)
            self.assertEqual(torch.count_nonzero(a[:, 1:]).item(), 0)
        self.assertNotAlmostEqual(out.z0[0, 0, 0].item(), (.2**2 * 2 + .6**2 * 8) / .9)

    def test_all_fifteen_node_class_indices(self):
        self.assertEqual(ORGAN_LABEL_IDS, tuple(range(1, 16)))
        self.assertEqual(CENTROID_AXES, ('D', 'H', 'W'))
        f = torch.tensor([3., 7.], dtype=torch.float64).reshape(1, 2, 1, 1, 1)
        p = (torch.arange(1, 17, dtype=f.dtype) / 136).reshape(1, 16, 1, 1, 1)
        out = SpaceToNode(epsilon=.01)(f, p)
        self.assertEqual(out.z0.shape, (1, 15, 2))
        for node, label in enumerate(ORGAN_LABEL_IDS):
            mass = (label + 1) / 136
            torch.testing.assert_close(out.mass[0, node], f.new_tensor([mass]))
            torch.testing.assert_close(out.z0[0, node], f.flatten() * mass / (mass + .01))
            torch.testing.assert_close(out.size[0, node], f.new_tensor([mass]))
            torch.testing.assert_close(out.confidence[0, node], f.new_tensor([mass**2 / (mass + .01)]))

    def test_background_not_pooled(self):
        f, p = self.inputs()
        changed = p.clone()
        changed[:, 0] = 0  # Deliberately hold all foreground weights fixed.
        node = SpaceToNode(epsilon=.1)
        self.assertOutputsClose(node(f, p), node(f, changed), rtol=0, atol=0)

    def test_background_logit_participates_in_softmax(self):
        f, _ = self.inputs()
        head = CoarseHead(3, bias=True).double()
        before = head(f).probabilities
        with torch.no_grad():
            head.projection.bias[0].add_(2)
        after = head(f).probabilities
        self.assertTrue((after[:, 1:] < before[:, 1:]).all())
        node = SpaceToNode(epsilon=.1)
        self.assertTrue((node(f, after).mass < node(f, before).mass).all())

    def test_fp64_scalar_reference_all_outputs(self):
        f, p = self.inputs()
        self.assertOutputsClose(SpaceToNode(epsilon=.17)(f, p),
                                explicit_reference(f, p, .17), rtol=2e-13, atol=2e-14)

    def test_fp32_against_fp64_reference(self):
        f, p = self.inputs(dtype=torch.float32)
        out = SpaceToNode(epsilon=1e-6)(f, p)
        ref = explicit_reference(f.double(), p.double(), 1e-6)
        self.assertOutputsClose(tuple(x.double() for x in out), ref, rtol=3e-6, atol=2e-7)
        self.assertTrue(all(x.dtype == f.dtype and x.device == f.device for x in out))

    def test_zero_mass_no_mask_and_nonzero_probability_gradient(self):
        f = torch.ones(2, 3, 2, 2, 2, dtype=torch.float64, requires_grad=True)
        p = torch.zeros(2, 16, 2, 2, 2, dtype=f.dtype)
        p[:, 0] = 1
        p.requires_grad_()
        out = SpaceToNode(epsilon=1e-4)(f, p)
        for value in out:
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(torch.count_nonzero(value).item(), 0)
        out.z0.sum().backward()
        torch.testing.assert_close(p.grad[:, 1:], torch.full_like(p.grad[:, 1:], 3 / 1e-4))
        self.assertEqual(torch.count_nonzero(p.grad[:, 0]).item(), 0)
        self.assertEqual(torch.count_nonzero(f.grad).item(), 0)

    def test_tiny_mass_is_retained_and_uses_additive_epsilon(self):
        f = torch.full((2, 2, 1, 2, 2), 5., dtype=torch.float64)
        p = torch.full((2, 16, 1, 2, 2), 1e-12, dtype=f.dtype)
        p[:, 0] = 1 - 15e-12
        out = SpaceToNode(epsilon=1e-6)(f, p)
        self.assertOutputsClose(out, explicit_reference(f, p, 1e-6), rtol=2e-14, atol=0)
        for value in (out.z0, out.mass, out.size, out.confidence):
            self.assertTrue((value > 0).all())
        torch.testing.assert_close(out.z0, torch.full_like(out.z0, 20e-12 / (4e-12 + 1e-6)))

    def test_each_singleton_axis_and_single_voxel(self):
        for shape in ((1, 2, 3), (2, 1, 3), (2, 3, 1), (1, 1, 1)):
            with self.subTest(shape=shape):
                f, p = self.inputs(shape=(2, 2, *shape))
                out = SpaceToNode(epsilon=.1)(f, p)
                self.assertOutputsClose(out, explicit_reference(f, p, .1))
                for axis, n in enumerate(shape):
                    if n == 1:
                        self.assertEqual(torch.count_nonzero(out.centroid[..., axis]).item(), 0)

    def test_tiny_mass_gradients_match_scalar_reference(self):
        f = torch.full((2, 2, 1, 2, 2), 5., dtype=torch.float64, requires_grad=True)
        p = torch.full((2, 16, 1, 2, 2), 1e-12, dtype=f.dtype, requires_grad=True)
        out = SpaceToNode(epsilon=1e-6)(f, p)
        reference = explicit_reference(f, p, 1e-6)
        for value, expected in zip(out, reference):
            actual_grad = torch.autograd.grad(value.sum(), (f, p), retain_graph=True, allow_unused=True)
            ref_grad = torch.autograd.grad(expected.sum(), (f, p), retain_graph=True, allow_unused=True)
            for a, e in zip(actual_grad, ref_grad):
                if e is None:
                    self.assertIsNone(a)
                else:
                    self.assertTrue(torch.isfinite(a).all())
                    torch.testing.assert_close(a, e, rtol=2e-13, atol=1e-15)

    def test_batch_independence(self):
        f, p = self.inputs(shape=(3, 2, 2, 2, 3))
        node = SpaceToNode(epsilon=.1)
        out = node(f, p)
        for b in range(3):
            self.assertOutputsClose(tuple(x[b:b+1] for x in out), node(f[b:b+1], p[b:b+1]))

    def test_coordinate_axes_endpoints_and_epsilon(self):
        f = torch.ones(1, 1, 2, 3, 4, dtype=torch.float64)
        p = torch.zeros(1, 16, 2, 3, 4, dtype=f.dtype)
        p[:, 0] = 1
        for label, position in enumerate(((1, 0, 0), (0, 2, 0), (0, 0, 3)), start=1):
            p[(0, label, *position)] = 1
            p[(0, 0, *position)] = 0
        out = SpaceToNode(epsilon=.25)(f, p)
        torch.testing.assert_close(out.centroid[0, :3], torch.eye(3, dtype=f.dtype) / 1.25)

    def test_noncontiguous_and_spatial_permutation(self):
        f, p = self.inputs()
        node = SpaceToNode(epsilon=.1)
        before = node(f, p)
        fp = f.permute(0, 1, 4, 2, 3)
        pp = p.permute(0, 1, 4, 2, 3)
        self.assertFalse(fp.is_contiguous())
        after = node(fp, pp)
        self.assertOutputsClose(after, explicit_reference(fp, pp, .1))
        torch.testing.assert_close(after.centroid, before.centroid[..., [2, 0, 1]])
        for name in ('z0', 'mass', 'size', 'confidence'):
            torch.testing.assert_close(getattr(after, name), getattr(before, name))

    def test_gradcheck_features_and_probabilities_fp64(self):
        f, p = self.inputs(shape=(2, 2, 1, 2, 2))
        self.assertTrue(torch.autograd.gradcheck(SpaceToNode(epsilon=.1),
                        (f.requires_grad_(), p.requires_grad_())))

    def test_gradcheck_features_and_logits_fp64(self):
        f, _ = self.inputs(shape=(2, 2, 2, 1, 2))
        logits = torch.randn(2, 16, 2, 1, 2, dtype=f.dtype, requires_grad=True)
        node = SpaceToNode(epsilon=.1)
        self.assertTrue(torch.autograd.gradcheck(lambda x, l: node(x, l.softmax(dim=1)),
                        (f.requires_grad_(), logits)))

    def test_reference_gradient_for_every_output(self):
        f, p = self.inputs(shape=(2, 2, 2, 2, 2))
        f.requires_grad_(); p.requires_grad_()
        out = SpaceToNode(epsilon=.13)(f, p)
        ref = explicit_reference(f, p, .13)
        for name, value, expected in zip(OrganNodes._fields, out, ref):
            with self.subTest(field=name):
                weights = torch.randn_like(value)
                actual_grad = torch.autograd.grad((value * weights).sum(), (f, p),
                                                  retain_graph=True, allow_unused=True)
                ref_grad = torch.autograd.grad((expected * weights).sum(), (f, p),
                                               retain_graph=True, allow_unused=True)
                for a, e in zip(actual_grad, ref_grad):
                    if e is None:
                        self.assertIsNone(a)
                    else:
                        torch.testing.assert_close(a, e, rtol=3e-12, atol=3e-13)

    def test_each_fixed_attribute_keeps_logit_gradient_including_background(self):
        f, p = self.inputs()
        logits = p.log().requires_grad_()
        probability = logits.softmax(dim=1)
        out = SpaceToNode(epsilon=.1)(f, probability)
        for name in ('mass', 'centroid', 'size', 'confidence'):
            with self.subTest(field=name):
                value = getattr(out, name)
                gradient, = torch.autograd.grad((value * torch.randn_like(value)).sum(), logits,
                                                retain_graph=True)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient[:, 0].abs().sum().item(), 0)
                self.assertTrue((gradient[:, 1:].flatten(2).abs().sum(-1) > 0).all())

    def test_encoder_coarse_nodes_gradient_and_parameter_coverage(self):
        raw = json.loads((ROOT / 'configs/backbone_micro.json').read_text(encoding='utf-8'))['model']
        encoder = Encoder3D(BackboneConfig(**raw))
        head = CoarseHead(raw['channels'][-1], bias=True)
        image = torch.randn(2, 1, 9, 10, 11, requires_grad=True)
        encoded = encoder(image)
        f = encoded.deepest
        prediction = head(f)
        for t in (f, prediction.logits, prediction.probabilities):
            t.retain_grad()
        nodes = SpaceToNode(epsilon=1e-6)(f, prediction.probabilities)
        self.assertEqual(nodes.z0.shape, (2, 15, raw['channels'][-1]))
        # Synthetic differentiation probe, NOT the method's training loss.
        probe = sum((value * torch.randn_like(value)).sum() for value in nodes)
        probe.backward()
        for tensor in (image, f, prediction.logits, prediction.probabilities):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        for prefix, module in (('encoder', encoder), ('coarse', head)):
            for name, parameter in module.named_parameters():
                with self.subTest(parameter=f'{prefix}.{name}'):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_no_parameters_no_mutation_and_no_label_interface(self):
        f, p = self.inputs()
        old_f, old_p = f.clone(), p.clone()
        node = SpaceToNode(epsilon=.1)
        self.assertEqual(list(node.parameters()), [])
        self.assertEqual(list(node.buffers()), [])
        self.assertEqual(tuple(inspect.signature(node.forward).parameters), ('features', 'probabilities'))
        head = CoarseHead(3, bias=True).double()
        self.assertEqual(tuple(inspect.signature(head.forward).parameters), ('features',))
        node(f, p); head(f)
        torch.testing.assert_close(f, old_f, rtol=0, atol=0)
        torch.testing.assert_close(p, old_p, rtol=0, atol=0)
        label = torch.zeros_like(p[:, 0], dtype=torch.long)
        for invoke in (lambda: head(f, label), lambda: head(f, label=label),
                       lambda: node(f, p, label), lambda: node(f, p, label=label)):
            with self.assertRaises(TypeError):
                invoke()

    def test_invalid_configuration(self):
        for epsilon in (0, -1, float('nan'), float('inf'), True, '1e-6'):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                SpaceToNode(epsilon=epsilon)
        with self.assertRaises(TypeError):
            SpaceToNode()
        with self.assertRaises(TypeError):
            CoarseHead(3)
        for channels, bias in ((0, True), (True, True), (3.5, True), (3, 1)):
            with self.assertRaises(ValueError):
                CoarseHead(channels, bias=bias)

    def test_invalid_shapes_dtype_device_and_unrepresentable_epsilon(self):
        f, p = self.inputs()
        node = SpaceToNode(epsilon=.1)
        for bad_f, bad_p in ((f[0], p), (f[:, :0], p), (f, p[0]), (f, p[:, :15]),
                             (f, p[:1]), (f, p[..., :1]), (f.float(), p),
                             (f.long(), p.long()), (f.half(), p.half()),
                             (f.bfloat16(), p.bfloat16()),
                             (f, torch.empty(p.shape, dtype=p.dtype, device='meta'))):
            with self.assertRaises(ValueError):
                node(bad_f, bad_p)
        for epsilon in (1e-100, 1e100):
            with self.assertRaises(ValueError):
                SpaceToNode(epsilon=epsilon)(f.float(), p.float())
        head = CoarseHead(3, bias=False).double()
        for bad in (f[0], f[:, :2], f[:, :, :0], f.long()):
            with self.assertRaises(ValueError):
                head(bad)

    def test_autocast_rejected_even_for_float32_inputs(self):
        f, p = self.inputs(dtype=torch.float32)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, 'autocast disabled'):
                SpaceToNode(epsilon=1e-6)(f, p)


if __name__ == '__main__':
    unittest.main()

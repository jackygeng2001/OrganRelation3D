"""CPU synthetic residual fusion and complete forward graph tests; no training."""
from dataclasses import replace
import gc
import inspect
import itertools
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from organ_relation.models.segmentor_config import SegmentorConfig
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True
    from organ_relation.models.residual_fusion import ResidualFusion
    from organ_relation.models.segmentor import Segmentor, SegmentorOutput, SegmentorDiagnostics


def micro():
    return SegmentorConfig(**json.loads((ROOT / 'configs/segmentor_micro.json').read_text(encoding='utf-8'))['model'])


class SegmentorConfigTests(unittest.TestCase):
    def test_explicit_configuration_roundtrip(self):
        c = micro()
        self.assertEqual(SegmentorConfig(**json.loads(json.dumps(c.to_dict()))), c)
        raw = json.loads((ROOT / 'configs/segmentor_micro.json').read_text(encoding='utf-8'))
        self.assertEqual(raw['purpose'], 'cpu_synthetic_test_not_formal_experiment')
        self.assertIs(type(c.fusion_bias), bool)
        self.assertEqual(c.backbone.spatial_pyramid((17, 18, 19))[-1], (5, 5, 5))

    def test_invalid_configuration_and_required_bias(self):
        c = micro()
        for name in ('coarse_bias', 'fusion_bias'):
            with self.assertRaises(ValueError):
                replace(c, **{name: 1})
        for name in ('rounds', 'relation_channels', 'attention_channels', 'content_channels'):
            for value in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    replace(c, **{name: value})
        for name in ('epsilon', 'beta_init'):
            for value in (float('nan'), float('inf'), True, '0'):
                with self.assertRaises(ValueError):
                    replace(c, **{name: value})
        with self.assertRaises(ValueError):
            replace(c, epsilon=0)
        with self.assertRaises(ValueError):
            replace(c, backbone=None)
        fields = c.to_dict()
        del fields['fusion_bias']
        with self.assertRaises(TypeError):
            SegmentorConfig(**fields)


@unittest.skipUnless(TORCH_AVAILABLE, 'isolated PyTorch CPU environment required')
class SegmentorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(20260925)

    def controlled(self):
        """Positive small weights and no norm give a deliberately active graph path.

        This is only a differentiability fixture, never a production setting.
        """
        config = micro()
        backbone = replace(config.backbone, channels=(2, 4), downsample_strides=((2, 2, 2),),
                           normalization='none', conv_bias=True)
        config = replace(config, backbone=backbone, relation_channels=3, attention_channels=2,
                         content_channels=3, fusion_bias=False)
        model = Segmentor(config).double()
        with torch.no_grad():
            for p in model.parameters():
                p.uniform_(.02, .08)
            model.relation.relation_hidden.bias.fill_(.5)
        return model

    def test_fusion_hand_calculation_shapes_and_only_one_convolution(self):
        fusion = ResidualFusion(2, content_channels=1, bias=True).double()
        with torch.no_grad():
            fusion.phi.weight.copy_(torch.tensor([2., -1.], dtype=torch.float64).reshape(2, 1, 1, 1, 1))
            fusion.phi.bias.copy_(torch.tensor([.5, -2.], dtype=torch.float64))
        f = torch.tensor([1., 2., 3., 4.], dtype=torch.float64).reshape(1, 2, 1, 1, 2)
        g = torch.tensor([5., 6.], dtype=torch.float64).reshape(1, 1, 1, 1, 2)
        before_f, before_g = f.clone(), g.clone()
        projected = fusion.phi(g)
        self.assertEqual(projected.shape, f.shape)
        torch.testing.assert_close(projected.flatten(), f.new_tensor([10.5, 12.5, -7., -8.]), rtol=0, atol=0)
        torch.testing.assert_close(fusion(f, g).flatten(), f.new_tensor([11.5, 14.5, -4., -4.]), rtol=0, atol=0)
        self.assertEqual(list(dict(fusion.named_children())), ['phi'])
        self.assertEqual(set(dict(fusion.named_parameters())), {'phi.weight', 'phi.bias'})
        self.assertEqual(fusion.phi.kernel_size, (1, 1, 1))
        self.assertEqual(fusion.phi.stride, (1, 1, 1))
        self.assertEqual(fusion.phi.padding, (0, 0, 0))
        torch.testing.assert_close(f, before_f, rtol=0, atol=0)
        torch.testing.assert_close(g, before_g, rtol=0, atol=0)

    def test_fusion_F_G_phi_analytical_gradients(self):
        fusion = ResidualFusion(2, content_channels=3, bias=True).double()
        f = torch.randn(2, 2, 2, 1, 3, dtype=torch.float64, requires_grad=True)
        g = torch.randn(2, 3, 2, 1, 3, dtype=torch.float64, requires_grad=True)
        weight = torch.randn_like(f)
        gradients = torch.autograd.grad((fusion(f, g) * weight).sum(), (f, g, fusion.phi.weight, fusion.phi.bias))
        expected_g = torch.einsum('oc,bodhw->bcdhw', fusion.phi.weight[:, :, 0, 0, 0], weight)
        expected_weight = torch.einsum('bodhw,bcdhw->oc', weight, g).reshape_as(fusion.phi.weight)
        expected_bias = weight.sum(dim=(0, 2, 3, 4))
        for actual, expected in zip(gradients, (weight, expected_g, expected_weight, expected_bias)):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)

    def test_fusion_fp64_gradcheck_inputs_and_all_parameters(self):
        fusion = ResidualFusion(2, content_channels=1, bias=True).double()
        f = torch.randn(1, 2, 1, 1, 2, dtype=torch.float64, requires_grad=True)
        g = torch.randn(1, 1, 1, 1, 2, dtype=torch.float64, requires_grad=True)
        names, parameters = zip(*fusion.named_parameters())
        def functional(f, g, *values):
            return torch.func.functional_call(fusion, dict(zip(names, values)), (f, g))
        self.assertTrue(torch.autograd.gradcheck(functional, (f, g, *parameters)))

    def test_fusion_bias_config_and_noncontiguous_inputs(self):
        for bias in (False, True):
            fusion = ResidualFusion(3, content_channels=2, bias=bias).double()
            self.assertEqual(sum(p.numel() for p in fusion.parameters()), 6 + 3 * bias)
            f = torch.randn(2, 3, 2, 3, 4, dtype=torch.float64).transpose(2, 4)
            g = torch.randn(2, 2, 2, 3, 4, dtype=torch.float64).transpose(2, 4)
            self.assertFalse(f.is_contiguous())
            torch.testing.assert_close(fusion(f, g), f + fusion.phi(g))
            if not bias:
                torch.testing.assert_close(fusion(f, torch.zeros_like(g)), f, rtol=0, atol=0)

    def test_fusion_invalid_inputs_and_configuration(self):
        fusion = ResidualFusion(2, content_channels=3, bias=False).double()
        f = torch.ones(2, 2, 2, 3, 4, dtype=torch.float64)
        g = torch.ones(2, 3, 2, 3, 4, dtype=torch.float64)
        for bad_f, bad_g in ((None, g), (f[0], g), (f[:, :1], g), (f[:, :, :0], g),
                             (f, None), (f, g[:1]), (f, g[:, :2]), (f, g[..., :1]),
                             (f, g.float()), (f.half(), g.half()), (f, g.to('meta'))):
            with self.assertRaises(ValueError):
                fusion(bad_f, bad_g)
        for value in (float('nan'), float('inf'), -float('inf')):
            for index in (0, 1):
                args = [f.clone(), g.clone()]
                args[index].flatten()[0] = value
                with self.assertRaises(ValueError):
                    fusion(*args)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16), self.assertRaises(ValueError):
            fusion(f, g)
        with self.assertRaises(ValueError):
            fusion.float()(f, g)
        with self.assertRaises(TypeError):
            ResidualFusion(2, content_channels=3)
        for kwargs in ({'channels': 0, 'content_channels': 3, 'bias': True},
                       {'channels': 2, 'content_channels': True, 'bias': False},
                       {'channels': 2, 'content_channels': 3, 'bias': 1}):
            with self.assertRaises(ValueError):
                ResidualFusion(**kwargs)

    def test_complete_call_order_sources_and_no_repeated_build(self):
        model = Segmentor(micro()).double()
        names = ('encoder', 'coarse_head', 'space_to_node', 'relation', 'node_to_space', 'fusion', 'decoder')
        args, results, order = {}, {}, []
        handles = []
        def before(name):
            def record(module, values):
                order.append(name)
                args[name] = values
            return record
        def after(name):
            def record(module, values, result):
                results[name] = result
            return record
        for name in names:
            module = getattr(model, name)
            handles.extend((module.register_forward_pre_hook(before(name)), module.register_forward_hook(after(name))))
        image = torch.randn(2, 1, 9, 10, 11, dtype=torch.float64)
        try:
            output = model(image)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(order, list(names))
        self.assertIs(args['encoder'][0], image)
        f, skips = results['encoder']
        self.assertIs(args['coarse_head'][0], f)
        self.assertIs(args['space_to_node'][0], f)
        self.assertIs(args['space_to_node'][1], results['coarse_head'].probabilities)
        nodes = results['space_to_node']
        for actual, expected in zip(args['relation'], (nodes.z0, nodes.centroid, nodes.size, nodes.confidence)):
            self.assertIs(actual, expected)
        self.assertIs(args['node_to_space'][0], f)
        self.assertIs(args['node_to_space'][1], results['relation'])
        self.assertIsNot(args['node_to_space'][1], nodes.z0)
        self.assertIs(args['fusion'][0], f)
        self.assertIs(args['fusion'][1], results['node_to_space'])
        self.assertIs(args['decoder'][0], results['fusion'])
        self.assertIsNot(args['decoder'][0], f)
        self.assertIs(args['decoder'][1], skips)
        self.assertIs(output.coarse_logits, results['coarse_head'].logits)
        self.assertIs(output.final_logits, results['decoder'])

    def test_micro_config_output_shapes(self):
        config = micro()
        raw = json.loads((ROOT / 'configs/segmentor_micro.json').read_text(encoding='utf-8'))
        model = Segmentor(config)
        output = model(torch.randn(*raw['probe']['input_shape_bcdhw']))
        self.assertIsInstance(output, SegmentorOutput)
        self.assertEqual(output.coarse_logits.shape, (1, 16, 5, 5, 5))
        self.assertEqual(output.final_logits.shape, (1, 16, 17, 18, 19))

    def test_all_axis_parities_singletons_and_variable_pyramids(self):
        config = micro()
        configs = (config, replace(config, backbone=replace(config.backbone,
                    downsample_strides=((1, 2, 2), (2, 1, 2)))))
        shapes = list(itertools.product((5, 6), repeat=3)) + [(1, 1, 1), (1, 6, 7), (7, 1, 6), (6, 7, 1)]
        for config in configs:
            model = Segmentor(config)
            with torch.no_grad():
                for spatial in shapes:
                    with self.subTest(strides=config.backbone.downsample_strides, shape=spatial):
                        image = torch.randn(2, 1, *spatial)
                        output = model(image)
                        self.assertEqual(output.final_logits.shape, (2, 16, *spatial))
                        self.assertEqual(output.coarse_logits.shape, (2, 16, *config.backbone.spatial_pyramid(spatial)[-1]))

    def test_manual_composition_matches_segmentor(self):
        model = Segmentor(micro()).double()
        image = torch.randn(1, 1, 7, 8, 9, dtype=torch.float64)
        encoded = model.encoder(image)
        coarse = model.coarse_head(encoded.deepest)
        nodes = model.space_to_node(encoded.deepest, coarse.probabilities)
        zK = model.relation(nodes.z0, nodes.centroid, nodes.size, nodes.confidence)
        g = model.node_to_space(encoded.deepest, zK)
        final = model.decoder(model.fusion(encoded.deepest, g), encoded.skips)
        output = model(image)
        torch.testing.assert_close(output.coarse_logits, coarse.logits, rtol=0, atol=0)
        torch.testing.assert_close(output.final_logits, final, rtol=0, atol=0)

    def test_final_only_backward_all_modules_and_intermediates(self):
        model = self.controlled()
        image = torch.rand(2, 1, 5, 6, 7, dtype=torch.float64, requires_grad=True)
        d = model.forward_with_diagnostics(image)
        intermediates = (image, d.encoder.deepest, *d.encoder.skips, d.coarse.logits, d.coarse.probabilities,
                         d.nodes.z0, d.nodes.centroid, d.nodes.size, d.nodes.confidence,
                         d.relation.zK, d.writeback.G, d.fused)
        for t in intermediates:
            t.retain_grad()
        # Only final logits contribute: no coarse auxiliary loss or optimizer.
        d.output.final_logits.sum().backward()
        for t in intermediates:
            self.assertIsNotNone(t.grad)
            self.assertTrue(torch.isfinite(t.grad).all())
            self.assertGreater(t.grad.abs().sum().item(), 0)
        for name in ('encoder', 'coarse_head', 'relation', 'node_to_space', 'fusion', 'decoder'):
            module = getattr(model, name)
            total = 0.
            for parameter_name, p in module.named_parameters():
                with self.subTest(module=name, parameter=parameter_name):
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
                    total += p.grad.abs().sum().item()
            self.assertGreater(total, 0)

    def test_final_gradient_to_coarse_is_via_writeback_not_skips(self):
        model = self.controlled()
        image = torch.rand(1, 1, 5, 6, 7, dtype=torch.float64)
        parameters = (model.coarse_head.projection.weight, model.relation.W_m.weight,
                      next(model.encoder.parameters()), model.decoder.logits.weight)
        original = model(image)
        normal_grad = torch.autograd.grad(original.final_logits.sum(), parameters)
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in normal_grad))
        # TEST-ONLY causal intervention: identical G values, sever its autograd
        # connection. Encoder/decoder paths remain, graph->coarse must disappear.
        hook = model.node_to_space.register_forward_hook(lambda module, args, result: result.detach())
        try:
            severed = model(image)
        finally:
            hook.remove()
        torch.testing.assert_close(original.final_logits, severed.final_logits, rtol=0, atol=0)
        severed_grad = torch.autograd.grad(severed.final_logits.sum(), parameters, allow_unused=True)
        self.assertIsNone(severed_grad[0])
        self.assertIsNone(severed_grad[1])
        for g in severed_grad[2:]:
            self.assertIsNotNone(g)
            self.assertTrue(torch.isfinite(g).all())
            self.assertGreater(g.abs().sum().item(), 0)

    def test_remove_or_change_G_changes_only_final_branch(self):
        model = self.controlled()
        image = torch.rand(1, 1, 5, 6, 7, dtype=torch.float64)
        original = model(image)
        for change in (lambda g: torch.zeros_like(g), lambda g: g + .25):
            hook = model.node_to_space.register_forward_hook(lambda module, args, result: change(result))
            try:
                changed = model(image)
            finally:
                hook.remove()
            torch.testing.assert_close(original.coarse_logits, changed.coarse_logits, rtol=0, atol=0)
            self.assertGreater((original.final_logits - changed.final_logits).abs().max().item(), 1e-8)

    def test_normal_forward_no_diagnostics_or_cached_intermediates(self):
        model = Segmentor(micro())
        image = torch.randn(1, 1, 7, 8, 9)
        refs = []
        def track(module, args, result):
            if isinstance(result, torch.Tensor):
                refs.append(weakref.ref(result))
            else:
                for value in result:
                    if isinstance(value, torch.Tensor):
                        refs.append(weakref.ref(value))
        handles = [getattr(model, name).register_forward_hook(track) for name in
                   ('encoder', 'space_to_node', 'relation', 'node_to_space', 'fusion')]
        before = {name: set(vars(module)) for name, module in model.named_modules()}
        try:
            with torch.no_grad(), patch.object(model.relation, 'forward', wraps=model.relation.forward) as relation, \
                    patch.object(model.node_to_space, 'forward', wraps=model.node_to_space.forward) as writeback:
                output = model(image)
                self.assertIs(relation.call_args.kwargs['return_diagnostics'], False)
                self.assertIs(writeback.call_args.kwargs['return_diagnostics'], False)
                # Mocks themselves retain inputs in call history; release those
                # test-owned references before auditing model-owned retention.
                relation.reset_mock()
                writeback.reset_mock()
        finally:
            for h in handles:
                h.remove()
        self.assertEqual(output._fields, ('coarse_logits', 'final_logits'))
        self.assertEqual(before, {name: set(vars(module)) for name, module in model.named_modules()})
        gc.collect()
        self.assertTrue(refs)
        self.assertTrue(all(reference() is None for reference in refs))

    def test_diagnostics_match_normal_values_and_gradients(self):
        model = self.controlled()
        image = torch.rand(1, 1, 5, 6, 7, dtype=torch.float64, requires_grad=True)
        normal = model(image)
        diagnostic = model.forward_with_diagnostics(image)
        self.assertIsInstance(diagnostic, SegmentorDiagnostics)
        for a, b in zip(normal, diagnostic.output):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(len(diagnostic.relation.rounds), model.config.rounds)
        for t in (diagnostic.nodes.centroid, diagnostic.relation.zK, diagnostic.writeback.A, diagnostic.fused):
            self.assertTrue(t.requires_grad)
        gn, = torch.autograd.grad(normal.final_logits.sum(), image)
        gd, = torch.autograd.grad(diagnostic.output.final_logits.sum(), image)
        torch.testing.assert_close(gn, gd, rtol=0, atol=0)

    def test_fp32_random_micro_backward_finite_without_nonzero_element_requirement(self):
        model = Segmentor(micro())
        image = torch.randn(2, 1, 7, 8, 9, requires_grad=True)
        out = model(image)
        (out.final_logits * torch.randn_like(out.final_logits)).sum().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        for name, p in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_batch_independence_noncontiguous_and_eval(self):
        model = Segmentor(micro()).double().eval()
        image = torch.randn(2, 1, 5, 6, 7, dtype=torch.float64).transpose(2, 4)
        self.assertFalse(image.is_contiguous())
        out = model(image)
        for b in range(2):
            individual = model(image[b:b+1])
            for actual, expected in zip(out, individual):
                torch.testing.assert_close(actual[b:b+1], expected, rtol=1e-10, atol=1e-11)

    def test_segmentor_image_only_interface(self):
        model = Segmentor(micro())
        self.assertEqual(tuple(inspect.signature(model.forward).parameters), ('image',))
        self.assertEqual(tuple(inspect.signature(model.forward_with_diagnostics).parameters), ('image',))
        image = torch.randn(1, 1, 5, 6, 7)
        for keyword in ('label', 'mask', 'organ_presence', 'P', 'attention', 'return_diagnostics'):
            with self.assertRaises(TypeError):
                model(image, **{keyword: None})
        with self.assertRaises(TypeError):
            model(image, torch.zeros(1))

    def test_segmentor_invalid_input_rejected(self):
        model = Segmentor(micro())
        image = torch.randn(1, 1, 5, 6, 7)
        for bad in (None, image[0], image[:, :, :0], image.expand(1, 2, 5, 6, 7),
                    image.long(), image.half(), image.double(), torch.full_like(image, float('nan')),
                    torch.full_like(image, float('inf'))):
            with self.assertRaises(ValueError):
                model(bad)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16), self.assertRaises(ValueError):
            model(image)
        with self.assertRaises(ValueError):
            Segmentor(None)


if __name__ == '__main__':
    unittest.main()

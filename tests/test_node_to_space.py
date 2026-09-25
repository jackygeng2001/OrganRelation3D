"""CPU-only Node-to-Space formula, axis, writeback and gradient tests."""
import inspect
import itertools
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True
    from organ_relation.node_to_space import NodeToSpace, NodeToSpaceResult
    from organ_relation.backbone import Encoder3D
    from organ_relation.backbone_config import BackboneConfig
    from organ_relation.coarse_head import CoarseHead
    from organ_relation.space_to_node import SpaceToNode
    from organ_relation.dynamic_relation import DynamicRelation


def scalar_projection(weight, vector):
    return torch.stack([sum(weight[a, c] * vector[c] for c in range(vector.numel()))
                        for a in range(weight.shape[0])])


def explicit_reference(model, features, zK):
    """Independent scalar projections and batch/node/voxel loops, with autograd."""
    shape = features.shape[2:]
    positions = list(itertools.product(*(range(n) for n in shape)))
    all_g, all_q, all_k, all_v, all_a = [], [], [], [], []
    for b in range(features.shape[0]):
        queries = [scalar_projection(model.W_Q.weight, zK[b, i]) for i in range(15)]
        values = [scalar_projection(model.W_V.weight, zK[b, i]) for i in range(15)]
        keys = [scalar_projection(model.W_K.weight, features[b, :, d, h, w]) for d, h, w in positions]
        attention = []
        for i in range(15):
            attention.append(torch.stack([
                torch.sigmoid(sum(queries[i][a] * key[a] for a in range(model.attention_channels))
                              / math.sqrt(model.attention_channels) + model.beta[i])
                for key in keys]))
        content = []
        for g in range(model.content_channels):
            content.append(torch.stack([sum(attention[i][x] * values[i][g] for i in range(15))
                                        for x in range(len(positions))]).reshape(shape))
        all_g.append(torch.stack(content))
        all_q.append(torch.stack(queries))
        all_k.append(torch.stack(keys))
        all_v.append(torch.stack(values))
        all_a.append(torch.stack(attention))
    return tuple(torch.stack(items) for items in (all_g, all_q, all_k, all_v, all_a))


@unittest.skipUnless(TORCH_AVAILABLE, 'isolated PyTorch CPU environment required')
class NodeToSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(20260925)

    def inputs(self, channels=3, batch=2, spatial=(2, 3, 2), dtype=None, grad=False):
        dtype = torch.float64 if dtype is None else dtype
        return (torch.randn(batch, channels, *spatial, dtype=dtype, requires_grad=grad),
                torch.randn(batch, 15, channels, dtype=dtype, requires_grad=grad))

    def model(self, channels=3, da=4, cg=2, dtype=None):
        return NodeToSpace(channels, attention_channels=da, content_channels=cg).to(
            dtype=torch.float64 if dtype is None else dtype)

    def assertReferenceClose(self, actual, expected, rtol=2e-12, atol=2e-13):
        for name, a, e in zip(NodeToSpaceResult._fields, actual, expected):
            with self.subTest(field=name):
                torch.testing.assert_close(a, e, rtol=rtol, atol=atol)

    def test_projection_shapes_axes_and_spatial_flatten_order(self):
        model = self.model()
        f, z = self.inputs()
        out = model(f, z, return_diagnostics=True)
        expected_shapes = ((2, 2, 2, 3, 2), (2, 15, 4), (2, 12, 4), (2, 15, 2), (2, 15, 12))
        self.assertEqual(tuple(tuple(t.shape) for t in out), expected_shapes)
        for b, i in ((0, 2), (1, 13)):
            torch.testing.assert_close(out.Q[b, i], model.W_Q.weight @ z[b, i])
            torch.testing.assert_close(out.V[b, i], model.W_V.weight @ z[b, i])
        for b in range(2):
            for x, (d, h, w) in enumerate(itertools.product(range(2), range(3), range(2))):
                torch.testing.assert_close(out.K[b, x], model.W_K.weight @ f[b, :, d, h, w])

    def test_three_bias_free_matrices_and_fifteen_trainable_betas(self):
        model = self.model(channels=3, da=4, cg=2)
        for projection in (model.W_Q, model.W_K, model.W_V):
            self.assertIsNone(projection.bias)
        self.assertEqual(model.beta.shape, (15,))
        self.assertTrue(model.beta.requires_grad)
        self.assertEqual(torch.count_nonzero(model.beta).item(), 0)
        self.assertEqual(set(dict(model.named_parameters())), {'W_Q.weight', 'W_K.weight', 'W_V.weight', 'beta'})
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3 * (2 * 4 + 2) + 15)
        initialized = NodeToSpace(3, attention_channels=2, content_channels=4, beta_init=.25)
        torch.testing.assert_close(initialized.beta, torch.full_like(initialized.beta, .25))

    def test_hand_attention_and_sqrt_scaling(self):
        model = self.model(channels=1, da=4, cg=1)
        with torch.no_grad():
            model.W_Q.weight.copy_(torch.tensor([[1.], [2.], [-1.], [.5]], dtype=torch.float64))
            model.W_K.weight.copy_(torch.tensor([[.5], [-.25], [1.], [4.]], dtype=torch.float64))
            model.W_V.weight.fill_(1)
            model.beta[0] = .2
        f = torch.tensor([2., -1.], dtype=torch.float64).reshape(1, 1, 1, 1, 2)
        z = torch.zeros(1, 15, 1, dtype=f.dtype)
        z[0, 0, 0] = 1.5
        out = model(f, z, return_diagnostics=True)
        # Q=[1.5,3,-1.5,.75], K(x0)=[1,-.5,2,8], dot=3.
        # dot(x1)=-1.5; sqrt(da)=2; beta applies AFTER scaling.
        expected = f.new_tensor([1 / (1 + math.exp(-1.7)), 1 / (1 + math.exp(.55))])
        torch.testing.assert_close(out.A[0, 0], expected, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(out.G.flatten(), 1.5 * expected, rtol=1e-14, atol=1e-14)
        unscaled = torch.sigmoid(f.new_tensor([3.2, -1.3]))
        self.assertFalse(torch.allclose(out.A[0, 0], unscaled))

    def test_beta_broadcasts_over_nodes_not_space_and_has_independent_gradients(self):
        model = self.model(channels=1, da=1, cg=1)
        f, z = self.inputs(channels=1, spatial=(1, 2, 3))
        with torch.no_grad():
            model.W_Q.weight.zero_()
            model.beta.copy_(torch.linspace(-2, 2, 15, dtype=f.dtype))
        out = model(f, z, return_diagnostics=True)
        expected = model.beta.sigmoid()[None, :, None].expand(2, 15, 6)
        torch.testing.assert_close(out.A, expected, rtol=0, atol=0)
        gradient, = torch.autograd.grad(out.A[:, 7].sum(), model.beta)
        self.assertEqual(torch.count_nonzero(gradient).item(), 1)
        torch.testing.assert_close(gradient[7], 12 * model.beta[7].sigmoid() * (1 - model.beta[7].sigmoid()))

    def test_sigmoid_not_softmax_on_either_axis(self):
        model = self.model()
        f, z = self.inputs()
        with torch.no_grad():
            model.W_Q.weight.zero_()
        out = model(f, z, return_diagnostics=True)
        torch.testing.assert_close(out.A, torch.full_like(out.A, .5), rtol=0, atol=0)
        torch.testing.assert_close(out.A.sum(dim=1), torch.full_like(out.A[:, 0], 7.5))
        torch.testing.assert_close(out.A.sum(dim=2), torch.full_like(out.A[:, :, 0], 6.))

    def test_many_organs_and_many_positions_can_all_have_high_response(self):
        model = self.model(channels=1, da=1, cg=1)
        with torch.no_grad():
            model.W_Q.weight.fill_(1)
            model.W_K.weight.fill_(1)
        f = torch.full((2, 1, 2, 2, 3), 3., dtype=torch.float64)
        z = torch.full((2, 15, 1), 3., dtype=f.dtype)
        out = model(f, z, return_diagnostics=True)
        self.assertTrue((out.A > .999).all())
        self.assertTrue((out.A.sum(dim=1) > 14).all())
        self.assertTrue((out.A.sum(dim=2) > 11).all())

    def test_single_nonzero_value_node_is_only_writeback_source(self):
        model = self.model(channels=2, da=3, cg=2)
        f, _ = self.inputs(channels=2)
        z = torch.zeros(2, 15, 2, dtype=f.dtype)
        z[:, 6] = f.new_tensor([2., -3.])
        out = model(f, z, return_diagnostics=True)
        expected = (out.V[:, 6, :, None] * out.A[:, 6, None, :]).reshape_as(out.G)
        torch.testing.assert_close(out.G, expected)
        other_nodes = [i for i in range(15) if i != 6]
        self.assertEqual(torch.count_nonzero(out.V[:, other_nodes]).item(), 0)
        self.assertTrue((out.A[:, other_nodes] > 0).all())

    def test_multinode_sum_and_distinguish_value_from_feature_writeback(self):
        model = self.model(channels=1, da=1, cg=1)
        with torch.no_grad():
            model.W_Q.weight.zero_()
            model.W_V.weight.fill_(2)
        f = torch.tensor([10., -4.], dtype=torch.float64).reshape(1, 1, 1, 1, 2)
        z = torch.zeros(1, 15, 1, dtype=f.dtype)
        z[0, 0, 0], z[0, 1, 0] = 1., 3.
        out = model(f, z, return_diagnostics=True)
        # A=.5; V0=2, V1=6, so G(x)=.5*2+.5*6=4 at both positions.
        torch.testing.assert_close(out.G, torch.full_like(f, 4.), rtol=0, atol=0)
        wrong = out.A.sum(dim=1).reshape_as(f) * f
        torch.testing.assert_close(wrong.flatten(), f.new_tensor([75., -30.]))
        self.assertFalse(torch.allclose(out.G, wrong))

    def test_fp64_all_intermediates_and_output_match_scalar_reference(self):
        model = self.model()
        inputs = self.inputs()
        self.assertReferenceClose(model(*inputs, return_diagnostics=True), explicit_reference(model, *inputs))

    def test_input_and_all_parameter_gradients_match_scalar_reference(self):
        model = self.model(channels=2, da=3, cg=2)
        inputs = self.inputs(channels=2, spatial=(1, 2, 2), grad=True)
        actual = model(*inputs, return_diagnostics=True)
        reference = explicit_reference(model, *inputs)
        # Separate probes check the paths through A and through G.
        leaves = (*inputs, *model.parameters())
        names = ('F', 'zK', *dict(model.named_parameters()))
        for field in (0, 4):
            weights = torch.randn_like(actual[field])
            ag = torch.autograd.grad((actual[field] * weights).sum(), leaves, retain_graph=True, allow_unused=True)
            rg = torch.autograd.grad((reference[field] * weights).sum(), leaves, retain_graph=True, allow_unused=True)
            for name, a, r in zip(names, ag, rg):
                with self.subTest(output=('G' if field == 0 else 'A'), gradient=name):
                    if field == 4 and name == 'W_V.weight':
                        self.assertIsNone(a)
                        self.assertIsNone(r)
                    else:
                        self.assertIsNotNone(a)
                        self.assertTrue(torch.isfinite(a).all())
                        self.assertGreater(a.abs().sum().item(), 0)
                        torch.testing.assert_close(a, r, rtol=3e-11, atol=3e-12)

    def test_fp64_gradcheck_inputs_attention_and_content(self):
        model = self.model(channels=2, da=3, cg=2)
        inputs = self.inputs(channels=2, batch=1, spatial=(1, 1, 2), grad=True)
        def outputs(f, z):
            result = model(f, z, return_diagnostics=True)
            return result.G, result.A
        self.assertTrue(torch.autograd.gradcheck(outputs, inputs))

    def test_fp64_gradcheck_all_parameters_and_inputs(self):
        model = self.model(channels=1, da=2, cg=1)
        inputs = self.inputs(channels=1, batch=1, spatial=(1, 1, 2), grad=True)
        names, parameters = zip(*model.named_parameters())
        def functional(f, z, *values):
            return torch.func.functional_call(model, dict(zip(names, values)), (f, z))
        self.assertTrue(torch.autograd.gradcheck(functional, (*inputs, *parameters)))

    def test_fp32_against_fp64(self):
        model32 = self.model(dtype=torch.float32)
        model64 = self.model()
        model64.load_state_dict(model32.state_dict())
        f, z = self.inputs(dtype=torch.float32)
        actual = model32(f, z, return_diagnostics=True)
        expected = model64(f.double(), z.double(), return_diagnostics=True)
        self.assertReferenceClose(tuple(t.double() for t in actual), expected, rtol=5e-6, atol=5e-7)
        self.assertTrue(all(t.dtype == torch.float32 and t.device == f.device for t in actual))

    def test_batch_independence_and_different_dimensions(self):
        for batch, channels, da, cg, spatial in ((1, 1, 1, 1, (1, 1, 1)), (2, 2, 4, 3, (1, 2, 3)),
                                                (3, 4, 2, 5, (2, 1, 2))):
            with self.subTest(batch=batch, C=channels, da=da, Cg=cg):
                model = self.model(channels=channels, da=da, cg=cg)
                f, z = self.inputs(channels=channels, batch=batch, spatial=spatial)
                out = model(f, z)
                self.assertEqual(out.shape, (batch, cg, *spatial))
                for b in range(batch):
                    torch.testing.assert_close(out[b:b+1], model(f[b:b+1], z[b:b+1]))

    def test_noncontiguous_and_spatial_permutation(self):
        model = self.model()
        f, z = self.inputs()
        original = model(f, z, return_diagnostics=True)
        fp = f.permute(0, 1, 4, 2, 3)
        zp = z.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(fp.is_contiguous())
        self.assertFalse(zp.is_contiguous())
        changed = model(fp, zp, return_diagnostics=True)
        self.assertReferenceClose(changed, explicit_reference(model, fp, zp))
        torch.testing.assert_close(changed.G, original.G.permute(0, 1, 4, 2, 3))
        torch.testing.assert_close(changed.A.reshape(2, 15, *fp.shape[2:]),
                                   original.A.reshape(2, 15, *f.shape[2:]).permute(0, 1, 4, 2, 3))

    def test_diagnostics_do_not_change_values_or_gradients_and_inputs_unchanged(self):
        model = self.model()
        inputs = self.inputs(grad=True)
        copies = tuple(t.clone() for t in inputs)
        diagnostics = model(*inputs, return_diagnostics=True)
        default = model(*inputs)
        self.assertIsInstance(diagnostics, NodeToSpaceResult)
        self.assertTrue(all(t.requires_grad for t in diagnostics))
        torch.testing.assert_close(default, diagnostics.G, rtol=0, atol=0)
        weight = torch.randn_like(default)
        ga = torch.autograd.grad((default * weight).sum(), inputs)
        gb = torch.autograd.grad((diagnostics.G * weight).sum(), inputs)
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for t, old in zip(inputs, copies):
            torch.testing.assert_close(t, old, rtol=0, atol=0)

    def test_accepted_module_chain_uses_original_features_and_updated_nodes(self):
        config = BackboneConfig(**json.loads((ROOT / 'configs/backbone_micro.json').read_text(encoding='utf-8'))['model'])
        encoder = Encoder3D(config).double()
        channels = config.channels[-1]
        head = CoarseHead(channels, bias=True).double()
        relation = DynamicRelation(channels, relation_channels=3, rounds=2).double()
        with torch.no_grad():
            relation.relation_hidden.weight.fill_(.01)
            relation.relation_hidden.bias.fill_(.5)
        writeback = self.model(channels=channels, da=3, cg=2)
        image = torch.randn(2, 1, 9, 10, 11, dtype=torch.float64, requires_grad=True)
        f = encoder(image).deepest
        before = f.clone()
        coarse = head(f)
        nodes = SpaceToNode(epsilon=.01)(f, coarse.probabilities)
        zK = relation(nodes.z0, nodes.centroid, nodes.size, nodes.confidence)
        self.assertFalse(torch.allclose(zK, nodes.z0))
        captured = {}
        def hook(name):
            def capture(module, args):
                captured[name] = args[0]
            return capture
        handles = [getattr(writeback, name).register_forward_pre_hook(hook(name)) for name in ('W_Q', 'W_K', 'W_V')]
        retained = (f, coarse.logits, coarse.probabilities, nodes.z0, nodes.centroid, nodes.size, nodes.confidence, zK)
        for t in retained:
            t.retain_grad()
        try:
            g = writeback(f, zK)
        finally:
            for handle in handles:
                handle.remove()
        self.assertIs(captured['W_Q'], zK)
        self.assertIs(captured['W_V'], zK)
        torch.testing.assert_close(captured['W_K'], before.flatten(2).transpose(1, 2), rtol=0, atol=0)
        torch.testing.assert_close(f, before, rtol=0, atol=0)
        (g * torch.randn_like(g)).sum().backward()  # Synthetic probe only; no method loss or Segmentor.
        for t in (image, *retained):
            self.assertIsNotNone(t.grad)
            self.assertTrue(torch.isfinite(t.grad).all())
            self.assertGreater(t.grad.abs().sum().item(), 0)
        for name, module in (('encoder', encoder), ('head', head), ('relation', relation), ('writeback', writeback)):
            for parameter_name, p in module.named_parameters():
                with self.subTest(module=name, parameter=parameter_name):
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
                    self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_nonfinite_inputs_rejected(self):
        model = self.model()
        for index, name in enumerate(('features', 'zK')):
            for value in (float('nan'), float('inf'), -float('inf')):
                inputs = list(self.inputs())
                inputs[index].reshape(-1)[0] = value
                with self.subTest(input=name, value=value), self.assertRaisesRegex(ValueError, name + '.*finite'):
                    model(*inputs)

    def test_nonfinite_projection_scores_and_writeback_rejected(self):
        for target in ('Q', 'K', 'V', 'scores', 'G'):
            with self.subTest(target=target):
                model = self.model(channels=1, da=1, cg=1, dtype=torch.float32)
                f = torch.ones(1, 1, 1, 1, 1)
                z = torch.ones(1, 15, 1)
                with torch.no_grad():
                    model.W_Q.weight.fill_(1)
                    model.W_K.weight.fill_(1)
                    model.W_V.weight.fill_(1)
                    if target in ('Q', 'K', 'V'):
                        getattr(model, 'W_' + target).weight.fill_(float('nan'))
                    elif target == 'scores':
                        model.W_Q.weight.fill_(1e20)
                        model.W_K.weight.fill_(1e20)
                    else:
                        model.W_V.weight.fill_(1e38)
                with self.assertRaises(ValueError):
                    model(f, z)

    def test_invalid_shapes_rejected_without_broadcasting(self):
        model = self.model()
        f, z = self.inputs()
        for bad_f, bad_z in ((None, z), (f[0], z), (f[:, :2], z), (f[:, :, :0], z),
                             (f[:0], z[:0]), (f, None), (f, z[0]), (f, z[:1]),
                             (f, z[:, :14]), (f, z[..., :2]), (f, z.unsqueeze(0))):
            with self.assertRaises(ValueError):
                model(bad_f, bad_z)

    def test_invalid_dtype_device_autocast_and_module_dtype_rejected(self):
        model = self.model()
        f, z = self.inputs()
        for dtype in (torch.int64, torch.float16, torch.bfloat16, torch.complex128):
            with self.assertRaises(ValueError):
                model(f.to(dtype=dtype), z.to(dtype=dtype))
        with self.assertRaises(ValueError):
            model(f, z.float())
        with self.assertRaises(ValueError):
            model(f, z.to('meta'))
        with self.assertRaisesRegex(ValueError, 'module parameters'):
            model.float()(f, z)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, 'autocast'):
                model(f.float(), z.float())

    def test_interface_rejects_label_P_and_external_attention(self):
        model = self.model()
        f, z = self.inputs()
        self.assertEqual(tuple(inspect.signature(model.forward).parameters), ('features', 'zK', 'return_diagnostics'))
        for keyword in ('label', 'P', 'probabilities', 'attention', 'A', 'z0', 'mask'):
            with self.subTest(keyword=keyword), self.assertRaises(TypeError):
                model(f, z, **{keyword: None})
        with self.assertRaises(TypeError):
            model(f, z, torch.zeros(1))
        with self.assertRaises(ValueError):
            model(f, z, return_diagnostics=1)

    def test_invalid_constructor_configuration(self):
        for key in ('channels', 'attention_channels', 'content_channels'):
            for value in (0, -1, True, 1.5, '3'):
                kwargs = dict(channels=2, attention_channels=3, content_channels=4)
                kwargs[key] = value
                with self.assertRaises(ValueError):
                    NodeToSpace(**kwargs)
        for value in (float('nan'), float('inf'), -float('inf'), 1e100, True, '0'):
            with self.assertRaises(ValueError):
                NodeToSpace(2, attention_channels=3, content_channels=4, beta_init=value)


if __name__ == '__main__':
    unittest.main()

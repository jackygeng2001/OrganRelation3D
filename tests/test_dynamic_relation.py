"""CPU-only directed-edge, explicit-GRU, recurrence and gradient tests."""
import inspect
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True
    from organ_relation.dynamic_relation import DynamicRelation, FormulaGRU, RelationResult
    from organ_relation.coarse_head import CoarseHead
    from organ_relation.space_to_node import SpaceToNode


def reference_gru(gru, message, z):
    """One vector, directly using matrices; no production forward/helpers."""
    u = torch.sigmoid(torch.mv(gru.W_u.weight, message) + torch.mv(gru.U_u.weight, z) + gru.W_u.bias)
    rho = torch.sigmoid(torch.mv(gru.W_rho.weight, message) + torch.mv(gru.U_rho.weight, z) + gru.W_rho.bias)
    candidate = torch.tanh(torch.mv(gru.W_z.weight, message) + torch.mv(gru.U_z.weight, rho * z) + gru.W_z.bias)
    return (1 - u) * z + u * candidate


def reference_edge(model, zi, zj, ci, cj, si, sj, qi, qj):
    descriptor = torch.cat((zi, zj, cj - ci, si, sj, qi, qj))
    hidden = torch.relu(torch.mv(model.relation_hidden.weight, descriptor) + model.relation_hidden.bias)
    return torch.sigmoid((model.relation_output.weight[0] * hidden).sum() + model.relation_output.bias[0])


def reference_round(model, z, centroid, size, confidence):
    """Independent loops over batch/sender/receiver; all use the same old z."""
    all_alpha, all_message, all_z = [], [], []
    for b in range(z.shape[0]):
        rows = []
        for i in range(15):
            rows.append(torch.stack([
                z.new_zeros(()) if i == j else reference_edge(
                    model, z[b, i], z[b, j], centroid[b, i], centroid[b, j],
                    size[b, i], size[b, j], confidence[b, i], confidence[b, j])
                for j in range(15)]))
        alpha = torch.stack(rows)
        messages, updated = [], []
        for j in range(15):
            message = z.new_zeros(model.channels)
            for i in range(15):
                if i != j:
                    message = message + alpha[i, j] * torch.mv(model.W_m.weight, z[b, i])
            messages.append(message)
            updated.append(reference_gru(model.gru, message, z[b, j]))
        all_alpha.append(alpha)
        all_message.append(torch.stack(messages))
        all_z.append(torch.stack(updated))
    return torch.stack(all_alpha), torch.stack(all_message), torch.stack(all_z)


def reference_unroll(model, z, centroid, size, confidence):
    history = []
    for _ in range(model.rounds):
        alpha, message, z = reference_round(model, z, centroid, size, confidence)
        history.append((alpha, message, z))
    return z, history


@unittest.skipUnless(TORCH_AVAILABLE, 'isolated PyTorch CPU environment required')
class DynamicRelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(20260925)

    def inputs(self, channels=2, batch=2, dtype=None, grad=False):
        dtype = torch.float64 if dtype is None else dtype
        return tuple((.1 + .7 * torch.rand(batch, 15, c, dtype=dtype)).requires_grad_(grad)
                     for c in (channels, 3, 1, 1))

    def model(self, channels=2, hidden=3, rounds=2, dtype=None):
        dtype = torch.float64 if dtype is None else dtype
        model = DynamicRelation(channels, relation_channels=hidden, rounds=rounds).to(dtype=dtype)
        # Keep ReLU away from its kink for numerical gradient comparisons.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.uniform_(-.2, .2)
            model.relation_hidden.weight.uniform_(.01, .05)
            model.relation_hidden.bias.fill_(.5)
            model.relation_output.weight.fill_(.3)
            model.relation_output.bias.fill_(-.1)
        return model

    def test_descriptor_order_sender_receiver_and_centroid_sign(self):
        model = self.model(channels=3)
        z, c, s, q = self.inputs(channels=3)
        r = model._relation_descriptors(z, c, s, q)
        self.assertEqual(r.shape, (2, 15, 15, 13))
        for b, i, j in ((0, 2, 9), (1, 12, 4)):
            expected = torch.cat((z[b, i], z[b, j], c[b, j] - c[b, i], s[b, i], s[b, j], q[b, i], q[b, j]))
            torch.testing.assert_close(r[b, i, j], expected, rtol=0, atol=0)
            torch.testing.assert_close(r[b, j, i, 6:9], -r[b, i, j, 6:9], rtol=0, atol=0)

    def test_single_edge_enters_only_its_receiver(self):
        model = self.model()
        z, *_ = self.inputs()
        with torch.no_grad():
            model.W_m.weight.copy_(z.new_tensor([[1., 2.], [-1., .5]]))
        alpha = z.new_zeros(2, 15, 15)
        alpha[0, 3, 9] = 1
        alpha[1, 7, 2] = .25
        actual = model._aggregate_messages(z, alpha)
        expected = torch.zeros_like(z)
        expected[0, 9] = model.W_m.weight @ z[0, 3]
        expected[1, 2] = .25 * (model.W_m.weight @ z[1, 7])
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)
        self.assertIsNone(model.W_m.bias)

    def test_210_edges_exact_zero_diagonal_independent_sigmoid(self):
        model = self.model(rounds=1)
        with torch.no_grad():
            model.relation_output.weight.zero_()
            model.relation_output.bias.zero_()
        out = model(*self.inputs(), return_diagnostics=True)
        alpha = out.rounds[0].alpha
        self.assertEqual(torch.count_nonzero(alpha.diagonal(dim1=1, dim2=2)).item(), 0)
        self.assertEqual(torch.count_nonzero(alpha[0]).item(), 210)
        self.assertEqual(torch.count_nonzero(alpha[1]).item(), 210)
        expected = (.5 * (1 - torch.eye(15, dtype=alpha.dtype))).expand_as(alpha)
        torch.testing.assert_close(alpha, expected, rtol=0, atol=0)
        torch.testing.assert_close(alpha.sum(dim=1), torch.full_like(alpha[:, 0], 7.), rtol=0, atol=0)

    def test_sum_over_sender_not_mean_or_receiver(self):
        model = self.model(channels=1)
        z = torch.arange(1, 16, dtype=torch.float64).reshape(1, 15, 1)
        alpha = torch.arange(225, dtype=z.dtype).reshape(1, 15, 15) / 225
        alpha[:, torch.arange(15), torch.arange(15)] = 0
        with torch.no_grad():
            model.W_m.weight.fill_(2)
        actual = model._aggregate_messages(z, alpha)
        expected = torch.stack([sum(alpha[0, i, j] * 2 * z[0, i] for i in range(15) if i != j)
                                for j in range(15)]).unsqueeze(0)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, expected / 14))
        self.assertFalse(torch.allclose(actual, torch.bmm(alpha, 2 * z)))

    def test_directed_weights_can_be_asymmetric(self):
        model = self.model(channels=1, hidden=1)
        z, c, s, q = self.inputs(channels=1, batch=1)
        c[:, :, 0] = torch.linspace(0, 1, 15, dtype=c.dtype)
        with torch.no_grad():
            model.relation_hidden.weight.zero_()
            model.relation_hidden.weight[0, 2] = 1  # c_j[D] - c_i[D]
            model.relation_hidden.bias.fill_(.5)
            model.relation_output.weight.fill_(1)
            model.relation_output.bias.zero_()
        alpha = model._edge_weights(z, c, s, q)
        self.assertGreater(alpha[0, 2, 12].item(), alpha[0, 12, 2].item())
        torch.testing.assert_close(alpha[0, 2, 12], torch.sigmoid(.5 + c[0, 12, 0] - c[0, 2, 0]))

    def test_self_edge_has_exact_zero_input_and_parameter_gradients(self):
        model = self.model()
        inputs = self.inputs(grad=True)
        alpha = model(*inputs, return_diagnostics=True).rounds[0].alpha
        gradients = torch.autograd.grad(alpha.diagonal(dim1=1, dim2=2).sum(),
                                       (*inputs, *model.relation_hidden.parameters(), *model.relation_output.parameters()))
        for gradient in gradients:
            self.assertEqual(torch.count_nonzero(gradient).item(), 0)

    def test_fp64_vectorization_matches_loop_reference_every_round(self):
        model = self.model(channels=3, hidden=4, rounds=3)
        inputs = self.inputs(channels=3)
        actual = model(*inputs, return_diagnostics=True)
        expected, history = reference_unroll(model, *inputs)
        torch.testing.assert_close(actual.zK, expected, rtol=2e-12, atol=2e-13)
        for got, ref in zip(actual.rounds, history):
            for a, e in zip(got, ref):
                torch.testing.assert_close(a, e, rtol=2e-12, atol=2e-13)

    def test_gru_four_formulas_scalar_math_and_reset_order(self):
        gru = FormulaGRU(2).double()
        matrices = {
            'W_u': [[.2, -.1], [.3, .4]], 'U_u': [[-.2, .5], [.1, -.3]],
            'W_rho': [[.1, .6], [-.4, .2]], 'U_rho': [[.5, -.2], [.3, .1]],
            'W_z': [[.7, .1], [-.2, .6]], 'U_z': [[0., 1.3], [-.7, .2]],
        }
        biases = {'W_u': [.05, -.1], 'W_rho': [-1., 1.], 'W_z': [.1, -.05]}
        with torch.no_grad():
            for name, weight in matrices.items():
                getattr(gru, name).weight.copy_(torch.tensor(weight, dtype=torch.float64))
            for name, bias in biases.items():
                getattr(gru, name).bias.copy_(torch.tensor(bias, dtype=torch.float64))
        m, old = [.4, -.8], [.7, -1.1]
        dot = lambda row, vec: sum(a * b for a, b in zip(row, vec))
        sigmoid = lambda x: 1 / (1 + math.exp(-x))
        u = [sigmoid(dot(matrices['W_u'][a], m) + dot(matrices['U_u'][a], old) + biases['W_u'][a]) for a in range(2)]
        rho = [sigmoid(dot(matrices['W_rho'][a], m) + dot(matrices['U_rho'][a], old) + biases['W_rho'][a]) for a in range(2)]
        reset_old = [rho[a] * old[a] for a in range(2)]
        candidate = [math.tanh(dot(matrices['W_z'][a], m) + dot(matrices['U_z'][a], reset_old) + biases['W_z'][a]) for a in range(2)]
        expected = [(1 - u[a]) * old[a] + u[a] * candidate[a] for a in range(2)]
        wrong_candidate = [math.tanh(dot(matrices['W_z'][a], m) + rho[a] * dot(matrices['U_z'][a], old) + biases['W_z'][a]) for a in range(2)]
        wrong = [(1 - u[a]) * old[a] + u[a] * wrong_candidate[a] for a in range(2)]
        message = torch.tensor(m, dtype=torch.float64).expand(2, 15, 2)
        z = torch.tensor(old, dtype=torch.float64).expand_as(message)
        reset_inputs = []
        hook = gru.U_z.register_forward_pre_hook(lambda module, args: reset_inputs.append(args[0]))
        try:
            actual = gru(message, z)
        finally:
            hook.remove()
        torch.testing.assert_close(actual, z.new_tensor(expected).expand_as(z), rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(reset_inputs[0], z.new_tensor(reset_old).expand_as(z), rtol=1e-14, atol=1e-14)
        self.assertGreater(abs(rho[0] - rho[1]), .1)
        self.assertGreater(max(abs(a - b) for a, b in zip(expected, wrong)), .01)

    def test_update_gate_weights_candidate_not_old_state(self):
        gru = FormulaGRU(1).double()
        with torch.no_grad():
            for parameter in gru.parameters():
                parameter.zero_()
            gru.W_u.bias.fill_(math.log(.2 / .8))
            gru.W_z.bias.fill_(math.atanh(-.4))
        old = torch.full((1, 15, 1), 1.2, dtype=torch.float64)
        actual = gru(torch.zeros_like(old), old)
        torch.testing.assert_close(actual, torch.full_like(old, .8 * 1.2 + .2 * -.4))
        self.assertFalse(torch.allclose(actual, torch.full_like(old, .2 * 1.2 + .8 * -.4)))

    def test_synchronous_differs_from_sequential_node_updates(self):
        model = self.model(rounds=1)
        z, c, s, q = self.inputs(batch=1)
        actual = model(z, c, s, q)
        _, _, synchronous = reference_round(model, z, c, s, q)
        current = [z[0, i] for i in range(15)]
        for j in range(15):
            message = torch.zeros_like(current[j])
            for i in range(15):
                if i != j:
                    alpha = reference_edge(model, current[i], current[j], c[0, i], c[0, j], s[0, i], s[0, j], q[0, i], q[0, j])
                    message = message + alpha * torch.mv(model.W_m.weight, current[i])
            current[j] = reference_gru(model.gru, message, current[j])  # Intentionally wrong asynchronous reference.
        asynchronous = torch.stack(current).unsqueeze(0)
        torch.testing.assert_close(actual, synchronous)
        self.assertGreater((actual - asynchronous).abs().max().item(), 1e-4)

    def test_second_round_recomputes_edges_from_updated_z(self):
        model = self.model(rounds=2)
        z, c, s, q = self.inputs()
        with patch.object(model, '_edge_weights', wraps=model._edge_weights) as edges:
            out = model(z, c, s, q, return_diagnostics=True)
        self.assertEqual(edges.call_count, 2)
        self.assertIs(edges.call_args_list[0].args[0], z)
        self.assertIs(edges.call_args_list[1].args[0], out.rounds[0].z)
        self.assertGreater((out.rounds[0].alpha - out.rounds[1].alpha).abs().max().item(), 1e-5)
        expected, _, _ = reference_round(model, out.rounds[0].z, c, s, q)
        torch.testing.assert_close(out.rounds[1].alpha, expected)
        frozen_message = model._aggregate_messages(out.rounds[0].z, out.rounds[0].alpha)
        frozen_output = model.gru(frozen_message, out.rounds[0].z)
        self.assertGreater((out.zK - frozen_output).abs().max().item(), 1e-6)

    def test_parameter_count_sharing_and_no_extra_bias(self):
        models = [self.model(channels=3, hidden=4, rounds=k) for k in (1, 2, 4)]
        expected_count = 4 * (2 * 3 + 9) + 1 + 7 * 3**2 + 3 * 3
        for model in models:
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected_count)
            self.assertEqual(len(list(model.parameters())), 14)
            self.assertEqual(sum(isinstance(m, FormulaGRU) for m in model.modules()), 1)
            self.assertIsNone(model.W_m.bias)
            for name in ('U_u', 'U_rho', 'U_z'):
                self.assertIsNone(getattr(model.gru, name).bias)
            model.load_state_dict(models[0].state_dict())
            snapshots = []
            hook = model.gru.register_forward_pre_hook(lambda module, args: snapshots.append(tuple(id(p) for p in module.parameters())))
            try:
                model(*self.inputs(channels=3))
            finally:
                hook.remove()
            self.assertEqual(len(snapshots), model.rounds)
            self.assertTrue(all(s == snapshots[0] for s in snapshots))
        self.assertEqual(tuple(models[0].state_dict()), tuple(models[2].state_dict()))

    def test_fixed_attributes_unmodified_reused_and_differentiable(self):
        model = self.model(rounds=3)
        inputs = self.inputs(grad=True)
        copies = tuple(x.clone() for x in inputs)
        with patch.object(model, '_edge_weights', wraps=model._edge_weights) as edges:
            out = model(*inputs)
        for call in edges.call_args_list:
            for passed, original in zip(call.args[1:], inputs[1:]):
                self.assertIs(passed, original)
        for actual, original in zip(inputs, copies):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        gradients = torch.autograd.grad((out * torch.randn_like(out)).sum(), inputs)
        for name, gradient in zip(('z0', 'centroid', 'size', 'confidence'), gradients):
            with self.subTest(input=name):
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 1e-12)

    def test_input_and_all_parameter_gradients_match_loop_reference(self):
        model = self.model(rounds=2)
        inputs = self.inputs(batch=1, grad=True)
        actual = model(*inputs)
        expected, _ = reference_unroll(model, *inputs)
        weight = torch.randn_like(actual)
        leaves = (*inputs, *model.parameters())
        actual_gradients = torch.autograd.grad((actual * weight).sum(), leaves)
        expected_gradients = torch.autograd.grad((expected * weight).sum(), leaves)
        for name, actual_gradient, expected_gradient in zip(
            ('z0', 'centroid', 'size', 'confidence', *dict(model.named_parameters())), actual_gradients, expected_gradients
        ):
            with self.subTest(gradient=name):
                self.assertTrue(torch.isfinite(actual_gradient).all())
                self.assertGreater(actual_gradient.abs().sum().item(), 1e-12)
                torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-10, atol=2e-12)

    def test_fp64_gradcheck_all_four_inputs_through_two_rounds(self):
        model = self.model(channels=1, hidden=2, rounds=2)
        inputs = self.inputs(channels=1, batch=1, grad=True)
        self.assertTrue(torch.autograd.gradcheck(model, inputs))

    def test_gru_fp64_gradcheck_message_and_state(self):
        gru = self.model(channels=2).gru
        message = torch.randn(1, 15, 2, dtype=torch.float64, requires_grad=True)
        state = torch.randn_like(message, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(gru, (message, state)))

    def test_different_batch_channel_and_round_counts(self):
        for batch, channels, hidden, rounds in ((1, 1, 1, 1), (2, 2, 3, 2), (3, 4, 2, 3)):
            with self.subTest(batch=batch, channels=channels, rounds=rounds):
                model = self.model(channels=channels, hidden=hidden, rounds=rounds)
                inputs = self.inputs(channels=channels, batch=batch)
                out = model(*inputs, return_diagnostics=True)
                self.assertEqual(out.zK.shape, (batch, 15, channels))
                self.assertEqual(len(out.rounds), rounds)
                torch.testing.assert_close(out.zK, model(*inputs), rtol=0, atol=0)
                for b in range(batch):
                    torch.testing.assert_close(out.zK[b:b+1], model(*(x[b:b+1] for x in inputs)))

    def test_node_permutation_equivariance(self):
        model = self.model(rounds=3)
        inputs = self.inputs()
        permutation = torch.randperm(15)
        actual = model(*(x[:, permutation] for x in inputs), return_diagnostics=True)
        original = model(*inputs, return_diagnostics=True)
        torch.testing.assert_close(actual.zK, original.zK[:, permutation])
        for a, b in zip(actual.rounds, original.rounds):
            torch.testing.assert_close(a.alpha, b.alpha[:, permutation][:, :, permutation])

    def test_fp32_against_fp64(self):
        model32 = self.model(dtype=torch.float32)
        model64 = self.model()
        model64.load_state_dict(model32.state_dict())
        inputs = self.inputs(dtype=torch.float32)
        actual = model32(*inputs, return_diagnostics=True)
        expected = model64(*(x.double() for x in inputs), return_diagnostics=True)
        for round32, round64 in zip(actual.rounds, expected.rounds):
            for a, e in zip(round32, round64):
                self.assertEqual(a.dtype, torch.float32)
                torch.testing.assert_close(a.double(), e, rtol=3e-6, atol=3e-7)

    def test_noncontiguous_inputs(self):
        model = self.model()
        inputs = tuple(x.transpose(1, 2).contiguous().transpose(1, 2) for x in self.inputs())
        self.assertFalse(inputs[0].is_contiguous())
        actual = model(*inputs)
        expected = model(*(x.contiguous() for x in inputs))
        torch.testing.assert_close(actual, expected)

    def test_coarse_node_relation_chain_and_attribute_paths(self):
        f = torch.randn(2, 2, 2, 2, 3, dtype=torch.float64, requires_grad=True)
        head = CoarseHead(2, bias=True).double()
        coarse = head(f)
        nodes = SpaceToNode(epsilon=.01)(f, coarse.probabilities)
        model = self.model(rounds=2)
        for value in (coarse.logits, coarse.probabilities, nodes.z0, nodes.centroid, nodes.size, nodes.confidence):
            value.retain_grad()
        output = model(nodes.z0, nodes.centroid, nodes.size, nodes.confidence)
        (output * torch.randn_like(output)).sum().backward()  # Synthetic probe, not a training loss.
        for value in (f, coarse.logits, coarse.probabilities, nodes.z0, nodes.centroid, nodes.size, nodes.confidence):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        for parameter in (*head.parameters(), *model.parameters()):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_diagnostics_preserve_graph_without_changing_results(self):
        model = self.model()
        inputs = self.inputs(grad=True)
        diagnostic = model(*inputs, return_diagnostics=True)
        self.assertIsInstance(diagnostic, RelationResult)
        self.assertIs(diagnostic.zK, diagnostic.rounds[-1].z)
        expected = model(*inputs)
        torch.testing.assert_close(diagnostic.zK, expected, rtol=0, atol=0)
        for round_result in diagnostic.rounds:
            self.assertTrue(all(x.requires_grad for x in round_result))
        weight = torch.randn_like(expected)
        gd = torch.autograd.grad((diagnostic.zK * weight).sum(), inputs)
        gn = torch.autograd.grad((expected * weight).sum(), inputs)
        for a, e in zip(gd, gn):
            torch.testing.assert_close(a, e, rtol=0, atol=0)

    def test_nonfinite_inputs_rejected(self):
        model = self.model()
        for index, name in enumerate(('z0', 'centroid', 'size', 'confidence')):
            for bad in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(input=name, value=bad):
                    inputs = list(self.inputs())
                    inputs[index][0, 0, 0] = bad
                    with self.assertRaisesRegex(ValueError, name + '.*finite'):
                        model(*inputs)

    def test_invalid_shapes_and_no_broadcasting(self):
        model = self.model()
        inputs = self.inputs()
        for index in range(4):
            original = inputs[index]
            for bad in (original[0], original[:, :14], original[:0], original[:1],
                        original[..., :0], original.unsqueeze(-1), None):
                # B=1 for z0 is valid alone but must reject the unchanged B=2 attributes.
                with self.subTest(input=index, shape=getattr(bad, 'shape', None)):
                    changed = list(inputs)
                    changed[index] = bad
                    with self.assertRaises(ValueError):
                        model(*changed)
        for index, channels in enumerate((3, 2, 2, 3)):
            changed = list(inputs)
            changed[index] = torch.ones(2, 15, channels, dtype=inputs[0].dtype)
            with self.assertRaises(ValueError):
                model(*changed)

    def test_dtype_device_and_autocast_rejected(self):
        model = self.model()
        inputs = self.inputs()
        for index in range(4):
            changed = list(inputs)
            changed[index] = changed[index].float()
            with self.assertRaises(ValueError):
                model(*changed)
        for dtype in (torch.int64, torch.float16, torch.bfloat16):
            with self.assertRaises(ValueError):
                model(*(x.to(dtype=dtype) for x in inputs))
        with self.assertRaisesRegex(ValueError, 'module parameters'):
            model.float()(*inputs)
        with self.assertRaises(ValueError):
            model.double()(inputs[0], inputs[1].to('meta'), *inputs[2:])
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, 'autocast'):
                model(*inputs)

    def test_invalid_configuration_and_no_label_or_edge_override(self):
        for name in ('channels', 'relation_channels', 'rounds'):
            for bad in (0, -1, True, 1.5, '2'):
                kwargs = dict(channels=2, relation_channels=3, rounds=2)
                kwargs[name] = bad
                with self.assertRaises(ValueError):
                    DynamicRelation(**kwargs)
        model = self.model()
        self.assertEqual(tuple(inspect.signature(model.forward).parameters),
                         ('z0', 'centroid', 'size', 'confidence', 'return_diagnostics'))
        for kwargs in ({'label': None}, {'alpha': None}, {'mask': None}):
            with self.assertRaises(TypeError):
                model(*self.inputs(), **kwargs)
        with self.assertRaises(ValueError):
            model(*self.inputs(), return_diagnostics=1)

    def test_standalone_gru_rejects_bad_inputs(self):
        gru = FormulaGRU(2).double()
        z, *_ = self.inputs()
        for bad in (z[0], z[:, :14], z.float(), torch.full_like(z, float('nan'))):
            with self.assertRaises(ValueError):
                gru(bad, z)
        with self.assertRaises(ValueError):
            FormulaGRU(0)


if __name__ == '__main__':
    unittest.main()

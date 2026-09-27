"""Frozen JointLoss formulas on CPU synthetic tensors; all numeric choices are test-only."""
import inspect
import itertools
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

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
    from torch.nn import functional as functional
    from organ_relation.losses import JointLoss, BranchLoss, JointLossResult
    from organ_relation.models.segmentor import Segmentor
    from organ_relation.models.segmentor_config import SegmentorConfig


def reference_resize(logits, target_shape, align_corners):
    """Explicit trilinear coordinates and eight neighbors; no interpolate call."""
    axes = []
    for source, target in zip(logits.shape[2:], target_shape):
        axis = []
        for j in range(target):
            if align_corners:
                coordinate = j * (source - 1) / (target - 1) if target > 1 else 0.
            else:
                coordinate = (j + .5) * source / target - .5
            coordinate = min(max(coordinate, 0.), source - 1)
            low = math.floor(coordinate)
            axis.append(((low, 1 - (coordinate - low)), (min(low + 1, source - 1), coordinate - low)))
        axes.append(axis)
    batches = []
    for b in range(logits.shape[0]):
        channels = []
        for c in range(16):
            voxels = []
            for d, h, w in itertools.product(*(range(n) for n in target_shape)):
                voxels.append(sum(logits[b, c, di, hi, wi] * dw * hw * ww
                                  for (di, dw), (hi, hw), (wi, ww) in itertools.product(axes[0][d], axes[1][h], axes[2][w])))
            channels.append(torch.stack(voxels).reshape(target_shape))
        batches.append(torch.stack(channels))
    return torch.stack(batches)


def reference_branch(logits, label, epsilon):
    """Independent per-case/per-voxel/per-class definition, without gather/scatter."""
    cases = []
    positions = list(itertools.product(*(range(n) for n in label.shape[1:])))
    for b in range(logits.shape[0]):
        probabilities = []
        truth = []
        for d, h, w in positions:
            vector = logits[b, :, d, h, w]
            exps = (vector - vector.max()).exp()
            probabilities.append(exps / exps.sum())
            truth.append(int(label[b, d, h, w]))
        ce = -sum(torch.log(p[y] + epsilon) for p, y in zip(probabilities, truth)) / len(positions)
        dices = []
        for c in range(1, 16):
            intersection = sum(p[c] * int(y == c) for p, y in zip(probabilities, truth))
            predicted = sum(p[c] for p in probabilities)
            target = sum(int(y == c) for y in truth)
            dices.append((2 * intersection + epsilon) / (predicted + target + epsilon))
        dices = torch.stack(dices)
        dice_loss = 1 - sum(dices) / 15
        cases.append((ce, dices, dice_loss, ce + dice_loss))
    return tuple(torch.stack([case[k] for case in cases]) for k in range(4))


@unittest.skipUnless(TORCH_AVAILABLE, 'isolated PyTorch CPU environment required')
class JointLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(20260925)

    def inputs(self, dtype=None, grad=False):
        dtype = torch.float64 if dtype is None else dtype
        coarse = torch.randn(2, 16, 1, 2, 2, dtype=dtype, requires_grad=grad)
        final = torch.randn(2, 16, 2, 3, 3, dtype=dtype, requires_grad=grad)
        label = torch.randint(0, 16, (2, 2, 3, 3))
        return coarse, final, label

    def test_uniform_hand_calculation_background_and_foreground(self):
        epsilon = 1 / 16  # Deliberately large synthetic value to expose epsilon placement.
        criterion = JointLoss(epsilon=epsilon, lambda_c=.37, align_corners=False)
        logits = torch.zeros(2, 16, 1, 1, 1, dtype=torch.float64)
        label = torch.tensor([0, 1]).reshape(2, 1, 1, 1)
        result = criterion(logits, logits, label)
        for branch in (result.final, result.coarse):
            torch.testing.assert_close(branch.ce, logits.new_full((2,), math.log(8)), rtol=1e-14, atol=1e-14)
            torch.testing.assert_close(branch.dice_per_class[0], logits.new_full((15,), .5), rtol=0, atol=0)
            torch.testing.assert_close(branch.dice_per_class[1, 0], logits.new_tensor(1 / 6))
            torch.testing.assert_close(branch.dice_per_class[1, 1:], logits.new_full((14,), .5))
            torch.testing.assert_close(branch.dice_loss, logits.new_tensor([.5, 47 / 90]))
        torch.testing.assert_close(result.total, logits.new_tensor(1.37 * (math.log(8) + 23 / 45)))

    def test_interpolate_exactly_once_on_logits_and_never_on_GT_or_final(self):
        coarse, final, label = self.inputs()
        old_label = label.clone()
        criterion = JointLoss(epsilon=.1, lambda_c=.37, align_corners=True)
        interpolate = functional.interpolate
        captured = {}
        def resize(*args, **kwargs):
            captured['up'] = interpolate(*args, **kwargs)
            return captured['up']
        with patch('organ_relation.losses.functional.interpolate', side_effect=resize) as resize_spy, \
                patch('organ_relation.losses.torch.softmax', wraps=torch.softmax) as softmax_spy:
            criterion(coarse, final, label)
        self.assertEqual(resize_spy.call_count, 1)
        self.assertIs(resize_spy.call_args.args[0], coarse)
        self.assertEqual(resize_spy.call_args.kwargs, {'size': (2, 3, 3), 'mode': 'trilinear', 'align_corners': True})
        self.assertEqual(softmax_spy.call_count, 2)
        self.assertIs(softmax_spy.call_args_list[0].args[0], captured['up'])
        self.assertIs(softmax_spy.call_args_list[1].args[0], final)
        self.assertTrue(all(call.kwargs == {'dim': 1} for call in softmax_spy.call_args_list))
        torch.testing.assert_close(label, old_label, rtol=0, atol=0)

    def test_counterexample_softmax_then_interpolate_is_wrong(self):
        coarse = torch.zeros(1, 16, 1, 1, 2, dtype=torch.float64)
        coarse[0, 0, 0, 0] = coarse.new_tensor([0., 4.])
        coarse[0, 1, 0, 0] = coarse.new_tensor([4., 0.])
        final = torch.zeros(1, 16, 1, 1, 3, dtype=coarse.dtype)
        label = torch.zeros(1, 1, 1, 3, dtype=torch.long)
        criterion = JointLoss(epsilon=.01, lambda_c=.37, align_corners=True)
        result = criterion(coarse, final, label)
        correct = reference_resize(coarse, (1, 1, 3), True).softmax(1)
        wrong = functional.interpolate(coarse.softmax(1), size=(1, 1, 3), mode='trilinear', align_corners=True)
        expected_correct = math.exp(2) / (2 * math.exp(2) + 14)
        expected_wrong = (1 + math.exp(4)) / (2 * (math.exp(4) + 15))
        torch.testing.assert_close(correct[0, 0, 0, 0, 1], coarse.new_tensor(expected_correct))
        torch.testing.assert_close(wrong[0, 0, 0, 0, 1], coarse.new_tensor(expected_wrong))
        self.assertGreater(abs(expected_correct - expected_wrong), .1)
        correct_ce = -torch.log(correct[:, 0] + .01).flatten(1).mean(1)
        wrong_ce = -torch.log(wrong[:, 0] + .01).flatten(1).mean(1)
        torch.testing.assert_close(result.coarse.ce, correct_ce)
        self.assertGreater((result.coarse.ce - wrong_ce).abs().max().item(), .1)

    def test_both_alignment_settings_against_manual_trilinear_reference(self):
        coarse, final, label = self.inputs()
        for align in (False, True):
            result = JointLoss(epsilon=.013, lambda_c=.37, align_corners=align)(coarse, final, label)
            reference = reference_branch(reference_resize(coarse, final.shape[2:], align), label, .013)
            for actual, expected in zip(result.coarse, reference):
                torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
        # A 2->3 resize samples the same locations for both settings; use 2->4.
        final4 = torch.zeros(2, 16, 2, 4, 4, dtype=coarse.dtype)
        label4 = torch.zeros(2, 2, 4, 4, dtype=torch.long)
        a = JointLoss(epsilon=.013, lambda_c=.37, align_corners=False)(coarse, final4, label4)
        b = JointLoss(epsilon=.013, lambda_c=.37, align_corners=True)(coarse, final4, label4)
        for result, align in ((a, False), (b, True)):
            reference = reference_branch(reference_resize(coarse, (2, 4, 4), align), label4, .013)
            for actual, expected in zip(result.coarse, reference):
                torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
        self.assertFalse(torch.allclose(a.coarse.segmentation, b.coarse.segmentation))

    def test_probability_CE_is_not_standard_logit_CE_at_positive_epsilon(self):
        logits = torch.zeros(1, 16, 1, 1, 2, dtype=torch.float64)
        label = torch.zeros(1, 1, 1, 2, dtype=torch.long)
        result = JointLoss(epsilon=1 / 16, lambda_c=.37, align_corners=False)(logits, logits, label)
        standard = functional.cross_entropy(logits, label)
        torch.testing.assert_close(result.final.ce[0], logits.new_tensor(math.log(8)))
        torch.testing.assert_close(standard, logits.new_tensor(math.log(16)))
        torch.testing.assert_close(standard - result.final.ce[0], logits.new_tensor(math.log(2)))

    def test_background_contributes_to_CE(self):
        logits = torch.zeros(1, 16, 1, 1, 2, dtype=torch.float64)
        label = torch.tensor([0, 15]).reshape(1, 1, 1, 2)
        criterion = JointLoss(epsilon=.01, lambda_c=.37, align_corners=False)
        before = criterion(logits, logits, label).final.ce
        changed = logits.clone()
        changed[0, 0, 0, 0, 0] = 3
        after = criterion(changed, changed, label).final.ce
        self.assertLess(after.item(), before.item())
        target_probability = math.exp(3) / (math.exp(3) + 15)
        expected = -(math.log(target_probability + .01) + math.log(1 / 16 + .01)) / 2
        torch.testing.assert_close(after, logits.new_tensor([expected]))

    def test_dice_excludes_background_but_keeps_all_absent_foreground_classes(self):
        logits = torch.zeros(1, 16, 1, 1, 2, dtype=torch.float64)
        label = torch.zeros(1, 1, 1, 2, dtype=torch.long)
        result = JointLoss(epsilon=.1, lambda_c=.37, align_corners=False)(logits, logits, label)
        self.assertEqual(result.final.dice_per_class.shape, (1, 15))
        absent_dice = .1 / (2 / 16 + .1)
        torch.testing.assert_close(result.final.dice_per_class, logits.new_full((1, 15), absent_dice))
        background_dice = (2 * (2 / 16) + .1) / (2 / 16 + 2 + .1)
        wrong_loss = 1 - (15 * absent_dice + background_dice) / 16
        self.assertGreater(abs(result.final.dice_loss.item() - wrong_loss), .001)

    def test_per_case_macro_dice_differs_from_whole_batch_aggregation(self):
        probabilities = torch.full((2, 16, 1, 1, 4), .005, dtype=torch.float64)
        probabilities[0, 1, 0, 0] = probabilities.new_tensor([.9, .1, .1, .1])
        probabilities[1, 1, 0, 0] = probabilities.new_tensor([.1, .1, .1, .9])
        probabilities[:, 0] = 1 - probabilities[:, 1:].sum(1)
        logits = probabilities.log()
        # Same dense grid, but class-1 organ size is 1 vs 3 voxels and performance differs.
        label = torch.tensor([1, 0, 0, 0, 1, 1, 1, 0]).reshape(2, 1, 1, 4)
        criterion = JointLoss(epsilon=.1, lambda_c=.37, align_corners=False)
        result = criterion(logits, logits, label)
        expected_class1 = logits.new_tensor([(1.8 + .1) / (1.2 + 1 + .1), (.6 + .1) / (1.2 + 3 + .1)])
        torch.testing.assert_close(result.final.dice_per_class[:, 0], expected_class1)
        reference = reference_branch(logits, label, .1)
        torch.testing.assert_close(result.final.dice_loss, reference[2])
        # Deliberately wrong reference: pool all batch voxels before computing Dice.
        global_dices = []
        for c in range(1, 16):
            mask = label == c
            global_dices.append((2 * (probabilities[:, c] * mask).sum() + .1) /
                                (probabilities[:, c].sum() + mask.sum() + .1))
        wrong = 1 - torch.stack(global_dices).mean()
        self.assertGreater(abs(result.final.dice_loss.mean().item() - wrong.item()), .01)
        torch.testing.assert_close(result.total, 1.37 * result.final.segmentation.mean())

    def test_fp64_full_reference_all_statistics(self):
        inputs = self.inputs()
        criterion = JointLoss(epsilon=.017, lambda_c=2.3, align_corners=False)
        result = criterion(*inputs)
        coarse, final, label = inputs
        rc = reference_branch(reference_resize(coarse, final.shape[2:], False), label, .017)
        rf = reference_branch(final, label, .017)
        for actual_branch, reference in ((result.coarse, rc), (result.final, rf)):
            for actual, expected in zip(actual_branch, reference):
                torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
        torch.testing.assert_close(result.per_case, rf[3] + 2.3 * rc[3])
        torch.testing.assert_close(result.total, (rf[3] + 2.3 * rc[3]).mean())

    def test_lambda_is_only_auxiliary_multiplier_and_gradients_scale_correctly(self):
        coarse, final, label = self.inputs(grad=True)
        a = JointLoss(epsilon=.01, lambda_c=.37, align_corners=False)(coarse, final, label)
        b = JointLoss(epsilon=.01, lambda_c=2.3, align_corners=False)(coarse, final, label)
        ga = torch.autograd.grad(a.total, (coarse, final))
        gb = torch.autograd.grad(b.total, (coarse, final))
        torch.testing.assert_close(gb[0], ga[0] * (2.3 / .37))
        torch.testing.assert_close(gb[1], ga[1])
        torch.testing.assert_close(b.total - a.total, (2.3 - .37) * a.coarse.segmentation.mean())
        for g in (*ga, *gb):
            self.assertTrue(torch.isfinite(g).all())
            self.assertGreater(g.abs().sum().item(), 0)

    def test_fp64_numeric_gradients_both_branches(self):
        coarse = torch.randn(1, 16, 1, 1, 2, dtype=torch.float64, requires_grad=True)
        final = torch.randn(1, 16, 1, 1, 3, dtype=torch.float64, requires_grad=True)
        label = torch.tensor([0, 1, 15]).reshape(1, 1, 1, 3)
        criterion = JointLoss(epsilon=.01, lambda_c=.37, align_corners=True)
        self.assertTrue(torch.autograd.gradcheck(lambda c, f: criterion(c, f, label).total, (coarse, final)))

    def test_gradients_against_independent_interpolation_and_branch_reference(self):
        coarse = torch.randn(1, 16, 1, 1, 2, dtype=torch.float64, requires_grad=True)
        final = torch.randn(1, 16, 1, 2, 3, dtype=torch.float64, requires_grad=True)
        label = torch.tensor([0, 1, 2, 15, 1, 0]).reshape(1, 1, 2, 3)
        result = JointLoss(epsilon=.013, lambda_c=.37, align_corners=False)(coarse, final, label)
        reference = (reference_branch(final, label, .013)[3] + .37 * reference_branch(
            reference_resize(coarse, final.shape[2:], False), label, .013)[3]).mean()
        actual_grad = torch.autograd.grad(result.total, (coarse, final))
        reference_grad = torch.autograd.grad(reference, (coarse, final))
        for actual, expected in zip(actual_grad, reference_grad):
            torch.testing.assert_close(actual, expected, rtol=3e-11, atol=3e-12)

    def test_saturated_probabilities_zero_predictions_and_no_clamping(self):
        logits = torch.full((1, 16, 1, 1, 1), -1000., dtype=torch.float64)
        logits[:, 0] = 1000
        label = torch.zeros(1, 1, 1, 1, dtype=torch.long)
        criterion = JointLoss(epsilon=.1, lambda_c=.37, align_corners=False)
        result = criterion(logits, logits, label)
        torch.testing.assert_close(result.final.dice_per_class, torch.ones(1, 15, dtype=logits.dtype))
        self.assertEqual(result.final.dice_loss.item(), 0)
        torch.testing.assert_close(result.final.ce, logits.new_tensor([-math.log(1.1)]))
        self.assertLess(result.total.item(), 0)  # log(S+eps) is not clamped to a nonnegative CE.
        absent_target = torch.ones_like(label)
        wrong_prediction = criterion(logits, logits, absent_target)
        torch.testing.assert_close(wrong_prediction.final.ce, logits.new_tensor([-math.log(.1)]))
        self.assertTrue(torch.isfinite(wrong_prediction.total))

    def test_noncontiguous_inputs_and_case_independence(self):
        coarse, final, label = self.inputs()
        coarse, final, label = coarse.transpose(2, 4), final.transpose(2, 4), label.transpose(1, 3)
        self.assertFalse(final.is_contiguous())
        criterion = JointLoss(epsilon=.01, lambda_c=.37, align_corners=False)
        result = criterion(coarse, final, label)
        individual = [criterion(coarse[b:b+1], final[b:b+1], label[b:b+1]).total for b in range(2)]
        torch.testing.assert_close(result.per_case, torch.stack(individual))
        torch.testing.assert_close(result.total, torch.stack(individual).mean())

    def test_fp32_against_fp64(self):
        coarse, final, label = self.inputs(dtype=torch.float32)
        criterion = JointLoss(epsilon=.001, lambda_c=.37, align_corners=False)
        a = criterion(coarse, final, label)
        b = criterion(coarse.double(), final.double(), label)
        torch.testing.assert_close(a.total.double(), b.total, rtol=3e-6, atol=3e-7)
        for a_branch, b_branch in ((a.coarse, b.coarse), (a.final, b.final)):
            for x, y in zip(a_branch, b_branch):
                torch.testing.assert_close(x.double(), y, rtol=3e-6, atol=3e-7)

    def test_label_no_gradient_or_mutation_and_only_small_output_statistics(self):
        inputs = self.inputs(grad=True)
        copies = tuple(x.detach().clone() for x in inputs)
        criterion = JointLoss(epsilon=.01, lambda_c=.37, align_corners=False)
        result = criterion(*inputs)
        result.total.backward()
        self.assertFalse(inputs[2].requires_grad)
        self.assertIsNone(inputs[2].grad)
        for x, old in zip(inputs, copies):
            torch.testing.assert_close(x, old, rtol=0, atol=0)
        self.assertIsInstance(result, JointLossResult)
        self.assertEqual(result.total.ndim, 0)
        self.assertEqual(result.per_case.shape, (2,))
        for branch in (result.coarse, result.final):
            self.assertIsInstance(branch, BranchLoss)
            self.assertEqual(tuple(t.shape for t in branch), ((2,), (2, 15), (2,), (2,)))
        self.assertEqual(list(criterion.parameters()), [])
        self.assertEqual(list(criterion.buffers()), [])

    def test_segmentor_joint_loss_complete_backward_and_supervision_separation(self):
        config = SegmentorConfig(**json.loads((ROOT / 'configs/segmentor_micro.json').read_text(encoding='utf-8'))['model'])
        model = Segmentor(config)
        # Dedicated differentiability fixture: keep the relation ReLU active.
        with torch.no_grad():
            model.relation.relation_hidden.weight.uniform_(.001, .005)
            model.relation.relation_hidden.bias.fill_(.5)
        image = torch.randn(2, 1, 7, 8, 9, requires_grad=True)
        label = torch.randint(0, 16, (2, 7, 8, 9))
        criterion = JointLoss(epsilon=config.epsilon, lambda_c=.37, align_corners=False)
        output = model(image)
        output.coarse_logits.retain_grad()
        output.final_logits.retain_grad()
        result = criterion(output.coarse_logits, output.final_logits, label)
        result.total.backward()
        for value in (image, output.coarse_logits, output.final_logits):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        for name in ('encoder', 'coarse_head', 'relation', 'node_to_space', 'fusion', 'decoder'):
            magnitude = 0.
            for pname, parameter in getattr(model, name).named_parameters():
                with self.subTest(module=name, parameter=pname):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    magnitude += parameter.grad.abs().sum().item()
            self.assertGreater(magnitude, 0)
        self.assertIsNone(label.grad)
        with self.assertRaises(TypeError):
            model(image, label=label)
        with torch.no_grad():
            other_label = (label + 1) % 16
            other_loss = criterion(output.coarse_logits, output.final_logits, other_label)
            self.assertFalse(torch.allclose(result.total, other_loss.total))
            rerun = model(image)
            for actual, expected in zip(rerun, output):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_final_segmentation_loss_reaches_coarse_via_graph_without_auxiliary(self):
        config = SegmentorConfig(**json.loads((ROOT / 'configs/segmentor_micro.json').read_text(encoding='utf-8'))['model'])
        model = Segmentor(config).double()
        with torch.no_grad():
            model.relation.relation_hidden.weight.uniform_(.001, .005)
            model.relation.relation_hidden.bias.fill_(.5)
        image = torch.randn(1, 1, 7, 8, 9, dtype=torch.float64)
        label = torch.randint(0, 16, (1, 7, 8, 9))
        output = model(image)
        result = JointLoss(epsilon=config.epsilon, lambda_c=.37, align_corners=False)(*output, label)
        targets = (model.coarse_head.projection.weight, model.relation.W_m.weight, model.node_to_space.W_V.weight)
        gradients = torch.autograd.grad(result.final.segmentation.mean(), targets)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_invalid_labels_range_dtype_and_shape(self):
        coarse, final, label = self.inputs()
        criterion = JointLoss(epsilon=.01, lambda_c=.37, align_corners=False)
        for value in (-1, 16, 255):
            invalid = label.clone()
            invalid.flatten()[0] = value
            with self.assertRaisesRegex(ValueError, '0..15'):
                criterion(coarse, final, invalid)
        for invalid in (None, label.float(), label.to(torch.int32), label.bool(), label.unsqueeze(1),
                        label[0], label[:1], label[..., :1], label.to('meta'), label.float().requires_grad_()):
            with self.assertRaises(ValueError):
                criterion(coarse, final, invalid)

    def test_invalid_logits_and_autocast(self):
        coarse, final, label = self.inputs()
        criterion = JointLoss(epsilon=.01, lambda_c=.37, align_corners=False)
        for c, f in ((None, final), (coarse[0], final), (coarse[:, :15], final), (coarse[:1], final),
                     (coarse[:, :, :0], final), (coarse, final[:, :15]), (coarse, final[0]),
                     (coarse.float(), final), (coarse.half(), final.half()), (coarse.long(), final.long()),
                     (coarse.to('meta'), final)):
            with self.assertRaises(ValueError):
                criterion(c, f, label)
        for index in (0, 1):
            for bad in (float('nan'), float('inf'), -float('inf')):
                args = [coarse.clone(), final.clone(), label]
                args[index].flatten()[0] = bad
                with self.assertRaisesRegex(ValueError, 'finite'):
                    criterion(*args)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16), self.assertRaisesRegex(ValueError, 'autocast'):
            criterion(coarse, final, label)

    def test_required_explicit_configuration_without_default_weights_or_smoothing(self):
        signature = inspect.signature(JointLoss)
        self.assertEqual(tuple(signature.parameters), ('epsilon', 'lambda_c', 'align_corners', 'ce_reduction_mode',
                                                      'ce_background_weight', 'ce_foreground_weight', 'foreground_ce_reduction'))
        for name in ('epsilon', 'lambda_c', 'align_corners'):
            self.assertIs(signature.parameters[name].default, inspect.Parameter.empty)
        self.assertEqual(signature.parameters['ce_reduction_mode'].default, 'voxel_mean')
        self.assertEqual(signature.parameters['ce_background_weight'].default, .5)
        self.assertEqual(signature.parameters['ce_foreground_weight'].default, .5)
        self.assertEqual(signature.parameters['foreground_ce_reduction'].default, 'voxel_mean')
        options = dict(epsilon=.01, lambda_c=.37, align_corners=False)
        for name in options:
            missing = options.copy()
            del missing[name]
            with self.assertRaises(TypeError):
                JointLoss(**missing)
        for name in ('epsilon', 'lambda_c'):
            for value in (0, -1, float('nan'), float('inf'), True, '1'):
                with self.assertRaises(ValueError):
                    JointLoss(**{**options, name: value})
        with self.assertRaises(ValueError):
            JointLoss(**{**options, 'align_corners': 0})
        for name in ('epsilon', 'lambda_c'):
            for value in (1e-100, 1e100):
                with self.assertRaises(ValueError):
                    JointLoss(**{**options, name: value})(*self.inputs(dtype=torch.float32))


if __name__ == '__main__':
    unittest.main()

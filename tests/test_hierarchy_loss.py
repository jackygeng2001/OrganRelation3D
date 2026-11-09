"""Hierarchy-consistency revision, eqs. (32)-(33): formula, gradients, scope and logs."""
import copy
import itertools
import unittest

import torch
from torch.nn import functional as F
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import support as fixtures
from test_brain_tumor import Cases, config
from organ_relation.losses import (JointLossResult, RegionJointLoss, RegionLossResult,
                                   hierarchy_violation, region_targets)
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.training.engine import Trainer
from organ_relation.training.state import seed_all


def loss(lambda_hier):
    return RegionJointLoss(epsilon=1e-6, lambda_c=.5, align_corners=False, lambda_hier=lambda_hier)


def looped_hierarchy(logits):
    """Literal eq. (32): explicit loop over cases and voxels."""
    p = logits.sigmoid()
    values = []
    for b in range(p.shape[0]):
        total, count = 0, 0
        for d, h, w in itertools.product(*map(range, p.shape[2:])):
            wt, tc, et = p[b, 0, d, h, w], p[b, 1, d, h, w], p[b, 2, d, h, w]
            total = total + torch.clamp(tc - wt, min=0) + torch.clamp(et - tc, min=0)
            count += 1
        values.append(total / (2 * count))
    return torch.stack(values)


class HierarchyLossTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(5)
        self.label = (torch.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4) % 4).long()
        self.coarse = torch.randn(2, 3, 1, 2, 2, dtype=torch.double, generator=generator).requires_grad_()
        self.final = torch.randn(2, 3, 2, 3, 4, dtype=torch.double, generator=generator).requires_grad_()

    def test_eq32_matches_literal_voxel_loop_and_gradient(self):
        expected = looped_hierarchy(self.final)
        actual = hierarchy_violation(self.final)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(torch.autograd.grad(actual.sum(), self.final, retain_graph=True),
                        torch.autograd.grad(expected.sum(), self.final)):
            torch.testing.assert_close(a, b)

    def test_ordered_probabilities_have_zero_loss_and_zero_gradient(self):
        base = torch.randn(1, 1, 2, 3, 4, dtype=torch.double)
        ordered = torch.cat((base + 2, base + 1, base), 1).requires_grad_()  # WT >= TC >= ET
        value = hierarchy_violation(ordered)
        self.assertEqual(value.item(), 0.0)
        self.assertEqual(torch.autograd.grad(value.sum(), ordered)[0].abs().sum().item(), 0.0)
        reversed_ = torch.cat((base, base + 1, base + 2), 1)  # Both inclusions violated everywhere.
        p = reversed_.sigmoid()
        expected = ((p[:, 1] - p[:, 0]) + (p[:, 2] - p[:, 1])).mean() / 2
        torch.testing.assert_close(hierarchy_violation(reversed_)[0], expected)

    def test_eq33_adds_weighted_final_only_term_to_unchanged_segmentation_loss(self):
        old = loss(None)(self.coarse, self.final, self.label)
        self.assertIs(type(old), JointLossResult)
        for weight in (0, 0.25, 2):
            new = loss(weight)(self.coarse, self.final, self.label)
            self.assertIs(type(new), RegionLossResult)
            torch.testing.assert_close(new.hierarchy, hierarchy_violation(self.final))
            torch.testing.assert_close(new.per_case, old.per_case + weight * new.hierarchy)
            torch.testing.assert_close(new.total, (old.per_case + weight * new.hierarchy).mean())
            for field in ('ce', 'dice_per_class', 'dice_loss', 'segmentation'):
                torch.testing.assert_close(getattr(new.final, field), getattr(old.final, field))
                torch.testing.assert_close(getattr(new.coarse, field), getattr(old.coarse, field))
            # Coarse gradient is untouched by lambda_hier: no hierarchy term on P^c.
            g_old = torch.autograd.grad(old.total, self.coarse, retain_graph=True)[0]
            g_new = torch.autograd.grad(new.total, self.coarse, retain_graph=True)[0]
            torch.testing.assert_close(g_new, g_old)

    def test_none_keeps_previous_objective_bitwise(self):
        criterion = loss(None)
        target = region_targets(self.label).double()
        eps = criterion.epsilon

        def branch(logits):
            s = logits.sigmoid()
            bce = -(target * (s + eps).log() + (1 - target) * (1 - s + eps).log()).mean((1, 2, 3, 4))
            dice = (2 * (s * target).sum((2, 3, 4)) + eps) / (s.sum((2, 3, 4)) + target.sum((2, 3, 4)) + eps)
            return bce + 1 - dice.mean(1)
        aligned = F.interpolate(self.coarse, size=self.label.shape[1:], mode='trilinear', align_corners=False)
        torch.testing.assert_close(criterion(self.coarse, self.final, self.label).total,
                                   (branch(self.final) + .5 * branch(aligned)).mean())

    def test_lambda_hier_validation(self):
        self.assertIsNone(loss(None).lambda_hier)
        self.assertEqual(loss(1).lambda_hier, 1.0)
        for bad in (-0.1, float('nan'), float('inf'), True, '1'):
            with self.assertRaises(ValueError):
                loss(bad)


class HierarchyEngineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cfg = config()
        self.cfg['loss'] = dict(self.cfg['loss'], lambda_hier=1.0)

    def test_train_and_monitor_rows_and_tensorboard_record_hierarchy(self):
        seed_all(171)
        model = BrainTumorSegmentor(BrainTumorConfig(**self.cfg['model']))
        options = dict(self.cfg['training'], max_steps=2, validation_every=1, diagnostics_every=1,
                       checkpoint_every=1, validation_role='train_monitor')
        identity = dict(mode='brats_regions', model=self.cfg['model'], loss=self.cfg['loss'], threshold=.5,
                        training=options, preprocessing=self.cfg['preprocessing'], provenance='test',
                        data={'manifest_hash': 'synthetic'})
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False, fused=False)
        trainer = Trainer(model, RegionJointLoss(**self.cfg['loss']), optimizer, Cases(), ['one', 'two'],
                          device='cpu', options=options, identity=identity, run_dir=self.fixture.root / 'run',
                          validation_dataset=Cases(), validation_case_ids=['one', 'two'], tensorboard=True)
        self.fixture.quiet_run(trainer)
        rows = self.fixture.rows(trainer.run_dir)
        train = [r for r in rows if r['phase'] == 'train']
        monitor = [r for r in rows if r['phase'] == 'train_monitor']
        self.assertEqual(len(train), 2)
        for row in train:
            self.assertGreaterEqual(row['hierarchy'], 0)
            self.assertAlmostEqual(row['total_loss'], row['final']['segmentation']
                                   + .5 * row['coarse']['segmentation'] + row['hierarchy'], places=5)
        self.assertTrue(monitor and all('hierarchy' in c for r in monitor
                                        for c in r['region_diagnostic_cases'].values()))
        tags = EventAccumulator(str(trainer.board.directory), size_guidance={'scalars': 0}).Reload().Tags()['scalars']
        self.assertIn('Train/final/hierarchy', tags)
        self.assertIn('Monitor/final/hierarchy', tags)


if __name__ == '__main__':
    unittest.main()

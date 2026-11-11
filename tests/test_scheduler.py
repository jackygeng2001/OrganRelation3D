"""学习率调度：缺省不变；余弦调度无状态、逐步写入日志，中途续训与一口气跑完逐值相同。"""
import math
import unittest

import torch

import support
from organ_relation.training.engine import scheduled_lr, validate_scheduler
from organ_relation.training.state import load_checkpoint

COSINE = {'kind': 'cosine', 'min_lr': 1e-6}


class SchedulerTests(support.TrainingTests):
    def test_schedule_values_and_validation(self):
        self.assertEqual(scheduled_lr(None, 3e-4, 5, 10), 3e-4)
        self.assertEqual(scheduled_lr(COSINE, 3e-4, 1, 10), 3e-4)
        self.assertAlmostEqual(scheduled_lr(COSINE, 3e-4, 6, 10), 1e-6 + (3e-4 - 1e-6) * .5, places=12)
        last = scheduled_lr(COSINE, 3e-4, 10, 10)
        self.assertAlmostEqual(last, 1e-6 + (3e-4 - 1e-6) * .5 * (1 + math.cos(math.pi * .9)), places=12)
        values = [scheduled_lr(COSINE, 3e-4, step, 10) for step in range(1, 11)]
        self.assertEqual(values, sorted(values, reverse=True))
        for bad in ('cosine', {'kind': 'step', 'min_lr': 0}, {'kind': 'cosine'}, {'kind': 'cosine', 'min_lr': -1},
                    {'kind': 'cosine', 'min_lr': True}, {'kind': 'cosine', 'min_lr': 0, 'warmup': 1}):
            with self.assertRaises(ValueError):
                validate_scheduler(bad)
        with self.assertRaises(ValueError):
            scheduled_lr(COSINE, 3e-4, 11, 10)
        with self.assertRaises(ValueError):
            scheduled_lr({'kind': 'cosine', 'min_lr': 1e-3}, 3e-4, 1, 10)

    def test_logged_learning_rate_follows_schedule_and_default_is_constant(self):
        constant = self.root / 'constant'
        self.quiet_run(self.trainer(constant))
        self.assertEqual({r['lr'] for r in self.rows(constant) if r['phase'] == 'train'}, {3e-4})
        scheduled = self.root / 'scheduled'
        self.quiet_run(self.trainer(scheduled, options=dict(scheduler=COSINE)))
        rates = [r['lr'] for r in self.rows(scheduled) if r['phase'] == 'train']
        self.assertEqual(rates, [scheduled_lr(COSINE, 3e-4, step, 7) for step in range(1, 8)])

    def test_resume_reproduces_uninterrupted_scheduled_run(self):
        options = dict(scheduler=COSINE)
        whole, parts = self.root / 'whole', self.root / 'parts'
        first = self.trainer(whole, options=options)
        self.quiet_run(first)
        self.quiet_run(self.trainer(parts, options=options), stop_after=3)
        resumed = self.trainer(parts, options=options, resume=True)
        self.quiet_run(resumed)
        strip = lambda rows: [{k: v for k, v in r.items() if k not in ('run_id', 'step_seconds', 'progress', 'memory')}
                              for r in rows]
        self.assert_nested_equal(strip(self.rows(whole)), strip(self.rows(parts)))
        a = load_checkpoint(whole / 'last.ckpt', first.identity)
        b = load_checkpoint(parts / 'last.ckpt', resumed.identity)
        for key in ('model', 'optimizer', 'progress', 'rng'):
            self.assert_nested_equal(a[key], b[key])

    def test_scheduled_run_cannot_be_extended(self):
        directory = self.root / 'run'
        self.quiet_run(self.trainer(directory, options=dict(scheduler=COSINE)))
        with self.assertRaises(ValueError):
            self.trainer(directory, options=dict(scheduler=COSINE), resume=True, extend_to=9)


if __name__ == '__main__':
    unittest.main()

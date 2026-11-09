"""后台预读下一例：与现读逐位相同（日志、权重、优化器、随机数、早停、续跑），只用于纯函数数据集，后台读取错误在主线程原样抛出。"""
import json
import threading
import unittest

import numpy as np
import torch

import support as fixtures
from organ_relation.data.full_scan import FullScanSample
from organ_relation.losses import RegionJointLoss
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.training.engine import NextCase, Trainer, prefetch_safe
from organ_relation.training.state import load_checkpoint, seed_all

STATE_KEYS = ('model', 'optimizer', 'rng', 'progress', 'sampler_generator', 'loader_generator',
              'development_state', 'early_stopping')
VOLATILE = ('run_id', 'step_seconds', 'progress')  # 运行编号与计时，两次运行本来就不同


class PureCases(torch.utils.data.Dataset):
    """确定性、不消耗任何随机数的小体积病例；记录每次读取发生在哪个线程。"""
    prefetch_safe = True

    def __init__(self, count, *, offset=0, fail_at=None):
        self.count, self.offset, self.fail_at, self.loads = count, offset, fail_at, []

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        self.loads.append((index, threading.current_thread().name))
        if index == self.fail_at:
            raise OSError(f'unreadable case {index}')
        shape = (4 + (index + self.offset) % 3, 6, 7)
        grid = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)
        image = torch.stack([torch.sin(grid * (c + 1 + self.offset) / 17) for c in range(4)])
        label = ((grid.long() * (index + 3)) // 11) % 4
        return FullScanSample(image, label, {})


class PrefetchTests(fixtures.TrainingTests):
    def setUp(self):
        super().setUp()
        self.cfg = json.loads((fixtures.ROOT / 'configs/development/year_fold_bbox_fp32.json').read_text())

    def trainer(self, path, *, prefetch, resume=False, train=None, dev=None, epochs=3):
        """按轮训练 + 轻量增强 + 每轮验证 + 早停，与正式五折协议的结构相同。"""
        seed_all(97)
        model = BrainTumorSegmentor(BrainTumorConfig(**self.cfg['model'])).to(memory_format=torch.channels_last_3d)
        options = dict(self.cfg['training'], max_epochs=epochs, checkpoint_every_steps=1,
                       early_stopping=dict(min_epochs=2, patience_epochs=1, min_delta=.001,
                                           monitor='dev_mean_region_hard_dice'))
        identity = dict(mode='brats_regions', training=options, model=self.cfg['model'], loss=self.cfg['loss'],
                        threshold=.5, preprocessing=self.cfg['preprocessing'], data=dict(manifest_hash='synthetic'),
                        provenance='CPU prefetch test')
        train = train if train is not None else PureCases(3)
        dev = dev if dev is not None else PureCases(2, offset=5)
        return Trainer(model, RegionJointLoss(**self.cfg['loss']),
                       torch.optim.AdamW(model.parameters(), **self.cfg['optimizer']),
                       train, ['a', 'b', 'c'][:len(train)], device='cpu', options=options, identity=identity,
                       run_dir=path, validation_dataset=dev, validation_case_ids=['v1', 'v2'][:len(dev)],
                       resume=path / 'last.ckpt' if resume else None, prefetch=prefetch)

    def stable_rows(self, path):
        return [{k: v for k, v in row.items() if k not in VOLATILE} for row in self.rows(path)]

    def assert_same_run(self, a, b, *, best=True):
        self.assertEqual(self.stable_rows(a.run_dir), self.stable_rows(b.run_dir))
        names = ('last.ckpt', 'best-dev.ckpt') if best else ('last.ckpt',)
        for name in names:
            x = load_checkpoint(a.run_dir / name, a.identity)
            y = load_checkpoint(b.run_dir / name, b.identity)
            for key in STATE_KEYS:
                self.assert_nested_equal(x[key], y[key])

    def test_prefetch_matches_synchronous_loading_bitwise(self):
        plain = self.trainer(self.root / 'plain', prefetch=False)
        ahead = self.trainer(self.root / 'ahead', prefetch=True)
        self.quiet_run(plain); self.quiet_run(ahead)
        self.assert_same_run(plain, ahead)
        rows = self.rows(ahead.run_dir)
        self.assertGreaterEqual(len([r for r in rows if r['phase'] == 'train']), 6)
        self.assertTrue(any('augmentation' in r for r in rows))
        for name in ('dataset', 'validation_dataset'):
            loads = [getattr(t, name).loads for t in (plain, ahead)]
            # 每一例只读一次：预读不重复读盘，也不多读
            self.assertEqual(sorted(i for i, _ in loads[0]), sorted(i for i, _ in loads[1]))
            self.assertTrue(all(thread == 'MainThread' for _, thread in loads[0]))
            self.assertTrue(any(thread.startswith('next-case') for _, thread in loads[1]), name)

    def test_resume_with_prefetch_matches_uninterrupted(self):
        whole = self.trainer(self.root / 'whole', prefetch=True)
        self.quiet_run(whole)
        for stop in (2, 3, 4):  # 轮中、轮末（验证前）、下一轮中
            with self.subTest(stop=stop):
                part = self.trainer(self.root / f'part{stop}', prefetch=True)
                self.quiet_run(part, stop_after=stop)
                resumed = self.trainer(part.run_dir, prefetch=True, resume=True)
                self.quiet_run(resumed)
                for key in STATE_KEYS:
                    self.assert_nested_equal(load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)[key],
                                             load_checkpoint(part.run_dir / 'last.ckpt', resumed.identity)[key])

    def test_background_failure_is_raised_when_that_case_is_needed(self):
        outcomes = []
        for prefetch in (False, True):
            trainer = self.trainer(self.root / f'fail{prefetch}', prefetch=prefetch, train=PureCases(3, fail_at=1))
            with self.assertRaisesRegex(OSError, 'unreadable case 1'):
                self.quiet_run(trainer)
            path = trainer.run_dir / 'last.ckpt'
            progress = load_checkpoint(path, trainer.identity)['progress'] if path.exists() else None
            outcomes.append((progress, self.stable_rows(trainer.run_dir)))
        self.assertEqual(outcomes[0], outcomes[1])  # 出错前已提交的进度与日志完全相同

    def test_only_pure_datasets_may_be_prefetched(self):
        self.assertTrue(prefetch_safe(PureCases(1)))
        self.assertFalse(prefetch_safe(fixtures.RandomCases()))  # 读取时消耗随机数
        with self.assertRaisesRegex(ValueError, 'consumes no random numbers'):
            self.trainer(self.root / 'random', prefetch=True, train=fixtures.RandomCases())
        with self.assertRaisesRegex(ValueError, 'consumes no random numbers'):
            self.trainer(self.root / 'random_dev', prefetch=True, dev=fixtures.RandomCases())
        with self.assertRaisesRegex(ValueError, 'boolean'):
            self.trainer(self.root / 'flag', prefetch=1)

    def test_next_case_slot_hits_misses_and_reopens(self):
        data, other = PureCases(3), PureCases(3, offset=1)
        slot = NextCase(pin_memory=False)
        self.addCleanup(slot.close)
        slot.submit(data, 1)
        self.assertIsNone(slot.take(data, 2))          # 下标不同：丢弃，交给调用方现读
        slot.submit(data, 1)
        self.assertIsNone(slot.take(other, 1))         # 数据集不同：同样丢弃
        slot.submit(data, 2)
        sample = slot.take(data, 2)
        self.assertTrue(torch.equal(sample.image, data[2].image))
        self.assertIsNone(slot.take(data, 2))          # 只能取一次
        slot.close()
        slot.submit(data, 0)                           # close 之后还能再用（同一 Trainer 多次 run）
        self.assertTrue(torch.equal(slot.take(data, 0).label, data[0].label))
        failing = PureCases(2, fail_at=0)
        slot.submit(failing, 0)
        with self.assertRaisesRegex(OSError, 'unreadable case 0'):
            slot.take(failing, 0)


if __name__ == '__main__':
    unittest.main()

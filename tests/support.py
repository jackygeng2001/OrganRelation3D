"""测试公用工具：临时目录、静默运行、嵌套比较，以及一个小型 BraTS 训练器工厂。

不是测试文件（文件名不以 test_ 开头），只被各测试模块导入。
"""
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from organ_relation.data.full_scan import FullScanSample
from organ_relation.losses import RegionJointLoss
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.training.engine import Trainer
from organ_relation.training.state import seed_all


def config():
    """实现检查配置（小模型宽度），只用于 CPU 测试。"""
    return json.loads((ROOT / 'configs/brats_regions.json').read_text())


def training_options(**changes):
    options = dict(config()['training'], max_steps=10, checkpoint_every=1, diagnostics_every=1,
                   validation_every=5, validation_role='train_monitor')
    options.update(changes)
    return options


class RandomCases(torch.utils.data.Dataset):
    """三例不同尺寸的四模态体积；读取时消耗 Python/NumPy/torch 三个随机流，用来暴露续训时随机状态恢复错误。"""

    def __len__(self):
        return 3

    def __getitem__(self, index):
        shape = (3 + index, 5, 6)
        image = torch.rand(4, *shape) + random.random() + float(np.random.random())
        label = torch.arange(int(np.prod(shape))).reshape(shape).long() % 4
        return FullScanSample(image, label, {})


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def trainer(self, directory, *, resume=False, validation=False, options=None, identity=None, extend_to=None,
                console=None, tensorboard=False):
        """默认 7 步、三例训练；validation=True 时每 2 步在同三例上监测一次。"""
        seed_all(712)
        cfg = config()
        model = BrainTumorSegmentor(BrainTumorConfig(**cfg['model'])).to(memory_format=torch.channels_last_3d)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0, foreach=False, fused=False)
        opts = training_options(max_steps=7, validation_every=2 if validation else 0)
        if options:
            opts.update(options)
        ident = identity or dict(mode='brats_regions', model=cfg['model'], loss=cfg['loss'], threshold=.5,
                                 training=opts, optimizer='AdamW', preprocessing=cfg['preprocessing'],
                                 provenance='test', data={'manifest_hash': 'test'})
        dataset = RandomCases()
        return Trainer(model, RegionJointLoss(**cfg['loss']), optimizer, dataset, ['one', 'two', 'three'],
                       device='cpu', options=opts, identity=ident, run_dir=directory,
                       validation_dataset=dataset if validation else None,
                       validation_case_ids=['one', 'two', 'three'] if validation else [],
                       resume=directory / 'last.ckpt' if resume else None, extend_to=extend_to,
                       console=console, tensorboard=tensorboard)

    def quiet_run(self, trainer, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return trainer.run(**kwargs)

    def assert_nested_equal(self, a, b):
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assert_nested_equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def rows(self, directory):
        return [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]

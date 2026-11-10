"""Run a declared CPU test suite; missing dependencies or skipped tests fail."""
import argparse
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    # 模型、损失与数值执行：公式、梯度、混合精度、消融变体
    'model': ('brain_tumor', 'coarse_nodes', 'dynamic_relation', 'node_to_space', 'hierarchy_loss',
              'depth_sliced_conv', 'conv_precision', 'mixed_precision', 'ablation'),
    # 数据：读取、裁剪、缓存、五折划分、增强
    'data': ('brats_crop', 'brats_cache', 'brats_folds', 'brats_pilot', 'augmentation'),
    # 训练引擎：续训、checkpoint、观测层、epoch 协议、入口
    'training': ('training', 'monitoring', 'epoch_protocol', 'training_cadence', 'auto_resume',
                 'brats_overfit', 'brats_diagnose', 'brats_relation_probe', 'run_experiment', 'prefetch'),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=(*SUITES, 'all'), required=True)
    args = parser.parse_args(argv)
    # Do not let a new test file silently disappear from the declared suites.
    declared = {f'test_{name}.py' for names in SUITES.values() for name in names}
    discovered = {p.name for p in (ROOT / 'tests').glob('test_*.py')}
    if declared != discovered:
        parser.error(f'update SUITES to match test files: {sorted(declared ^ discovered)}')
    names = sum(SUITES.values(), ()) if args.suite == 'all' else SUITES[args.suite]
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for name in names:
        suite.addTests(loader.discover(str(ROOT / 'tests'), pattern=f'test_{name}.py'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print('Incomplete validation: skipped tests are not a pass. Use the matching CPU environment.',
              file=sys.stderr)
    return 0 if result.wasSuccessful() and not result.skipped and result.testsRun else 1


if __name__ == '__main__':
    raise SystemExit(main())

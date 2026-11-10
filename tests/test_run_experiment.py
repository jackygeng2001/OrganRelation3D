"""五折运行入口：本机配置校验、每折命令、失败即停与同一命令续跑（不启动真实训练）。"""
import contextlib
import importlib.util
import io
import json
import unittest
from types import SimpleNamespace

import support

spec = importlib.util.spec_from_file_location('run_experiment', support.ROOT / 'scripts/run_experiment.py')
runner_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner_module)


class RunExperimentTests(support.TrainingTests):
    def machine(self, **changes):
        root = self.root
        (root / 'brats.zip').write_bytes(b'zip')
        for fold in range(5):
            (root / f'fold_{fold}.local.json').write_text('{}')
        machine = dict(device='cuda:0', convolution_backend='depth_sliced', cohort='brats2018',
                       source_partition='training', source=str(root / 'brats.zip'), data_cache=str(root / 'cache'),
                       split_template=str(root / 'fold_{fold}.local.json'), runs_dir=str(root / 'runs'),
                       environment={'PYTORCH_MIOPEN_SUGGEST_NHWC': '1'})
        machine.update(changes)
        path = root / 'machine.local.json'
        path.write_text(json.dumps(machine))
        return path

    def invoke(self, argv, runner=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = runner_module.main(argv, runner=runner or (lambda *a, **k: self.fail('must not run')))
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_prints_one_auto_resume_command_per_fold(self):
        config = str(support.ROOT / 'configs/experiments/r2_nodes.json')
        code, out, _ = self.invoke([config, '--machine', str(self.machine()), '--dry-run'])
        self.assertEqual(code, 0)
        for fold in range(5):
            self.assertIn(f'brats2018_r2_nodes_fold{fold}', out)
            self.assertIn(f'fold_{fold}.local.json', out)
        self.assertEqual(out.count('--auto-resume'), 5)
        self.assertEqual(out.count('PYTORCH_MIOPEN_SUGGEST_NHWC=1'), 5)

    def test_runs_each_fold_with_explicit_environment_and_stops_at_first_failure(self):
        calls = []
        def fake(command, *, cwd, env):
            calls.append((command, cwd, env))
            return SimpleNamespace(returncode=3 if len(calls) == 3 else 0)
        config = str(support.ROOT / 'configs/experiments/r4_full.json')
        code, _, err = self.invoke([config, '--machine', str(self.machine())], runner=fake)
        self.assertEqual(code, 3)
        self.assertEqual(len(calls), 3)  # 第三折失败后不再启动第四、五折
        self.assertIn('rerun the same command', err)
        for fold, (command, cwd, env) in enumerate(calls):
            self.assertEqual(cwd, support.ROOT)
            self.assertEqual(env['PYTORCH_MIOPEN_SUGGEST_NHWC'], '1')
            self.assertEqual(command[2:4], ['train', '--config'])
            self.assertEqual(command[-3:], ['--run-dir', str(self.root / 'runs' / f'brats2018_r4_full_fold{fold}'), '--auto-resume'])
            self.assertNotIn('--resume', command)

    def test_selected_folds_and_relative_paths(self):
        path = self.machine(source='datasets/x.zip')
        code, _, err = self.invoke([str(support.ROOT / 'configs/experiments/r4_full.json'), '--machine', str(path), '--dry-run'])
        self.assertEqual(code, 2)
        self.assertIn(str(support.ROOT / 'datasets/x.zip'), err)  # 相对路径按仓库根目录解析
        code, out, _ = self.invoke([str(support.ROOT / 'configs/experiments/r4_full.json'), '--machine', str(self.machine()),
                                 '--folds', '2', '4', '--dry-run'])
        self.assertEqual(code, 0)
        self.assertEqual(out.count('[fold '), 2)
        self.assertIn('r4_full_fold2', out); self.assertIn('r4_full_fold4', out)

    def test_invalid_inputs_are_refused_before_training(self):
        config = str(support.ROOT / 'configs/experiments/r4_full.json')
        machine = lambda **changes: str(self.machine(**changes))  # 每个用例在循环内单独写入本机配置
        cases = [
            (lambda: [config, '--machine', str(self.root / 'absent.json')], 'machine.example.json'),
            (lambda: ['configs/experiments/absent.json', '--machine', machine()], 'not found'),
            (lambda: [config, '--machine', machine(extra='x')], 'keys must be exactly'),
            (lambda: [config, '--machine', machine(split_template=str(self.root / 'fold.json'))], '{fold}'),
            (lambda: [config, '--machine', machine(environment={'X': 1})], 'environment'),
            (lambda: [config, '--machine', machine(split_template=str(self.root / 'nope_{fold}.json'))], 'missing input'),
            (lambda: [config, '--machine', machine(), '--folds', '1', '1'], 'distinct'),
        ]
        for argv, message in cases:
            with self.subTest(message=message):
                code, _, err = self.invoke(argv())
                self.assertEqual(code, 2)
                self.assertIn(message, err)


if __name__ == '__main__':
    unittest.main()

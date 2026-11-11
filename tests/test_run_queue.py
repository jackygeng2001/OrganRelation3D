"""并行队列入口：顺序、并行数、失败重试后继续、状态文件续跑（不启动真实训练）。"""
import contextlib
import importlib.util
import io
import json
import sys
import unittest

import support

sys.path.insert(0, str(support.ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('run_queue', support.ROOT / 'scripts/run_queue.py')
queue_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(queue_module)


class FakeProcess:
    def __init__(self, code, polls):
        self.code, self.polls, self.pid = code, polls, 0

    def poll(self):
        self.polls -= 1
        return self.code if self.polls <= 0 else None


class RunQueueTests(support.TrainingTests):
    def machine(self):
        root = self.root
        (root / 'brats.zip').write_bytes(b'zip')
        for fold in range(5):
            (root / f'fold_{fold}.local.json').write_text('{}')
        path = root / 'machine.local.json'
        path.write_text(json.dumps(dict(
            device='cuda:0', convolution_backend='depth_sliced', cohort='brats2018', source_partition='training',
            source=str(root / 'brats.zip'), data_cache=str(root / 'cache'),
            split_template=str(root / 'fold_{fold}.local.json'), runs_dir=str(root / 'runs'), environment={'X': '1'})))
        return path

    def queue(self, jobs, workers=2):
        path = self.root / 'queue.json'
        path.write_text(json.dumps(dict(workers=workers, jobs=jobs)))
        return path

    def invoke(self, argv, codes=None):
        """codes：运行目录名 -> 依次返回的退出码列表；记录每次启动时已在运行的作业数。"""
        launches, active = [], []

        def popen(command, *, cwd, env, stdout, stderr, start_new_session):
            name = command[command.index('--run-dir') + 1].rsplit('/', 1)[-1]
            active[:] = [p for p in active if p.polls > 0]
            launches.append((name, len(active), env['X'], command[-1]))
            sequence = (codes or {}).get(name, [0])
            process = FakeProcess(sequence[min(sum(1 for l in launches if l[0] == name) - 1, len(sequence) - 1)], 2)
            active.append(process)
            return process
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = queue_module.main(argv, popen=popen, sleep=lambda seconds: None)
        return code, launches, out.getvalue()

    def jobs(self, *pairs):
        return [dict(config=f'configs/experiments/{name}.json', fold=fold) for name, fold in pairs]

    def test_order_parallelism_and_status_resume(self):
        queue = self.queue(self.jobs(('r2_nodes', 0), ('r4_full', 0), ('r2_nodes', 1), ('r4_full', 1)))
        argv = [str(queue), '--machine', str(self.machine())]
        code, launches, _ = self.invoke(argv)
        self.assertEqual(code, 0)
        self.assertEqual([l[0] for l in launches], ['brats2018_r2_nodes_fold0', 'brats2018_r4_full_fold0',
                                                    'brats2018_r2_nodes_fold1', 'brats2018_r4_full_fold1'])
        self.assertLessEqual(max(l[1] for l in launches), 1)  # 启动时至多已有一个在跑：同时不超过两个
        self.assertEqual({l[2:] for l in launches}, {('1', '--auto-resume')})
        status = json.loads((self.root / 'runs' / 'queue_status' / 'queue.json').read_text())
        self.assertEqual({v['state'] for v in status.values()}, {'done'})
        code, launches, out = self.invoke(argv)  # 全部完成后再次执行：什么都不启动
        self.assertEqual((code, launches), (0, []))
        self.assertIn('4 already done', out)

    def test_failed_job_is_retried_once_then_skipped_and_retried_on_rerun(self):
        queue = self.queue(self.jobs(('r2_nodes', 0), ('r4_full', 0), ('r2_nodes', 1)), workers=1)
        argv = [str(queue), '--machine', str(self.machine())]
        code, launches, out = self.invoke(argv, codes={'brats2018_r4_full_fold0': [2, 2, 2], 'brats2018_r2_nodes_fold0': [3, 0]})
        self.assertEqual(code, 1)
        self.assertEqual([l[0] for l in launches], ['brats2018_r2_nodes_fold0', 'brats2018_r2_nodes_fold0',
                                                    'brats2018_r4_full_fold0', 'brats2018_r4_full_fold0',
                                                    'brats2018_r2_nodes_fold1', 'brats2018_r4_full_fold0'])  # 队尾再试一次
        self.assertIn('FAILED brats2018_r4_full_fold0', out)
        code, launches, _ = self.invoke(argv)  # 再次执行只重跑失败的那个
        self.assertEqual((code, [l[0] for l in launches]), (0, ['brats2018_r4_full_fold0']))

    def test_invalid_queue_rejected_before_anything_runs(self):
        machine = str(self.machine())
        for jobs in ([], self.jobs(('r2_nodes', 0), ('r2_nodes', 0)), [dict(config='configs/experiments/none.json', fold=0)],
                     [dict(config='configs/experiments/r2_nodes.json', fold=-1)], self.jobs(('r2_nodes', 7))):
            code, launches, _ = self.invoke([str(self.queue(jobs)), '--machine', machine])
            self.assertEqual((code, launches), (2, []))
        code, launches, out = self.invoke([str(self.queue(self.jobs(('r2_nodes', 0)))), '--machine', machine, '--dry-run'])
        self.assertEqual((code, launches), (0, []))
        self.assertIn('brats2018_r2_nodes_fold0', out)


if __name__ == '__main__':
    unittest.main()

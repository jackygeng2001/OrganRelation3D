"""CPU-only synthetic end-to-end probe and reporting failure-path tests."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from test_full_scan import ROOT, configuration, synthetic_pair

spec = importlib.util.spec_from_file_location('validate_full_scan', ROOT / 'scripts/validate_full_scan.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'; self.data.mkdir()
        synthetic_pair(self.data)
        self.old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, self.old_threads)

    def arguments(self, stage, suffix='run'):
        return ['--data-root', str(self.data), '--case', 'amos_0001', '--candidate', 'A',
                '--backend', 'cpu', '--device', 'cpu', '--cpu-synthetic', '--through', stage,
                '--model-config', str(ROOT / 'configs/segmentor_micro.json'), '--allow-micro-model',
                '--output', str(self.root / (suffix+'.json'))]

    def run_probe(self, arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = probe.main(arguments)
        return code, json.loads(Path(arguments[arguments.index('--output')+1]).read_text(encoding='utf-8'))

    def test_cli_explicit_capacity_and_optimizer_arguments(self):
        args = probe.parser().parse_args(self.arguments('step'))
        with self.assertRaisesRegex(ValueError, 'optimizer'):
            probe.check_arguments(args)
        args = probe.parser().parse_args(self.arguments('step')+['--optimizer', 'adamw', '--lr', '0.0001'])
        probe.check_arguments(args)
        self.assertEqual(args.betas, (0.9, 0.999))

    def test_each_prefix_stops_after_requested_stage(self):
        for stage in probe.STAGES[:-1]:
            with self.subTest(stage=stage):
                code, report = self.run_probe(self.arguments(stage, stage))
                self.assertEqual(code, 0, report)
                expected = list(probe.STAGES[:probe.STAGES.index(stage)+1])
                actual = [s['stage'] for s in report['stages'] if s['stage'] in probe.STAGES]
                self.assertEqual(actual, expected)
                self.assertEqual(report['shape_dhw'], [4, 6, 8])
                self.assertFalse(report['oom'])
                for record in report['stages']:
                    self.assertEqual(record['status'], 'passed')
                    self.assertGreaterEqual(record['seconds'], 0.)
                    self.assertIsNone(record['peak_allocated_bytes'])
                    self.assertIsNone(record['device_used_bytes'])

    def test_full_real_file_bridge_model_loss_backward_adamw_step(self):
        code, report = self.run_probe(self.arguments('step')+['--optimizer', 'adamw', '--lr', '0.0001'])
        self.assertEqual(code, 0, report)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['output_shapes']['final_logits'], [1, 16, 4, 6, 8])
        self.assertEqual(report['gradients']['missing_gradients'], [])
        self.assertEqual(report['gradients']['nonfinite_gradients'], [])
        self.assertGreater(report['optimizer_state_bytes'], 0)
        self.assertEqual(report['optimizer']['options']['foreach'], False)
        self.assertEqual(report['config']['epsilon'], 1e-6)
        self.assertEqual(report['config']['loss'], {'lambda_c': .5, 'align_corners': False})
        self.assertIn('commit', report['git'])
        self.assertIn('manifest_sha256', report)
        self.assertIsNone(report['peak_reserved_bytes'])

    def test_probe_batching_matches_dataloader_outputs_loss_and_all_gradients(self):
        from torch.utils.data import DataLoader
        from organ_relation.data.full_scan import FullScanDataset, FullScanPreprocessor, ScanPair
        from organ_relation.models.segmentor import Segmentor
        from organ_relation.losses import JointLoss

        pair = ScanPair('amos_0001', self.data / 'imagesTr/amos_0001.nii.gz',
                        self.data / 'labelsTr/amos_0001.nii.gz')
        original_forward = Segmentor.forward
        for candidate in ('A', 'B', 'C'):
            with self.subTest(candidate=candidate):
                captured = {}
                def forward(model, image):
                    captured['model'] = model
                    captured['reference_model'] = copy.deepcopy(model)
                    captured['image'] = image.detach().clone()
                    output = original_forward(model, image)
                    captured['output'] = tuple(t.detach().clone() for t in output)
                    return output
                args = self.arguments('backward', candidate)
                args[args.index('--candidate')+1] = candidate
                with patch.object(Segmentor, 'forward', forward):
                    code, report = self.run_probe(args)
                self.assertEqual(code, 0, report)
                dataset = FullScanDataset([pair], FullScanPreprocessor(configuration(), candidate))
                batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
                self.assertTrue(torch.equal(captured['image'], batch.image))
                reference = captured['reference_model']
                output = reference(batch.image)
                for actual, expected in zip(captured['output'], output):
                    self.assertTrue(torch.equal(actual, expected))
                loss = JointLoss(epsilon=1e-6, lambda_c=.5, align_corners=False)(
                    output.coarse_logits, output.final_logits, batch.label)
                self.assertEqual(report['loss_values']['total'], loss.total.item())
                loss.total.backward()
                actual_parameters = dict(captured['model'].named_parameters())
                for name, parameter in reference.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.equal(actual_parameters[name].grad, parameter.grad), name)

    def test_sgd_step_is_called_exactly_once(self):
        original = torch.optim.SGD.step
        calls = []
        def step(optimizer, *args, **kwargs):
            calls.append(1)
            before = [p.detach().clone() for group in optimizer.param_groups for p in group['params']]
            value = original(optimizer, *args, **kwargs)
            after = [p for group in optimizer.param_groups for p in group['params']]
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))
            return value
        with patch.object(torch.optim.SGD, 'step', step):
            code, report = self.run_probe(self.arguments('step')+['--optimizer', 'sgd', '--lr', '0.0001'])
        self.assertEqual(code, 0, report)
        self.assertEqual(calls, [1])

    def test_capacity_not_invented_and_micro_requires_explicit_opt_in(self):
        args = self.arguments('forward')
        index = args.index('--model-config'); del args[index:index+2]
        code, report = self.run_probe(args)
        self.assertEqual(code, 2)
        self.assertIn('capacity is not frozen', report['error'])
        args = self.arguments('forward', 'micro'); args.remove('--allow-micro-model')
        code, report = self.run_probe(args)
        self.assertEqual(code, 2)
        self.assertIn('micro config requires', report['error'])

    def test_baseline_epsilon_mismatch_rejected(self):
        config = json.loads((ROOT / 'configs/segmentor_micro.json').read_text())
        config['model']['epsilon'] = 1e-4
        path = self.root / 'wrong.json'; path.write_text(json.dumps(config))
        args = self.arguments('forward'); args[args.index('--model-config')+1] = str(path)
        code, report = self.run_probe(args)
        self.assertEqual(code, 2)
        self.assertIn('same baseline epsilon', report['error'])

    def test_oom_retains_prior_stages_and_does_not_retry_or_shrink(self):
        from organ_relation.models.segmentor import Segmentor
        with patch.object(Segmentor, 'forward', side_effect=torch.OutOfMemoryError('synthetic OOM')) as forward:
            code, report = self.run_probe(self.arguments('backward'))
        self.assertEqual(code, 2)
        self.assertTrue(report['oom'])
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(report['stages'][-1]['stage'], 'forward')
        self.assertEqual(report['stages'][-1]['status'], 'failed')
        self.assertEqual(report['shape_dhw'], [4, 6, 8])
        self.assertEqual(report['stages'][0]['status'], 'passed')

    def test_nonfinite_failure_is_not_misreported_as_oom(self):
        from organ_relation.models.segmentor import Segmentor
        with patch.object(Segmentor, 'forward', side_effect=ValueError('nonfinite')):
            code, report = self.run_probe(self.arguments('forward'))
        self.assertEqual(code, 2)
        self.assertFalse(report['oom'])

    def test_output_protection_and_cpu_guard(self):
        args = self.arguments('preprocess'); args[args.index('--output')+1] = str(self.data / 'forbidden.json')
        with self.assertRaises(ValueError):
            probe.main(args)
        self.assertFalse((self.data / 'forbidden.json').exists())
        args = self.arguments('preprocess'); args.remove('--cpu-synthetic')
        code, report = self.run_probe(args)
        self.assertEqual(code, 2)
        with self.assertRaises(ValueError):
            probe.main(args)

    def test_no_rocm_fallback_on_cpu_host(self):
        args = self.arguments('preprocess')
        args[args.index('--backend')+1] = 'rocm'; args[args.index('--device')+1] = 'cuda:0'
        args.remove('--cpu-synthetic')
        with patch.object(torch.cuda, 'is_available', return_value=False), patch.object(torch.cuda, 'set_device', side_effect=AssertionError('GPU called')):
            code, report = self.run_probe(args)
        self.assertEqual(code, 2)
        self.assertIn('ROCm backend is unavailable', report['error'])

    def test_gpu_memory_reporting_is_nullable_and_stage_peaks_reset(self):
        # Mocked API contract only; this does NOT validate ROCm hardware support.
        report = {'stages': []}
        recorder = probe.StageRecorder(torch, torch.device('cuda:0'), report, lambda: None)
        with patch.object(torch.cuda, 'synchronize') as sync, patch.object(torch.cuda, 'reset_peak_memory_stats') as reset, \
             patch.object(torch.cuda, 'memory_allocated', return_value=10), patch.object(torch.cuda, 'memory_reserved', return_value=20), \
             patch.object(torch.cuda, 'max_memory_allocated', return_value=30), patch.object(torch.cuda, 'max_memory_reserved', return_value=40), \
             patch.object(torch.cuda, 'mem_get_info', side_effect=RuntimeError('not supported')):
            self.assertEqual(recorder.run('transfer', lambda: 7), 7)
            reset.assert_called_once()
            self.assertEqual(sync.call_count, 2)
        self.assertEqual(report['stages'][0]['peak_allocated_bytes'], 30)
        self.assertIsNone(report['stages'][0]['device_used_bytes'])
        self.assertIn('unavailable_reason', ' '.join(report['stages'][0]))


if __name__ == '__main__':
    unittest.main()

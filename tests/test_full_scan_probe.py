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

    def test_capacity_probe_configs_only_change_backbone_channels_and_strides(self):
        micro = json.loads((ROOT / 'configs/segmentor_micro.json').read_text())
        for index, widths in enumerate(([4, 8, 16, 32], [8, 16, 32, 64],
                                         [12, 24, 48, 96], [16, 32, 64, 128])):
            with self.subTest(probe=index):
                raw = json.loads((ROOT / f'configs/segmentor_4stage_probe_p{index}.json').read_text())
                expected = copy.deepcopy(micro['model'])
                expected['backbone']['channels'] = widths
                expected['backbone']['downsample_strides'] = [[2, 2, 2]] * 3
                self.assertEqual(raw['model'], expected)
                self.assertEqual(raw['probe'], micro['probe'])
                self.assertEqual(raw['schema_version'], micro['schema_version'])
                self.assertEqual(raw['purpose'], '4stage_capacity_feasibility_probe')

    def test_all_capacity_configs_load_in_rocm_probe_before_hardware_check(self):
        # CPU-only: exercise the actual CLI/config loader, then deliberately
        # stop at backend availability. Never initialize or mock-run a GPU.
        for index in range(4):
            with self.subTest(probe=index):
                path = ROOT / f'configs/segmentor_4stage_probe_p{index}.json'
                raw = json.loads(path.read_text())
                args = self.arguments('backward', f'capacity_load_{index}')
                args[args.index('--model-config')+1] = str(path)
                args[args.index('--backend')+1] = 'rocm'
                args[args.index('--device')+1] = 'cuda:0'
                args[args.index('--candidate')+1] = 'B'
                args.remove('--cpu-synthetic')
                args.remove('--allow-micro-model')
                args += ['--memory-format', 'channels_last_3d']
                with patch.dict(probe.os.environ, {'PYTORCH_MIOPEN_SUGGEST_NHWC': '1'}), \
                     patch.object(torch.cuda, 'is_available', return_value=False), \
                     patch.object(torch.cuda, 'set_device', side_effect=AssertionError('GPU called')):
                    code, report = self.run_probe(args)
                self.assertEqual(code, 2)
                self.assertEqual(report['error'], 'requested ROCm backend is unavailable; no backend/device fallback')
                self.assertEqual(report['model_config'], raw['model'])
                self.assertEqual(report['model_purpose'], raw['purpose'])
                self.assertFalse(report['formal_capacity_validated'])
                self.assertEqual(report['stages'], [])

    def test_capacity_probe_cpu_small_shapes_and_parameter_counts(self):
        from organ_relation.models.segmentor import Segmentor
        from organ_relation.models.segmentor_config import SegmentorConfig
        torch.set_num_threads(2)
        for index, count in enumerate((100756, 398032, 891948, 1582504)):
            with self.subTest(probe=index):
                raw = json.loads((ROOT / f'configs/segmentor_4stage_probe_p{index}.json').read_text())
                config = SegmentorConfig(**raw['model'])
                self.assertEqual(config.backbone.spatial_pyramid((210, 274, 274)),
                                 ((210, 274, 274), (105, 137, 137), (53, 69, 69), (27, 35, 35)))
                model = Segmentor(config).to(memory_format=torch.channels_last_3d)
                self.assertEqual(sum(p.numel() for p in model.parameters()), count)
                image = torch.randn(*raw['probe']['input_shape_bcdhw']).contiguous(memory_format=torch.channels_last_3d)
                with torch.no_grad():
                    output = model(image)
                self.assertEqual(output.coarse_logits.shape, (1, 16, 3, 3, 3))
                self.assertEqual(output.final_logits.shape, (1, 16, 17, 18, 19))
                self.assertTrue(all(torch.isfinite(tensor).all() for tensor in output))
                del model, image, output

    def test_3stage_probe_config_inheritance_shapes_and_parameter_counts(self):
        from organ_relation.models.segmentor import Segmentor
        from organ_relation.models.segmentor_config import SegmentorConfig
        micro = json.loads((ROOT / 'configs/segmentor_micro.json').read_text())
        torch.set_num_threads(2)
        for index, widths, count in ((1, [8, 16, 32], 97872), (2, [12, 24, 48], 218220)):
            with self.subTest(probe=index):
                raw = json.loads((ROOT / f'configs/segmentor_3stage_probe_p{index}.json').read_text())
                expected = copy.deepcopy(micro['model'])
                expected['backbone']['channels'] = widths
                self.assertEqual(raw['model'], expected)
                four_stage = json.loads((ROOT / f'configs/segmentor_4stage_probe_p{index}.json').read_text())
                four_stage['model']['backbone']['channels'] = widths
                four_stage['model']['backbone']['downsample_strides'] = [[2, 2, 2]] * 2
                self.assertEqual(raw['model'], four_stage['model'])
                self.assertEqual(raw['probe'], micro['probe'])
                self.assertEqual(raw['schema_version'], micro['schema_version'])
                self.assertEqual(raw['purpose'], '3stage_capacity_feasibility_probe')
                config = SegmentorConfig(**raw['model'])
                self.assertEqual(config.backbone.spatial_pyramid((210, 274, 274)),
                                 ((210, 274, 274), (105, 137, 137), (53, 69, 69)))
                model = Segmentor(config).to(memory_format=torch.channels_last_3d)
                self.assertEqual(sum(p.numel() for p in model.parameters()), count)
                image = torch.randn(*raw['probe']['input_shape_bcdhw']).contiguous(memory_format=torch.channels_last_3d)
                with torch.no_grad():
                    output = model(image)
                self.assertEqual(output.coarse_logits.shape, (1, 16, 5, 5, 5))
                self.assertEqual(output.final_logits.shape, (1, 16, 17, 18, 19))
                self.assertTrue(all(torch.isfinite(tensor).all() for tensor in output))
                del model, image, output

    def test_3stage_probe_configs_load_and_forward_in_cpu_cli(self):
        for index, count in ((1, 97872), (2, 218220)):
            with self.subTest(probe=index):
                path = ROOT / f'configs/segmentor_3stage_probe_p{index}.json'
                raw = json.loads(path.read_text())
                args = self.arguments('forward', f'3stage_p{index}')
                args[args.index('--model-config')+1] = str(path)
                args[args.index('--candidate')+1] = 'B'
                args.remove('--allow-micro-model')
                args += ['--memory-format', 'channels_last_3d']
                code, report = self.run_probe(args)
                self.assertEqual(code, 0, report)
                self.assertEqual(report['model_config'], raw['model'])
                self.assertEqual(report['model_purpose'], raw['purpose'])
                self.assertEqual(report['parameter_count'], count)
                self.assertFalse(report['formal_capacity_validated'])
                self.assertEqual(report['output_shapes']['coarse_logits'], [1, 16, 1, 2, 2])
                self.assertEqual(report['output_shapes']['final_logits'], [1, 16, 4, 5, 6])
                self.assertEqual(report['stages'][-1]['stage'], 'forward')
                self.assertTrue(all(stage['status'] == 'passed' for stage in report['stages']))

    def test_rocm_channels_last_requires_process_environment_before_data_loading(self):
        args = self.arguments('forward') + ['--memory-format', 'channels_last_3d']
        args[args.index('--backend')+1] = 'rocm'
        args[args.index('--device')+1] = 'cuda:0'
        args.remove('--cpu-synthetic')
        from organ_relation.data.full_scan import FullScanDataset
        for index, value in enumerate((None, '0', 'true')):
            with self.subTest(value=value), patch.dict(probe.os.environ, {}, clear=True):
                if value is not None:
                    probe.os.environ['PYTORCH_MIOPEN_SUGGEST_NHWC'] = value
                attempt = list(args)
                attempt[attempt.index('--output')+1] = str(self.root / f'guard_{index}.json')
                with patch.object(FullScanDataset, 'from_amos_training', side_effect=AssertionError('data loaded')), \
                     patch.object(torch.cuda, 'set_device', side_effect=AssertionError('GPU called')):
                    code, report = self.run_probe(attempt)
                self.assertEqual(code, 2)
                self.assertIn('PYTORCH_MIOPEN_SUGGEST_NHWC=1 before starting Python', report['error'])
                self.assertEqual(report['stages'], [])
                self.assertEqual(report['memory_format'], 'channels_last_3d')
                self.assertEqual(report['environment']['backend_environment']['PYTORCH_MIOPEN_SUGGEST_NHWC'], value)
                self.assertEqual(probe.os.environ.get('PYTORCH_MIOPEN_SUGGEST_NHWC'), value)
        with patch.dict(probe.os.environ, {'PYTORCH_MIOPEN_SUGGEST_NHWC': '1'}):
            probe.check_arguments(probe.parser().parse_args(args))  # Validation only; no GPU.

    def test_default_does_not_require_rocm_layout_environment(self):
        args = self.arguments('forward')
        args[args.index('--backend')+1] = 'rocm'
        args[args.index('--device')+1] = 'cuda:0'
        args.remove('--cpu-synthetic')
        with patch.dict(probe.os.environ, {}, clear=True):
            parsed = probe.parser().parse_args(args)
            self.assertEqual(parsed.memory_format, 'contiguous')
            probe.check_arguments(parsed)

    def test_cpu_channels_last_needs_no_rocm_environment_or_miopen_api(self):
        with patch.dict(probe.os.environ, {}, clear=True), patch.object(torch.backends, 'miopen', None, create=True):
            code, report = self.run_probe(self.arguments('backward') + ['--memory-format', 'channels_last_3d'])
        self.assertEqual(code, 0, report)
        self.assertIsNone(report['environment']['miopen_immediate'])
        self.assertIsNone(report['environment']['backend_environment']['PYTORCH_MIOPEN_SUGGEST_NHWC'])

    def test_memory_format_layout_provenance_numerics_and_no_persistent_state(self):
        original_run = probe.StageRecorder.run
        captured = {}
        def record(recorder, name, operation):
            result = original_run(recorder, name, operation)
            captured[name] = result
            if name == 'transfer':
                result[0].requires_grad_()  # Test input gradients, without changing the production entry point.
            return result
        modes = [('default', []), ('explicit', ['--memory-format', 'contiguous']),
                 ('cl3d', ['--memory-format', 'channels_last_3d']),
                 ('profiled', ['--memory-format', 'channels_last_3d', '--profile-forward-memory']),
                 ('default_after', [])]
        backend_env = {'PYTORCH_MIOPEN_SUGGEST_NHWC': '1', 'MIOPEN_FIND_MODE': 'NORMAL',
                       'PYTORCH_ALLOC_CONF': 'max_split_size_mb:128'}
        for shape in ((8, 6, 4), (15, 13, 9)):
            # Moderate synthetic intensities make the numerical comparison well-conditioned.
            synthetic_pair(self.data, shape=shape, slope=.001, intercept=-.1)
            reference = None
            for mode, flags in modes:
                with self.subTest(shape=shape, mode=mode):
                    captured = {}
                    with patch.dict(probe.os.environ, backend_env), patch.object(probe.StageRecorder, 'run', record):
                        environment_before = dict(probe.os.environ)
                        code, report = self.run_probe(self.arguments('backward', f'{shape[0]}_{mode}') + flags)
                        self.assertEqual(dict(probe.os.environ), environment_before)
                    self.assertEqual(code, 0, report)
                    cl3d = mode in ('cl3d', 'profiled')
                    self.assertEqual(report['memory_format'], 'channels_last_3d' if cl3d else 'contiguous')
                    for key, value in backend_env.items():
                        self.assertEqual(report['environment']['backend_environment'][key], value)
                    self.assertEqual('forward_memory' in report, mode == 'profiled')
                    image, label = captured['transfer']
                    sample = captured['preprocess']
                    model = captured['model_setup']
                    self.assertTrue(image.is_contiguous(memory_format=torch.channels_last_3d if cl3d else torch.contiguous_format))
                    self.assertTrue(torch.equal(image, sample.image.unsqueeze(0)))
                    self.assertEqual(image.dtype, torch.float32)
                    self.assertEqual(label.dtype, torch.int64)
                    self.assertTrue(torch.equal(label, sample.label.unsqueeze(0)))
                    self.assertEqual(label.data_ptr(), sample.label.data_ptr())  # CPU batching only; no label copy/reformat.
                    self.assertEqual(label.stride(), sample.label.unsqueeze(0).stride())
                    self.assertEqual(report['memory_layout']['image_stride'], list(image.stride()))
                    self.assertEqual(report['memory_layout']['label_stride'], list(label.stride()))
                    for name, module in model.named_modules():
                        self.assertFalse(module._forward_hooks or module._forward_pre_hooks or module._backward_hooks)
                        if isinstance(module, torch.nn.Conv3d):
                            self.assertTrue(module.weight.is_contiguous(memory_format=torch.channels_last_3d if cl3d else torch.contiguous_format))
                            self.assertEqual(report['memory_layout']['conv_weight_strides'][name], list(module.weight.stride()))
                            if cl3d and module.kernel_size == (3, 3, 3) and module.in_channels > 1:
                                self.assertFalse(module.weight.is_contiguous())
                    if reference is None:
                        reference, reference_report = captured, report
                    self.assertEqual(report['config'], reference_report['config'])
                    self.assertEqual(report['model_config'], reference_report['model_config'])
                    self.assertEqual(report['model_config_sha256'], reference_report['model_config_sha256'])
                    self.assertEqual(report['gradients']['missing_gradients'], [])
                    self.assertEqual(report['gradients']['nonfinite_gradients'], [])
                    tolerance = dict(rtol=2e-4, atol=2e-5) if cl3d else dict(rtol=0, atol=0)
                    for actual, expected in zip(captured['forward'], reference['forward']):
                        torch.testing.assert_close(actual, expected, **tolerance)
                    torch.testing.assert_close(captured['loss'].total, reference['loss'].total, **tolerance)
                    torch.testing.assert_close(image.grad, reference['transfer'][0].grad, **tolerance)
                    expected_parameters = dict(reference['model_setup'].named_parameters())
                    self.assertEqual(set(vars(model)), set(vars(reference['model_setup'])))
                    for name, parameter in model.named_parameters():
                        expected = expected_parameters[name]
                        self.assertTrue(torch.equal(parameter, expected), name)
                        self.assertIsNotNone(parameter.grad, name)
                        torch.testing.assert_close(parameter.grad, expected.grad, **tolerance, msg=name)

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

    def test_profile_preserves_outputs_loss_and_all_gradients(self):
        plain_code, plain = self.run_probe(self.arguments('backward', 'plain'))
        code, profiled = self.run_probe(self.arguments('backward', 'profiled')+['--profile-forward-memory'])
        self.assertEqual((plain_code, code), (0, 0))
        self.assertNotIn('forward_memory', plain)
        self.assertEqual(plain['loss_values'], profiled['loss_values'])
        self.assertEqual(plain['gradients'], profiled['gradients'])

        from organ_relation.models.segmentor import Segmentor
        from organ_relation.models.segmentor_config import SegmentorConfig
        from organ_relation.models.forward_memory import ForwardMemoryProfile
        config = json.loads((ROOT / 'configs/segmentor_micro.json').read_text())
        model = Segmentor(SegmentorConfig(**config['model']))
        reference = copy.deepcopy(model)
        image = torch.randn(1, 1, 5, 7, 9, requires_grad=True)
        reference_image = image.detach().clone().requires_grad_()
        trace = {}
        with ForwardMemoryProfile(model, 'cpu', trace, lambda: None):
            actual = model(image)
        expected = reference(reference_image)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))
        sum(t.square().mean() for t in actual).backward()
        sum(t.square().mean() for t in expected).backward()
        self.assertTrue(torch.equal(image.grad, reference_image.grad))
        for a, b in zip(model.parameters(), reference.parameters()):
            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.equal(a.grad, b.grad))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        json.dumps(trace, allow_nan=False)  # No tensor or nonfinite JSON payload.
        stages = [e['name'] for e in trace['entries'] if e['kind'] == 'module' and e['parent'] == 0]
        self.assertEqual(stages, ['encoder', 'coarse_head', 'space_to_node', 'relation',
                                  'node_to_space', 'fusion', 'decoder'])
        ops = [e['name'] for e in trace['entries'] if e['kind'] == 'operator']
        self.assertIn('aten.cat.default', ops)
        self.assertIn('aten.upsample_trilinear3d.default', ops)
        self.assertTrue(all(e['status'] == 'passed' and e['peak_allocated_bytes'] is None
                            for e in trace['entries']))

    def test_profile_operator_oom_is_saved_and_hooks_are_removed(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        from organ_relation.models.segmentor import Segmentor
        captured = []
        original = Segmentor.forward
        def forward(model, image):
            captured.append(model)
            return original(model, image)
        class FailConvolution(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if str(func) == 'aten.convolution.default' and args[0].shape[1] == 12:
                    raise torch.OutOfMemoryError('synthetic Tried to allocate 11.26 GiB')
                return func(*args, **(kwargs or {}))
        with FailConvolution(), patch.object(Segmentor, 'forward', forward):
            code, report = self.run_probe(self.arguments('forward')+['--profile-forward-memory'])
        self.assertEqual(code, 2)
        self.assertTrue(report['oom'])
        trace = report['forward_memory']
        self.assertEqual(trace['status'], 'failed')
        active = [trace['entries'][i] for i in trace['failure']['active_entries']]
        self.assertEqual(active[-1]['name'], 'aten.convolution.default')
        self.assertEqual(active[-2]['name'], 'decoder.blocks.1.layers.0')
        self.assertEqual(active[-1]['inputs']['args'][0]['shape'], [1, 12, 4, 6, 8])
        self.assertIn('11.26 GiB', trace['failure']['error'])
        self.assertEqual(len(captured), 1)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in captured[0].modules()))
        self.assertTrue(all(e['status'] != 'running' for e in trace['entries']))

    def test_profile_nested_peaks_survive_resets(self):
        from organ_relation.models.forward_memory import ForwardMemoryProfile
        trace = {}
        profiler = ForwardMemoryProfile(torch.nn.Identity(), 'cuda:0', trace, lambda: None)
        with patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'reset_peak_memory_stats'), \
             patch.object(torch.cuda, 'memory_allocated', return_value=10), \
             patch.object(torch.cuda, 'memory_reserved', return_value=20), \
             patch.object(torch.cuda, 'max_memory_allocated', side_effect=[100, 40, 90, 20]), \
             patch.object(torch.cuda, 'max_memory_reserved', side_effect=[200, 80, 180, 40]):
            root = profiler._start('module', 'root', (), {})
            child = profiler._start('operator', 'op', (), {})
            profiler._finish(child, None)
            profiler._finish(root, None)
        self.assertEqual(root['peak_allocated_bytes'], 90)
        self.assertEqual(root['peak_reserved_bytes'], 180)
        self.assertEqual(child['peak_allocated_bytes'], 90)
        # The parent StageRecorder must not lose these peaks to the final reset.
        report = dict(stages=[], forward_memory=trace)
        recorder = probe.StageRecorder(torch, torch.device('cpu'), report, lambda: None)
        recorder.run('forward', lambda: None)
        self.assertEqual(report['stages'][0]['peak_allocated_bytes'], 90)

    def test_profile_requires_forward(self):
        code, report = self.run_probe(self.arguments('preprocess')+['--profile-forward-memory'])
        self.assertEqual(code, 2)
        self.assertIn('requires --through forward', report['error'])


if __name__ == '__main__':
    unittest.main()

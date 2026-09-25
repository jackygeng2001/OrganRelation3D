"""Single-case FP32 full-scan engineering probe. No epochs or checkpointing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from organ_relation.provenance import code_hashes, git_state, sha256

STAGES = ('preprocess', 'transfer', 'forward', 'loss', 'backward', 'step')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--case', required=True, help='Training CT identifier, e.g. amos_0043')
    p.add_argument('--candidate', choices=('A', 'B', 'C'), required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/baseline.json')
    p.add_argument('--selection-config', type=Path, default=ROOT / 'configs/ct_stats.json')
    p.add_argument('--model-config', type=Path, help='Explicit JSON with model fields; no capacity default')
    p.add_argument('--allow-micro-model', action='store_true', help='Explicit diagnostic use, never formal capacity validation')
    p.add_argument('--through', choices=STAGES, required=True, help='Run prerequisites and stop after this stage')
    p.add_argument('--backend', choices=('rocm', 'cpu'), default='rocm')
    p.add_argument('--device', required=True, help='cuda:0 for ROCm; cpu only for small synthetic checks')
    p.add_argument('--cpu-synthetic', action='store_true', help='Required on CPU; header preflight enforces a voxel limit')
    p.add_argument('--optimizer', choices=('sgd', 'adamw'), help='Required only for --through step')
    p.add_argument('--lr', type=float, help='Explicit probe step size; not a formal training choice')
    p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--momentum', type=float, default=0.)
    p.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999))
    p.add_argument('--optimizer-eps', type=float, default=1e-8)
    p.add_argument('--output', type=Path, required=True, help='New JSON outside the raw data directory')
    return p


def check_arguments(args):
    if args.backend == 'cpu' and (args.device != 'cpu' or not args.cpu_synthetic):
        raise ValueError('CPU probe requires --device cpu --cpu-synthetic; never use real full scans on the laptop')
    if args.backend == 'rocm' and (not args.device.startswith('cuda:') or args.cpu_synthetic):
        raise ValueError('ROCm requires explicit cuda:N and no --cpu-synthetic')
    if args.through == 'step' and (args.optimizer is None or args.lr is None):
        raise ValueError('--through step requires explicit --optimizer and --lr')
    if args.through != 'step' and (args.optimizer is not None or args.lr is not None):
        raise ValueError('optimizer arguments require --through step')
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError('lr must be finite and positive')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('weight_decay must be finite and nonnegative')
    if not math.isfinite(args.momentum) or not 0 <= args.momentum < 1:
        raise ValueError('momentum must be in [0,1)')
    if any(not math.isfinite(x) or not 0 <= x < 1 for x in args.betas):
        raise ValueError('betas must be in [0,1)')
    if not math.isfinite(args.optimizer_eps) or args.optimizer_eps <= 0:
        raise ValueError('optimizer_eps must be finite and positive')


class StageRecorder:
    """Synchronized wall times and stage-local allocator peaks, including live tensors."""

    def __init__(self, torch, device, report, save):
        self.torch, self.device, self.report, self.save = torch, device, report, save

    def memory(self):
        values = dict(allocated_bytes=None, reserved_bytes=None,
                      peak_allocated_bytes=None, peak_reserved_bytes=None,
                      device_free_bytes=None, device_total_bytes=None, device_used_bytes=None)
        if self.device.type == 'cuda':
            cuda = self.torch.cuda
            values.update(allocated_bytes=cuda.memory_allocated(self.device),
                          reserved_bytes=cuda.memory_reserved(self.device),
                          peak_allocated_bytes=cuda.max_memory_allocated(self.device),
                          peak_reserved_bytes=cuda.max_memory_reserved(self.device))
            try:
                free, total = cuda.mem_get_info(self.device)
                if not 0 <= free <= total or total <= 0:
                    raise RuntimeError('backend returned inconsistent whole-device memory counters')
                values.update(device_free_bytes=free, device_total_bytes=total, device_used_bytes=total-free,
                              device_memory_source='torch.cuda.mem_get_info backend snapshot; includes other processes, not a peak')
            except (RuntimeError, AttributeError) as exc:
                values['device_memory_unavailable_reason'] = str(exc)
        return values

    def run(self, name, operation):
        entry = {'stage': name, 'status': 'running', 'seconds': None, 'oom': False}
        self.report['stages'].append(entry)
        self.report['active_stage'] = name
        self.save()
        started = time.perf_counter()
        try:
            if self.device.type == 'cuda':
                self.torch.cuda.synchronize(self.device)
                self.torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            result = operation()
            if self.device.type == 'cuda':
                self.torch.cuda.synchronize(self.device)
            entry['seconds'] = time.perf_counter() - started
            entry.update(self.memory(), status='passed')
            return result
        except Exception as exc:
            entry.update(status='failed', seconds=time.perf_counter()-started,
                         error_type=type(exc).__name__, error=str(exc),
                         oom=isinstance(exc, self.torch.OutOfMemoryError))
            # Do not synchronize or clear allocator state after OOM. Preserve partial data.
            try:
                entry.update(self.memory())
            except Exception as memory_error:
                entry['memory_unavailable_reason'] = str(memory_error)
            raise
        finally:
            self.save()


def execute(args, report, save):
    import torch
    from organ_relation.data.full_scan import FullScanDataset, FullScanPreprocessor
    from organ_relation.losses import JointLoss
    from organ_relation.models.diagnostics import parameter_inventory
    from organ_relation.models.segmentor import Segmentor
    from organ_relation.models.segmentor_config import SegmentorConfig

    check_arguments(args)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config.get('schema_version') != 1 or config.get('epsilon') != 1e-6 or config.get('loss') != {'lambda_c': 0.5, 'align_corners': False}:
        raise ValueError('this baseline probe requires epsilon=1e-6, lambda_c=0.5, coarse align_corners=False')
    probe = config['probe']
    if probe['dtype'] != 'float32' or probe['batch_size'] != 1:
        raise ValueError('first feasibility probe requires batch 1 and float32')
    if type(probe['cpu_threads']) is not int or probe['cpu_threads'] < 1 or type(probe['seed']) is not int:
        raise ValueError('explicit positive cpu_threads and integer seed required')
    model_raw = json.loads(args.model_config.read_text(encoding='utf-8')) if args.model_config else None
    if STAGES.index(args.through) >= STAGES.index('forward') and model_raw is None:
        raise ValueError('model capacity is not frozen: provide --model-config explicitly')
    model_config = SegmentorConfig(**model_raw['model']) if model_raw else None
    if model_config and model_config.epsilon != config['epsilon']:
        raise ValueError('model and loss must use the same baseline epsilon')
    is_micro = bool(model_raw and model_raw.get('purpose') == 'cpu_synthetic_test_not_formal_experiment')
    if is_micro and not args.allow_micro_model:
        raise ValueError('micro config requires --allow-micro-model; never a formal capacity result')
    if args.backend == 'cpu' and model_config and max(model_config.backbone.channels) > 64:
        raise ValueError('CPU synthetic probe refuses non-micro channel widths')
    device = torch.device(args.device)
    report.update(config=config, config_sha256=sha256(args.config),
                  model_config=model_config.to_dict() if model_config else None,
                  model_config_sha256=sha256(args.model_config) if model_raw else None,
                  model_purpose=model_raw.get('purpose', 'explicit_external_config') if model_raw else None,
                  formal_capacity_validated=False, parameter_count=None,
                  selection_config_sha256=sha256(args.selection_config))
    report['environment'].update(torch=torch.__version__, rocm=torch.version.hip,
                                 cuda=torch.version.cuda, gpu=None,
                                 dependencies={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'nibabel')})
    if args.backend == 'rocm':
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('requested ROCm backend is unavailable; no backend/device fallback')
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        report['environment']['gpu'] = {'name': properties.name, 'total_memory_bytes': properties.total_memory}
    torch.set_num_threads(probe['cpu_threads'])
    torch.manual_seed(probe['seed'])
    report['environment']['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
    report['environment']['cudnn_benchmark'] = torch.backends.cudnn.benchmark
    report['environment']['cudnn_deterministic'] = torch.backends.cudnn.deterministic
    recorder = StageRecorder(torch, device, report, save)
    processor = FullScanPreprocessor(config['preprocessing'], args.candidate,
                                     max_voxels=262144 if args.backend == 'cpu' else None)

    def preprocess():
        selection = json.loads(args.selection_config.read_text(encoding='utf-8'))
        dataset = FullScanDataset.from_amos_training(args.data_root, [args.case], selection, processor)
        report['manifest_sha256'] = sha256(args.data_root / selection['manifest'])
        report['sample_metadata'] = processor.inspect(dataset.pairs[0])
        report['shape_dhw'] = report['sample_metadata']['shape_dhw']
        save()  # Shape is available even if reading/interpolating voxels later fails.
        sample = dataset[0]
        report['sample_metadata'] = sample.metadata
        return sample

    sample = recorder.run('preprocess', preprocess)
    if args.through == 'preprocess':
        return
    # Dataset owns sample/channel axes; this single-case probe owns batching.
    image, label = recorder.run('transfer', lambda: (
        sample.image.unsqueeze(0).to(device), sample.label.unsqueeze(0).to(device)))
    del sample
    if args.through == 'transfer':
        return
    def setup_model():
        model = Segmentor(model_config)
        report['parameter_count'] = sum(p.numel() for p in model.parameters())
        report['parameter_bytes'] = sum(p.numel()*p.element_size() for p in model.parameters())
        return model.to(device=device, dtype=torch.float32)
    model = recorder.run('model_setup', setup_model)
    model.train()
    criterion = JointLoss(epsilon=config['epsilon'], **config['loss'])
    optimizer = None
    if args.through == 'step':
        if args.optimizer == 'adamw':
            options = dict(lr=args.lr, betas=tuple(args.betas), eps=args.optimizer_eps,
                           weight_decay=args.weight_decay, foreach=False, fused=False)
            optimizer = recorder.run('optimizer_setup', lambda: torch.optim.AdamW(model.parameters(), **options))
        else:
            options = dict(lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, foreach=False)
            optimizer = recorder.run('optimizer_setup', lambda: torch.optim.SGD(model.parameters(), **options))
        report['optimizer'] = {'name': args.optimizer, 'options': options, 'purpose': 'single engineering step; not formal training config'}
        optimizer.zero_grad(set_to_none=True)
    # No autocast context; all model inputs/parameters are FP32.
    output = recorder.run('forward', lambda: model(image))
    report['output_shapes'] = {'coarse_logits': list(output.coarse_logits.shape), 'final_logits': list(output.final_logits.shape)}
    if args.through == 'forward':
        return
    loss = recorder.run('loss', lambda: criterion(output.coarse_logits, output.final_logits, label))
    report['loss_values'] = {'total': loss.total.detach().item(),
                            'coarse': loss.coarse.segmentation.detach().item(),
                            'final': loss.final.segmentation.detach().item()}
    if args.through == 'loss':
        return
    recorder.run('backward', loss.total.backward)

    def check_gradients():
        inventory = parameter_inventory(model)
        report['gradients'] = inventory
        if inventory['missing_gradients'] or inventory['nonfinite_gradients']:
            raise ValueError('missing/nonfinite parameter gradients; see gradients in JSON')
    recorder.run('gradient_check', check_gradients)
    if args.through == 'step':
        recorder.run('step', optimizer.step)

        def check_parameters():
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('optimizer produced nonfinite parameters')
        recorder.run('parameter_check', check_parameters)
        report['optimizer_state_bytes'] = sum(t.numel()*t.element_size() for state in optimizer.state.values()
                                             for t in state.values() if isinstance(t, torch.Tensor))


def main(argv=None):
    args = parser().parse_args(argv)
    output, data = args.output.resolve(), args.data_root.resolve()
    if output.exists() or output.is_relative_to(data):
        raise ValueError('output must be a new JSON outside the original data directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve this report before execution; subsequent writes only update this run.
    with output.open('x', encoding='utf-8'):
        pass
    report = {'schema_version': 1, 'utc_time': datetime.now(timezone.utc).isoformat(),
              'case_id': args.case, 'candidate': args.candidate, 'through': args.through,
              'device': args.device, 'dtype': 'float32', 'status': 'running', 'oom': False,
              'shape_dhw': None, 'stages': [], 'git': git_state(), 'source_sha256': code_hashes(),
              'environment': {'python': platform.python_version(), 'platform': platform.platform()},
              'notes': ['Single cold pass, train mode; timings include runtime finite checks and no warmup.',
                        'Stage allocator peaks include live allocations from preceding stages.',
                        'Whole-device memory is an optional backend snapshot, never a measured device peak.',
                        'First optimizer step only; not steady-state training memory or throughput.']}

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    save()
    try:
        execute(args, report, save)
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      oom=any(stage['oom'] for stage in report['stages']))
    finally:
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            observed = [s[key] for s in report['stages'] if s.get(key) is not None]
            report[key] = max(observed, default=None)
        save()
    print(f"{report['status'].upper()}: {output}")
    if report['status'] != 'passed':
        print(report['error'], file=sys.stderr)
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())

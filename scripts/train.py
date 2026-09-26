"""Unified FP32 full-scan training. Only tiny synthetic scans are permitted on CPU."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from organ_relation.data.ct_stats import contained_path
from organ_relation.data.splits import (create_development, load_split, select_cases,
                                       training_manifest, write_split)
from organ_relation.provenance import code_hashes, git_state
from organ_relation.training.state import digest, seed_all


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--split', type=Path, help='Existing fixed split artifact; never regenerated while training')
    p.add_argument('--resume', type=Path, help='Trusted checkpoint in the original run directory')
    p.add_argument('--prepare-split', type=Path, help='Only write a NEW 160/40 artifact, then exit; no voxels loaded')
    p.add_argument('--split-seed', type=int, default=20260925)
    p.add_argument('--cases', nargs='+', help='Explicit subset of the configured training role')
    p.add_argument('--cpu-synthetic', action='store_true', help='Only with CPU config; enforces small voxel/channel limits')
    p.add_argument('--stop-after', type=int, help='Safe invocation step budget for resume acceptance, not a new run length')
    p.add_argument('--no-tensorboard', action='store_true', help='Disable scalar observer; JSONL/checkpoints unchanged')
    p.add_argument('--quiet-console', action='store_true', help='Disable presentation only; errors still visible')
    return p


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def outside_data(path, data_root):
    if path.resolve().is_relative_to(data_root.resolve()):
        raise ValueError('outputs must be outside original data root')


def execute(args):
    import torch
    from organ_relation.data.full_scan import FullScanDataset, FullScanPreprocessor, ScanPair
    from organ_relation.losses import JointLoss
    from organ_relation.models.segmentor import Segmentor
    from organ_relation.models.segmentor_config import SegmentorConfig
    from organ_relation.training.engine import Trainer, validate_options
    from organ_relation.training.console import TrainingConsole

    config = read_json(args.config)
    if config.get('schema_version') != 1:
        raise ValueError('unsupported training config schema')
    base = args.config.resolve().parent
    selection = read_json(base / config['selection_config'])
    manifest = training_manifest(args.data_root, selection)
    if not args.cpu_synthetic and len(manifest['records']) != 200:
        raise ValueError('real AMOS training pool must contain exactly 200 CT cases')
    if args.prepare_split:
        if args.resume or args.run_dir or args.split or args.cpu_synthetic:
            raise ValueError('prepare-split is a separate real-data metadata-only operation')
        outside_data(args.prepare_split, args.data_root)
        artifact = create_development(manifest, args.split_seed, 160)
        write_split(args.prepare_split, artifact)
        print(f'Created fixed 160/40 split: {args.prepare_split}; hash={artifact["split_hash"]}', flush=True)
        return
    if args.run_dir is None:
        raise ValueError('--run-dir is required for training')
    outside_data(args.run_dir, args.data_root)
    split_path = args.split or (base / config['split_artifact'] if config.get('split_artifact') else None)
    if split_path is None:
        raise ValueError('provide an existing --split; first use --prepare-split once')
    artifact = load_split(split_path)
    if artifact['manifest'] != manifest:
        raise ValueError('live official training manifest differs from frozen artifact')
    if not args.cpu_synthetic and (len(artifact['development']['train']), len(artifact['development']['internal_dev'])) != (160, 40):
        raise ValueError('real development split must be 160/40')
    data, runtime, options = config['data'], config['runtime'], config['training']
    validate_options(options)
    if data['role'] not in ('development_train', 'final_train'):
        raise ValueError('validation/test roles cannot train')
    chosen = select_cases(artifact, data['role'], args.cases if args.cases else data['train_cases'],
                          None if args.cases else data['train_limit'])
    val_records = []
    if options['validation_every']:
        if options['validation_role'] == 'train_monitor':
            val_records = chosen
        elif options['validation_role'] == 'internal_dev' and data['role'] == 'development_train':
            val_records = select_cases(artifact, 'internal_dev', data['validation_cases'], data['validation_limit'])
        else:
            raise ValueError('training v1 validates only train_monitor or development internal_dev')
    if runtime['dtype'] != 'float32' or type(runtime['cpu_threads']) is not int or runtime['cpu_threads'] < 1:
        raise ValueError('FP32 and positive cpu_threads required')
    if runtime['backend'] == 'cpu':
        if runtime['device'] != 'cpu' or not args.cpu_synthetic:
            raise ValueError('CPU requires cpu device and --cpu-synthetic')
    elif runtime['backend'] == 'rocm':
        if (args.cpu_synthetic or not runtime['device'].startswith('cuda:')
                or options['memory_format'] != 'channels_last_3d'
                or os.environ.get('PYTORCH_MIOPEN_SUGGEST_NHWC') != '1'):
            raise ValueError('ROCm requires channels_last_3d, cuda:N and process-start PYTORCH_MIOPEN_SUGGEST_NHWC=1')
        if not torch.version.hip or not torch.cuda.is_available():
            raise ValueError('ROCm unavailable; no fallback')
        torch.cuda.set_device(runtime['device'])
    else:
        raise ValueError('training v1 supports CPU synthetic / ROCm only')
    baseline = read_json(base / config['baseline_config'])
    model_config = SegmentorConfig(**read_json(base / config['model_config'])['model'])
    if model_config.epsilon != baseline['epsilon']:
        raise ValueError('model/loss epsilon mismatch')
    if args.cpu_synthetic and max(model_config.backbone.channels) > 64:
        raise ValueError('CPU synthetic channel limit exceeded')
    processor = FullScanPreprocessor(baseline['preprocessing'], data['candidate'],
                                    max_voxels=262144 if args.cpu_synthetic else None)

    def dataset(records):
        return FullScanDataset([ScanPair(r['case_id'], contained_path(args.data_root, r['image']),
                                         contained_path(args.data_root, r['label'])) for r in records], processor)

    train_data = dataset(chosen)
    val_data = dataset(val_records) if val_records else None
    case_identity = {}
    for ds in (train_data, val_data):
        if ds is not None:
            for pair in ds.pairs:
                meta = processor.inspect(pair)
                case_identity[pair.case_id] = {role: {key: meta[role][key] for key in
                    ('header_sha256', 'file_size_bytes', 'mtime_ns')} for role in ('image', 'label')}
    optimizer_config = config['optimizer']
    if (optimizer_config['name'] != 'AdamW' or optimizer_config['foreach'] is not False
            or optimizer_config['fused'] is not False):
        raise ValueError('v1 requires explicit AdamW foreach=false fused=false')
    for key in ('lr', 'eps', 'weight_decay'):
        value = optimizer_config[key]
        if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (key != 'weight_decay' and value == 0):
            raise ValueError('invalid optimizer ' + key)
    if len(optimizer_config['betas']) != 2 or any(not 0 <= b < 1 for b in optimizer_config['betas']):
        raise ValueError('invalid AdamW betas')
    torch.set_num_threads(runtime['cpu_threads'])
    seed_all(options['seed'])
    model = Segmentor(model_config).to(device=runtime['device'], dtype=torch.float32)
    if options['memory_format'] == 'channels_last_3d':
        model.to(memory_format=torch.channels_last_3d)
    optimizer = torch.optim.AdamW(model.parameters(), **{k: v for k, v in optimizer_config.items() if k != 'name'})
    criterion = JointLoss(epsilon=baseline['epsilon'], **baseline['loss'])
    data_identity = dict(manifest=manifest, manifest_hash=digest(manifest), split=artifact['development'],
                         split_hash=artifact['split_hash'], role=data['role'], actual_train=chosen,
                         actual_validation=val_records, subset_hash=digest([chosen, val_records]),
                         case_identity=case_identity)
    identity = dict(model=model_config.to_dict(), loss=dict(epsilon=baseline['epsilon'], **baseline['loss']),
                    preprocessing=dict(candidate=data['candidate'], **baseline['preprocessing']),
                    optimizer=optimizer_config, runtime=runtime, training=options, data=data_identity,
                    provenance=dict(git=git_state(), source_hashes=code_hashes()),
                    environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                        packages={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'nibabel')},
                        gpu_name=torch.cuda.get_device_name(runtime['device']) if runtime['backend'] == 'rocm' else None,
                        rocm=torch.version.hip, cuda=torch.version.cuda,
                        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                        cudnn_benchmark=torch.backends.cudnn.benchmark,
                        cudnn_deterministic=torch.backends.cudnn.deterministic,
                        backend_environment={k: v for k, v in os.environ.items()
                            if k.startswith(('MIOPEN_', 'PYTORCH_MIOPEN_')) or k in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')}))
    # Normalize tuples to JSON lists so saved manifest and checkpoint identities agree.
    identity = json.loads(json.dumps(identity))
    if identity['provenance']['git'].get('commit') is None:
        raise ValueError('training requires readable Git/source provenance; use a Git clone')
    trainer = Trainer(model, criterion, optimizer, train_data, [r['case_id'] for r in chosen],
                      device=runtime['device'], options=options, identity=identity, run_dir=args.run_dir,
                      validation_dataset=val_data, validation_case_ids=[r['case_id'] for r in val_records],
                      resume=args.resume, console=TrainingConsole(enabled=not args.quiet_console),
                      tensorboard=not args.no_tensorboard)
    trainer.run(stop_after=args.stop_after)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        execute(args)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f'Training stopped: {type(exc).__name__}: {exc}. Resume only from the last valid checkpoint.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

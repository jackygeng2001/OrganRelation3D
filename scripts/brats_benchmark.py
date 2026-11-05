"""Throughput probe: real-case training steps for one execution variant.

Measures data loading, then synchronized forward+loss+backward+AdamW steps and
no-grad validation forwards on a few real cases, without creating a run,
checkpoint or log. Weights are freshly initialized and discarded; the numbers
are timing only, never training results. Run one variant per process so
allocator and MIOpen state do not leak between variants.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def parse_crop(text):
    if text == 'none':
        return None
    kind, margin, multiple = text.split(':')
    if kind != 'bbox':
        raise ValueError('crop must be none or bbox:MARGIN:MULTIPLE')
    return dict(kind='nonzero_bbox', margin=int(margin), multiple_of=int(multiple))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/train_brats_year_fold.json')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--source-partition', choices=('all', 'training'), default='all')
    parser.add_argument('--split', type=Path, required=True, help='Cases are taken from its train list, in order')
    parser.add_argument('--cases', type=int, default=6)
    parser.add_argument('--passes', type=int, default=2, help='Pass 1 includes first-shape solver search')
    parser.add_argument('--eval-cases', type=int, default=2, help='No-grad validation forwards to time')
    parser.add_argument('--crop', default='none', help='none or bbox:MARGIN:MULTIPLE')
    parser.add_argument('--convolution-backend', choices=('native', 'depth_sliced'), default='depth_sliced')
    parser.add_argument('--cudnn-benchmark', action='store_true', help='MIOpen exhaustive solver search')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)

    import torch
    from organ_relation.data.brats import BraTSDataset, index_cases
    from organ_relation.losses import RegionJointLoss
    from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
    from organ_relation.training.state import seed_all

    config = json.loads(args.config.read_text())
    if (torch.version.hip and config['training']['memory_format'] == 'channels_last_3d'
            and os.environ.get('PYTORCH_MIOPEN_SUGGEST_NHWC') != '1'):
        raise SystemExit('ROCm requires process-start PYTORCH_MIOPEN_SUGGEST_NHWC=1')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    crop = parse_crop(args.crop)

    manifest = index_cases(args.source, cohort=args.cohort, partition=args.source_partition)
    split = json.loads(args.split.read_text())
    by_id = {r['case_id']: r for r in manifest['records']}
    train_ids = split['train'][:args.cases]
    dataset = BraTSDataset(dict(manifest, records=[by_id[c] for c in train_ids]),
                           **dict(config['preprocessing'], spatial_crop=crop))
    samples, load_seconds = [], []
    for index in range(len(dataset)):
        started = time.perf_counter()
        samples.append(dataset[index])
        load_seconds.append(time.perf_counter() - started)

    seed_all(config['training']['seed'])
    model = BrainTumorSegmentor(BrainTumorConfig(**config['model']),
                               convolution_backend=args.convolution_backend).to(device)
    channels_last = config['training']['memory_format'] == 'channels_last_3d'
    if channels_last:
        model.to(memory_format=torch.channels_last_3d)
    criterion = RegionJointLoss(**config['loss'])
    optimizer = torch.optim.AdamW(model.parameters(), **config['optimizer'])

    def to_device(sample):
        image = sample.image.unsqueeze(0).to(device)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last_3d)
        return image, sample.label.unsqueeze(0).to(device)

    steps, failure = [], None
    for pass_index in range(args.passes):
        for sample in samples:
            image, label = to_device(sample)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            try:
                optimizer.zero_grad(set_to_none=True)
                output = model(image)
                loss = criterion(output.coarse_logits, output.final_logits, label)
                loss.total.backward()
                optimizer.step()
                torch.cuda.synchronize(device)
            except torch.OutOfMemoryError as exc:
                failure = dict(kind='oom', case_id=sample.metadata['case_id'], error=str(exc).splitlines()[0])
                break
            steps.append(dict(pass_index=pass_index, case_id=sample.metadata['case_id'],
                              shape_dhw=sample.metadata['shape_dhw'], seconds=time.perf_counter() - started,
                              loss=loss.total.item(),
                              peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30))
            print(json.dumps(steps[-1]), flush=True)
            del image, label, output, loss
        if failure:
            break

    evals = []
    if failure is None:
        model.eval()
        with torch.no_grad():
            for sample in samples[:args.eval_cases]:
                for repeat in range(2):
                    image, label = to_device(sample)
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    output = model(image)
                    torch.cuda.synchronize(device)
                    evals.append(dict(repeat=repeat, case_id=sample.metadata['case_id'],
                                      seconds=time.perf_counter() - started))
                    del image, label, output

    def mean(values):
        return statistics.fmean(values) if values else None
    last = [s for s in steps if s['pass_index'] == args.passes - 1]
    report = dict(
        variant=dict(crop=crop, convolution_backend=args.convolution_backend,
                     cudnn_benchmark=args.cudnn_benchmark, memory_format=config['training']['memory_format']),
        torch=torch.__version__, hip=torch.version.hip, gpu=torch.cuda.get_device_name(device),
        cases=train_ids, failure=failure,
        load_seconds_mean=mean(load_seconds),
        voxels_mean=mean([float(s.label.numel()) for s in samples]),
        first_pass_step_seconds_mean=mean([s['seconds'] for s in steps if s['pass_index'] == 0]),
        steady_step_seconds_mean=mean([s['seconds'] for s in last]),
        steady_seconds_per_megavoxel=mean([s['seconds'] / (s['shape_dhw'][0] * s['shape_dhw'][1] * s['shape_dhw'][2] / 1e6)
                                           for s in last]),
        eval_forward_seconds_mean=mean([e['seconds'] for e in evals if e['repeat'] == 1]),
        peak_allocated_gib_max=max((s['peak_allocated_gib'] for s in steps), default=None),
        peak_reserved_gib_max=max((s['peak_reserved_gib'] for s in steps), default=None),
        steps=steps, evals=evals)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text)
    print(json.dumps({k: v for k, v in report.items() if k not in ('steps', 'evals', 'cases')}, indent=2))
    return 0 if failure is None else 3


if __name__ == '__main__':
    raise SystemExit(main())

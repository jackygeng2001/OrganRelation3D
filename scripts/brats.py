"""BraTS code preflight and explicit resumable runs; no implicit experiment split."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('inspect','check','overfit','train'))
    parser.add_argument('--config',type=Path,default=ROOT/'configs/brats_regions.json')
    parser.add_argument('--source',type=Path,help='One cohort archive or extracted directory, read-only')
    parser.add_argument('--cohort',choices=('brats2023_gli','brats2024_gli_post','brats2018','brats2019','brats2020','brats2021'))
    parser.add_argument('--case',help='Single case for no-update check or explicit overfit diagnostic')
    parser.add_argument('--shape',type=int,nargs=3,default=(9,17,19),help='CPU synthetic preflight only')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--convolution-backend', choices=('native','depth_sliced'), default='native',
                        help='Explicit equivalent Conv3d execution; depth_sliced uses bounded NCHW Conv2d')
    parser.add_argument('--split',type=Path,help='Explicit patient-disjoint train/dev JSON with manifest_hash')
    parser.add_argument('--max-steps',type=int,help='Explicit engineering-run horizon; no formal protocol default')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--extend-to',type=int,help='With --resume: increase total steps; keep original --max-steps')
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--profile-memory',action='store_true',
                        help='check only: report phase/operator memory and failures to stdout')
    args = parser.parse_args(argv)
    try:
        execute(args)
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        print(f'BraTS stopped: {exc}',file=sys.stderr)
        return 2
    return 0


def profile_check(model, criterion, image, label, device):
    """One unchanged forward/loss/backward; scalar metadata only, never a retry.

    Reuse the existing ATen/module profiler. Peaks must be folded back across
    its nested resets; failed snapshots are taken BEFORE locals unwind. Output
    is bounded to stage summaries, largest operator peaks and failure context.
    """
    import torch
    from organ_relation.models.forward_memory import ForwardMemoryProfile, describe
    from organ_relation.models.diagnostics import DeviceMemoryMonitor, parameter_inventory

    report = dict(status='running', optimizer_steps=0, image=describe(image), label=describe(label),
                  failure=None, phases=[], notes=[
                      'One full-volume FP32 autograd pass; no optimizer step or activation release.',
                      'Synchronized profiling timings include overhead; not training throughput.',
                      'ATen memory does not identify the internal MIOpen solver/workspace.',
                      'Backward reports ATen operators; module scopes refer only to forward calls.'])
    monitor = DeviceMemoryMonitor(device)
    for phase in ('forward', 'loss', 'backward'):
        trace = {}
        monitor.begin()
        before = monitor.snapshot(phase)
        started = time.perf_counter()
        failure = None
        try:
            module = criterion if phase == 'loss' else model
            with ForwardMemoryProfile(module, device, trace, lambda: None):
                if phase == 'forward':
                    output = model(image)
                elif phase == 'loss':
                    loss = criterion(output.coarse_logits, output.final_logits, label)
                else:
                    loss.total.backward()
            memory = monitor.snapshot(phase)
        except Exception as exc:
            failure = trace.get('failure') or dict(error_type=type(exc).__name__, error=str(exc),
                oom=isinstance(exc, torch.OutOfMemoryError), active_entries=[])
            memory = failure.get('memory', {})  # Do not synchronize after a GPU failure.

        entries = trace.get('entries', [])
        def brief(entry):
            parents = []
            parent = entry['parent']
            while parent is not None:
                ancestor = entries[parent]
                if ancestor['kind'] == 'module':
                    parents.append(ancestor['name'])
                parent = ancestor['parent']
            item = {k:entry[k] for k in ('index','kind','name','status','inputs','outputs','before',
                                         'after','seconds','peak_allocated_bytes','peak_reserved_bytes')}
            item['module_scopes'] = list(reversed(parents))
            peak, allocated = entry['peak_allocated_bytes'], entry['before']['allocated_bytes']
            item['extra_peak_allocated_bytes'] = None if peak is None else max(0, peak-allocated)
            return item

        def major(entry):
            name = entry['name']
            return entry['kind'] == 'module' and (name.count('.') <= 1 or
                (name.count('.') == 2 and name.startswith(('encoder.stages.', 'decoder.up.',
                                                          'decoder.fuse.', 'decoder.skip_projection.'))))

        peaks = {}
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            values = [v for v in [memory.get(key), *(e[key] for e in entries)] if v is not None]
            peaks[key] = max(values) if values else None
        operators = [brief(e) for e in entries if e['kind'] == 'operator' and e['peak_allocated_bytes'] is not None]
        operators.sort(key=lambda e:e['extra_peak_allocated_bytes'], reverse=True)
        last = trace.get('last_successful_operator')
        last_module = trace.get('last_successful_module')
        record = dict(phase=phase, status='failed' if failure else 'passed', before=before,
                      after=memory, seconds=time.perf_counter()-started, **peaks,
                      modules=[brief(e) for e in entries if major(e)], largest_operator_peaks=operators[:8],
                      last_successful_module=brief(entries[last_module]) if last_module is not None else None,
                      last_successful_operator=brief(entries[last]) if last is not None else None)
        report['phases'].append(record)
        if failure:
            report.update(status='failed', failure=dict(phase=phase, **failure,
                active_scopes=[brief(entries[i]) for i in failure['active_entries']]))
            break
        if phase == 'forward':
            report.update(coarse_shape=list(output.coarse_logits.shape), final_shape=list(output.final_logits.shape))
        elif phase == 'loss':
            report['loss'] = loss.total.item()
    for key in ('peak_allocated_bytes','peak_reserved_bytes'):
        values = [p[key] for p in report['phases'] if p[key] is not None]
        report[key] = max(values) if values else None
    if report['failure'] is None:
        report.update(parameter_inventory(model))
        report['status'] = 'failed' if report['missing_gradients'] or report['nonfinite_gradients'] else 'passed'
    return report


def training_plan(args, manifest, config):
    """Keep diagnostic reuse of a case separate from patient-disjoint training."""
    from organ_relation.data.brats import validate_partition
    from organ_relation.training.state import digest

    if (manifest is None or args.run_dir is None or args.max_steps is None
            or args.max_steps < 1):
        raise ValueError('training requires --source, --run-dir and positive --max-steps')
    options = dict(config['training'], max_steps=args.max_steps)
    data = dict(manifest_hash=digest(manifest), manifest=manifest)
    if args.action == 'overfit':
        if not args.case or args.split is not None:
            raise ValueError('overfit requires --case and forbids --split; it is not independent validation')
        if (options['validation_role'] != 'train_monitor' or options['max_epochs'] is not None
                or options.get('cadence_unit','step') != 'step'):
            raise ValueError('overfit requires train_monitor, step cadence and max_epochs=null')
        selected = [r for r in manifest['records'] if r['case_id'] == args.case]
        if len(selected) != 1:
            raise ValueError('overfit case must be uniquely present in source')
        data['selection'] = dict(kind='single_case_overfit', train=[args.case], monitor=[args.case],
                                 subject_id=selected[0]['subject_id'], independent_validation=False)
        return selected, selected, options, data
    if args.action != 'train' or args.split is None or args.case:
        raise ValueError('train requires --split and forbids --case; use explicit overfit for a single case')
    if options['validation_role'] != 'internal_dev':
        raise ValueError('train requires internal_dev; same-case monitoring belongs to overfit')
    split = json.loads(args.split.read_text())
    train, dev = validate_partition(manifest, split)
    data.update(split_hash=digest(split), split=split)
    return train, dev, options, data


def execute(args):
    import torch
    from organ_relation.data.brats import BraTSDataset,index_cases
    from organ_relation.losses import RegionJointLoss
    from organ_relation.models.brain_tumor import BrainTumorConfig,BrainTumorSegmentor
    from organ_relation.models.diagnostics import DeviceMemoryMonitor,parameter_inventory
    from organ_relation.training.state import digest,seed_all
    from organ_relation.provenance import code_hashes,git_state

    if args.extend_to is not None and (args.action not in ('train','overfit') or args.resume is None or args.extend_to < 1):
        raise ValueError('--extend-to requires training --resume and a positive total')
    if args.profile_memory and args.action != 'check':
        raise ValueError('--profile-memory is only available for no-update check')
    config = json.loads(args.config.read_text())
    if config.get('schema_version') != 1 or config.get('mode') != 'brats_regions':
        raise ValueError('expected brain-region config')
    if args.source and not args.cohort:
        raise ValueError('--source requires an explicit --cohort; labels are not inferred from values')
    if config['training']['memory_format'] not in ('contiguous','channels_last_3d'):
        raise ValueError('unsupported memory format')
    manifest = index_cases(args.source,cohort=args.cohort) if args.source else None
    if args.action == 'inspect':
        if manifest is None:
            raise ValueError('--source required')
        print(json.dumps(dict(cohort=args.cohort,cases=len(manifest['records']),
            subjects=len({r['subject_id'] for r in manifest['records']}),manifest_hash=digest(manifest),
            three_region_compatible=args.cohort!='brats2024_gli_post'),indent=2))
        return
    if args.action == 'check' and (args.run_dir or args.resume or args.split or args.max_steps or args.stop_after):
        raise ValueError('check is a no-update/no-run preflight')
    if args.action in ('train','overfit'):
        train,dev,options,data_identity = training_plan(args,manifest,config)
    device = torch.device(args.device)
    if device.type not in ('cpu','cuda'):
        raise ValueError('CPU/CUDA/ROCm only; no fallback')
    if device.type == 'cuda':
        device = torch.device('cuda', 0 if device.index is None else device.index)
        if not torch.cuda.is_available():
            raise ValueError('requested accelerator unavailable')
        if (torch.version.hip and config['training']['memory_format'] == 'channels_last_3d'
                and os.environ.get('PYTORCH_MIOPEN_SUGGEST_NHWC') != '1'):
            raise ValueError('ROCm requires process-start PYTORCH_MIOPEN_SUGGEST_NHWC=1')
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    seed_all(config['training']['seed'])
    model = BrainTumorSegmentor(BrainTumorConfig(**config['model']),
                               convolution_backend=args.convolution_backend).to(device=device)
    from organ_relation.models.depth_sliced_conv import DEPTH_CHUNK
    execution = dict(convolution_backend=args.convolution_backend,
                     convolution_depth_chunk=DEPTH_CHUNK if args.convolution_backend=='depth_sliced' else None)
    if config['training']['memory_format'] == 'channels_last_3d':
        model.to(memory_format=torch.channels_last_3d)
    criterion = RegionJointLoss(**config['loss'])
    if args.action == 'check':
        if manifest is None:
            if args.case or min(args.shape) < 1 or max(args.shape) > 32 or device.type != 'cpu':
                raise ValueError('synthetic check is CPU only with dimensions 1..32')
            image = torch.randn(1,4,*args.shape,device=device)
            label = (torch.arange(image[0,0].numel(),device=device).reshape(1,*args.shape)%4).long()
        else:
            if not args.case:
                raise ValueError('real-volume check requires explicit --case')
            selected = [r for r in manifest['records'] if r['case_id']==args.case]
            if len(selected)!=1:
                raise ValueError('case not uniquely present')
            sample = BraTSDataset(dict(manifest,records=selected),**config['preprocessing'])[0]
            image,label = sample.image.unsqueeze(0).to(device),sample.label.unsqueeze(0).to(device)
        if config['training']['memory_format']=='channels_last_3d':
            image=image.contiguous(memory_format=torch.channels_last_3d)
        if args.profile_memory:
            report = profile_check(model,criterion,image,label,device)
            report['case_id'], report['cohort'] = args.case, args.cohort
            report['torch'], report['rocm'] = str(torch.__version__), torch.version.hip
            report['memory_format'] = config['training']['memory_format']
            report.update(execution)
            report['backend_environment'] = {k:v for k,v in os.environ.items()
                if k.startswith(('MIOPEN_','PYTORCH_MIOPEN_','PYTORCH_CUDA_','PYTORCH_ALLOC_'))}
            print(json.dumps(report,indent=2,allow_nan=False),flush=True)
            if report['status'] != 'passed':
                raise RuntimeError('profile failed; phase/operator details are in the diagnostic JSON on stdout')
            return
        monitor = DeviceMemoryMonitor(device);monitor.begin()
        start=time.perf_counter()
        output=model(image)
        loss=criterion(output.coarse_logits,output.final_logits,label)
        loss.total.backward()
        memory=monitor.snapshot('forward_loss_backward')
        inventory=parameter_inventory(model)
        print(json.dumps(dict(status='passed' if not inventory['missing_gradients'] and not inventory['nonfinite_gradients'] else 'failed',
            image_shape=list(image.shape),coarse_shape=list(output.coarse_logits.shape),final_shape=list(output.final_logits.shape),
            loss=loss.total.item(),seconds=time.perf_counter()-start,optimizer_steps=0,memory=memory,
            **execution,**inventory),indent=2))
        if inventory['missing_gradients'] or inventory['nonfinite_gradients']:
            raise ValueError('gradient preflight failed')
        return
    if device.type == 'cpu':
        raise ValueError('real CPU training disabled; use CPU synthetic tests for trajectory acceptance')
    if args.run_dir.resolve().is_relative_to(args.source.resolve()):
        raise ValueError('run output must be outside original dataset')
    train_data=BraTSDataset(dict(manifest,records=train),**config['preprocessing'])
    dev_data=BraTSDataset(dict(manifest,records=dev),**config['preprocessing'])
    identity=dict(mode='brats_regions',model=config['model'],loss=config['loss'],threshold=config['threshold'],
        preprocessing=config['preprocessing'],training=options,optimizer=dict(name='AdamW',**config['optimizer']),
        data=data_identity,
        runtime=dict(device=str(device),dtype='float32',cpu_threads=torch.get_num_threads(),
                     backend='rocm' if torch.version.hip else 'cuda',**execution),
        environment=dict(python=platform.python_version(),platform=platform.platform(),
            torch=str(torch.__version__),rocm=torch.version.hip,cuda=torch.version.cuda,
            numpy=importlib.metadata.version('numpy'),nibabel=importlib.metadata.version('nibabel'),
            gpu_name=torch.cuda.get_device_name(device),backend_environment={k:v for k,v in os.environ.items()
                if k.startswith(('MIOPEN_','PYTORCH_MIOPEN_','PYTORCH_CUDA_','PYTORCH_ALLOC_'))},
            deterministic=torch.are_deterministic_algorithms_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
        provenance=dict(git=git_state(),source_hashes=code_hashes()))
    identity=json.loads(json.dumps(identity))
    optimizer=torch.optim.AdamW(model.parameters(),**config['optimizer'])
    from organ_relation.training.engine import Trainer
    from organ_relation.training.console import TrainingConsole
    Trainer(model,criterion,optimizer,train_data,[r['case_id'] for r in train],device=device,
        options=options,identity=identity,run_dir=args.run_dir,validation_dataset=dev_data,
        validation_case_ids=[r['case_id'] for r in dev],resume=args.resume,tensorboard=True,
        console=TrainingConsole(),extend_to=args.extend_to).run(stop_after=args.stop_after)


if __name__=='__main__':
    raise SystemExit(main())

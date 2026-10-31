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
    parser.add_argument('action',choices=('inspect','check','train'))
    parser.add_argument('--config',type=Path,default=ROOT/'configs/brats_regions.json')
    parser.add_argument('--source',type=Path,help='One cohort archive or extracted directory, read-only')
    parser.add_argument('--cohort',choices=('brats2023_gli','brats2024_gli_post','brats2018','brats2019','brats2020','brats2021'))
    parser.add_argument('--case',help='Single case for no-update preflight')
    parser.add_argument('--shape',type=int,nargs=3,default=(9,17,19),help='CPU synthetic preflight only')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--split',type=Path,help='Explicit patient-disjoint train/dev JSON with manifest_hash')
    parser.add_argument('--max-steps',type=int,help='Explicit engineering-run horizon; no formal protocol default')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--stop-after',type=int)
    args = parser.parse_args(argv)
    try:
        execute(args)
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        print(f'BraTS stopped: {exc}',file=sys.stderr)
        return 2
    return 0


def execute(args):
    import torch
    from organ_relation.data.brats import BraTSDataset,index_cases,validate_partition
    from organ_relation.losses import RegionJointLoss
    from organ_relation.models.brain_tumor import BrainTumorConfig,BrainTumorSegmentor
    from organ_relation.models.diagnostics import DeviceMemoryMonitor,parameter_inventory
    from organ_relation.training.state import digest,seed_all
    from organ_relation.provenance import code_hashes,git_state

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
    if args.action == 'train' and (manifest is None or args.split is None or args.run_dir is None
                                   or args.max_steps is None or args.max_steps < 1 or args.case):
        raise ValueError('train requires --source, --split, --run-dir, --max-steps; --case is only for check')
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
    model = BrainTumorSegmentor(BrainTumorConfig(**config['model'])).to(device=device)
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
        monitor = DeviceMemoryMonitor(device);monitor.begin()
        start=time.perf_counter()
        output=model(image)
        loss=criterion(output.coarse_logits,output.final_logits,label)
        loss.total.backward()
        memory=monitor.snapshot('forward_loss_backward')
        inventory=parameter_inventory(model)
        print(json.dumps(dict(status='passed' if not inventory['missing_gradients'] and not inventory['nonfinite_gradients'] else 'failed',
            image_shape=list(image.shape),coarse_shape=list(output.coarse_logits.shape),final_shape=list(output.final_logits.shape),
            loss=loss.total.item(),seconds=time.perf_counter()-start,optimizer_steps=0,memory=memory,**inventory),indent=2))
        if inventory['missing_gradients'] or inventory['nonfinite_gradients']:
            raise ValueError('gradient preflight failed')
        return
    if device.type == 'cpu':
        raise ValueError('real CPU training disabled; use CPU synthetic tests for trajectory acceptance')
    if args.run_dir.resolve().is_relative_to(args.source.resolve()):
        raise ValueError('run output must be outside original dataset')
    split=json.loads(args.split.read_text())
    train,dev=validate_partition(manifest,split)
    train_data=BraTSDataset(dict(manifest,records=train),**config['preprocessing'])
    dev_data=BraTSDataset(dict(manifest,records=dev),**config['preprocessing'])
    options=dict(config['training'],max_steps=args.max_steps)
    identity=dict(mode='brats_regions',model=config['model'],loss=config['loss'],threshold=config['threshold'],
        preprocessing=config['preprocessing'],training=options,optimizer=dict(name='AdamW',**config['optimizer']),
        data=dict(manifest_hash=digest(manifest),manifest=manifest,split_hash=digest(split),split=split),
        runtime=dict(device=str(device),dtype='float32',cpu_threads=torch.get_num_threads(),
                     backend='rocm' if torch.version.hip else 'cuda'),
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
        console=TrainingConsole()).run(stop_after=args.stop_after)


if __name__=='__main__':
    raise SystemExit(main())

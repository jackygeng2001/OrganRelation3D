"""One read-only whole-volume inference from a trusted BraTS run checkpoint."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from organ_relation.data.brats import BraTSDataset, index_cases
from organ_relation.losses import REGION_NAMES, RegionJointLoss, region_targets
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.models.depth_sliced_conv import DEPTH_CHUNK
from organ_relation.provenance import code_hashes, git_state, sha256
from organ_relation.training.brain_tumor import region_diagnostics, region_dice
from organ_relation.training.state import digest, load_checkpoint, seed_all


def verify_source(identity):
    """Inference permits new diagnostics and the audited extension-only upgrade.

    Model/data/loss/engine code stays fixed. This is not resume: the training
    identity and checkpoint are not rewritten. Normal resume stays strict.
    """
    saved = {k:v for k,v in identity['provenance']['source_hashes'].items() if k.startswith('src/')}
    current = {k:v for k,v in code_hashes().items() if k.startswith('src/')}
    if not saved:
        raise ValueError('package source differs from checkpoint; use its original model/data/loss implementation')
    if saved == current:
        return 'identical'
    # The only permitted package difference is the audited extension allowlist
    # itself, never model/data/loss/engine changes. No checkpoint is migrated here.
    from organ_relation.training.extension import check_extension_identity
    expected=copy.deepcopy(identity)
    expected['provenance']=dict(git=git_state(),source_hashes=code_hashes())
    try:
        check_extension_identity(identity,expected)
    except ValueError as exc:
        raise ValueError('package source differs from checkpoint: '+str(exc)) from exc
    return 'audited_extension_only'


def load_run_checkpoint(checkpoint, run):
    """Read original or current extended execution identity, retaining run origin."""
    identity=run['identity']
    try:
        state=load_checkpoint(checkpoint,identity)
    except ValueError as exc:
        if not str(exc).startswith('resume identity mismatch:'):
            raise
        expected=copy.deepcopy(identity)
        expected['provenance']=dict(git=git_state(),source_hashes=code_hashes())
        state=load_checkpoint(checkpoint,expected)
        if state.get('origin_identity')!=identity:
            raise ValueError('extended checkpoint original run identity mismatch')
    if state['run_id']!=run['run_id']:raise ValueError('checkpoint/run_id mismatch')
    return state


def distribution(values):
    values = np.asarray(values)
    if not values.size:
        return dict(count=0, mean=None, min=None, p05=None, p50=None, p95=None, max=None, probability_sum=0.0)
    q = np.quantile(values, [0, .05, .5, .95, 1])
    return dict(count=int(values.size), mean=float(values.mean(dtype=np.float64)),
                **dict(zip(('min','p05','p50','p95','max'),map(float,q))),
                probability_sum=float(values.sum(dtype=np.float64)))


def probability_statistics(logits, label, threshold):
    """CPU observations; 'outside' means outside EACH overlapping GT region."""
    targets = region_targets(label)[0].numpy()
    probabilities = logits.sigmoid()[0].numpy()
    hard = region_dice(logits, label, threshold)
    rows = []
    for i, region in enumerate(hard['organs']):
        p, gt = probabilities[i], targets[i]
        inside, outside = distribution(p[gt]), distribution(p[~gt])
        total_mass = inside['probability_sum'] + outside['probability_sum']
        rows.append(dict(**region, false_negative_voxels=region['gt_voxels']-region['true_positive'],
            predicted_to_gt_ratio=region['predicted_voxels']/region['gt_voxels'] if region['gt_voxels'] else None,
            inside_gt=inside, outside_gt=outside,
            outside_gt_below_threshold=distribution(p[(~gt)&(p<threshold)]),
            probability_sum=total_mass,
            outside_probability_mass_fraction=outside['probability_sum']/total_mass if total_mass else None))
    return dict(mean_hard_dice=hard['mean_dice'], hierarchy_violations=hard['hierarchy_violations'], regions=rows)


def compare_monitor(run_dir, step, case, run_id, diagnostics, final_hard):
    path = run_dir / 'metrics.jsonl'
    if not path.exists():
        return dict(status='unavailable', reason='no metrics.jsonl')
    candidates = [r for line in path.read_text().splitlines() if line.strip()
                  for r in [json.loads(line)] if r.get('global_step')==step
                  and r.get('run_id')==run_id and r.get('phase') in ('train_monitor','internal_dev')
                  and case in r.get('region_diagnostic_cases',{})]
    if len(candidates) != 1:
        return dict(status='unavailable', reason='no unique comparable monitor at this checkpoint step')
    old = candidates[0]['region_diagnostic_cases'][case]
    comparisons = []
    def add(name, actual, expected):
        equal = actual is None and expected is None if actual is None or expected is None else math.isclose(actual,expected,rel_tol=1e-5,abs_tol=1e-6)
        comparisons.append(dict(field=name, actual=actual, recorded=expected, matches=equal))
    add('total_loss',diagnostics['total_loss'],old['total_loss'])
    for branch in ('final','coarse'):
        for key in ('bce','dice_loss','loss'):
            add(branch+'.'+key,diagnostics[branch][key],old[branch][key])
        for i,name in enumerate(REGION_NAMES):
            add(branch+'.soft_dice.'+name,diagnostics[branch]['soft_dice'][i],old[branch]['soft_dice'][i])
            if branch=='coarse':
                add('coarse.hard_dice.'+name,diagnostics[branch]['hard']['organs'][i]['dice'],old[branch]['hard']['organs'][i]['dice'])
    # A multi-case aggregate must not be compared with one case's Dice.
    if candidates[0]['metrics']['cases']==1:
        add('final.mean_hard_dice',final_hard,candidates[0]['metrics']['mean_case_dice'])
    return dict(status='matched' if all(c['matches'] for c in comparisons) else 'mismatch',
                tolerance=dict(rtol=1e-5,atol=1e-6),fields=comparisons)


def check_overlay_path(path, run_dir, source):
    if path is None:
        return
    path = path.resolve()
    if path.suffix.lower()!='.png' or path.exists() or path.is_relative_to(run_dir.resolve()) or path.is_relative_to(source.resolve()):
        raise ValueError('--overlay must be a new PNG outside the original run and source')


def render_overlay(path, image, label, final, coarse, threshold):
    """GT-selected display slices ONLY, after full-volume prediction."""
    from PIL import Image, ImageDraw, ImageFont
    target = region_targets(label)[0].numpy()
    final, coarse = final.sigmoid()[0].numpy(), coarse.sigmoid()[0].numpy()
    flair = image[3].numpy()
    nonzero = flair[flair!=0]
    lo,hi = np.quantile(nonzero,[.01,.99]) if nonzero.size else (0,1)
    gray = np.clip((flair-lo)/max(float(hi-lo),1e-8),0,1)
    panel, gap, top = 240, 10, 70
    canvas = Image.new('RGB',(5*(panel+gap)+gap,top+3*(panel+55)+65),'white')
    draw = ImageDraw.Draw(canvas)
    draw.font = ImageFont.load_default(size=14)
    draw.text((gap,8),'Whole-volume inference; display: native D slices (no anatomical L/R convention)',fill='black')
    headings = ('FLAIR','GT (green)','Final: TP/FP/FN','Coarse: TP/FP/FN','Final P: black=0 white=1')
    for j,title in enumerate(headings):draw.text((gap+j*(panel+gap),40),title,fill='black')
    for i,name in enumerate(REGION_NAMES):
        z = int(target[i].sum((1,2)).argmax()) if target[i].any() else target.shape[1]//2
        gt = target[i,z]
        base = np.repeat(gray[z,:,:,None],3,axis=2)
        gt_rgb=base.copy();gt_rgb[gt]=.45*gt_rgb[gt]+.55*np.array([0,1,0])
        tiles=[base,gt_rgb]
        for probability in (final[i,z],coarse[i,z]):
            pred=probability>=threshold;rgb=base.copy()
            for mask,color in ((gt&pred,[0,1,0]),(~gt&pred,[1,0,0]),(gt&~pred,[0,0,1])):
                rgb[mask]=.4*rgb[mask]+.6*np.array(color)
            tiles.append(rgb)
        tiles.append(np.repeat(final[i,z,:,:,None],3,axis=2))
        y=top+i*(panel+55)
        draw.text((gap,y),f'{name}: D={z}, largest GT area; visualization only',fill='black')
        for j,tile in enumerate(tiles):
            pic=Image.fromarray(np.uint8(np.clip(tile,0,1)*255))
            scale=panel/max(pic.size)
            pic=pic.resize(tuple(max(1,round(n*scale)) for n in pic.size),Image.Resampling.NEAREST)
            canvas.paste(pic,(gap+j*(panel+gap),y+22))
    draw.text((gap,canvas.height-40),'Green = true positive; red = false positive; blue = false negative. Threshold from checkpoint.',fill='black')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:canvas.save(stream,format='PNG')


def execute(args):
    checkpoint = args.checkpoint.resolve();run_dir=checkpoint.parent
    run = json.loads((run_dir/'run.json').read_text());identity=run['identity']
    if identity.get('mode')!='brats_regions':raise ValueError('BraTS region checkpoint required')
    state=load_run_checkpoint(checkpoint,run)
    identity=state['identity']
    source_compatibility=verify_source(identity)
    saved=identity['data']['manifest'];source=args.source or Path(saved['source'])
    check_overlay_path(args.overlay,run_dir,source)
    if args.overlay is not None:
        from PIL import Image  # Fail before inference if optional rendering dependency is absent.
    manifest=index_cases(source,cohort=saved['cohort'],partition=saved.get('source_partition','all'))
    if manifest!=saved or digest(manifest)!=identity['data']['manifest_hash']:
        raise ValueError('source metadata differs from checkpoint manifest')
    records=[r for r in manifest['records'] if r['case_id']==args.case]
    if len(records)!=1:raise ValueError('case must be uniquely present in the checkpoint source')
    runtime=identity['runtime'];backend=runtime['convolution_backend']
    if backend not in ('native','depth_sliced') or runtime['dtype']!='float32':raise ValueError('unsupported saved execution settings')
    if backend=='depth_sliced' and runtime['convolution_depth_chunk']!=DEPTH_CHUNK:raise ValueError('depth chunk differs from checkpoint')
    memory_format=identity['training']['memory_format']
    if memory_format not in ('contiguous','channels_last_3d'):raise ValueError('unsupported saved memory format')
    device=torch.device(args.device)
    if device.type not in ('cpu','cuda'):raise ValueError('CPU/CUDA/ROCm only')
    if device.type=='cuda':
        if not torch.cuda.is_available():raise ValueError('requested accelerator unavailable')
        if torch.version.hip and memory_format=='channels_last_3d' and os.environ.get('PYTORCH_MIOPEN_SUGGEST_NHWC')!='1':
            raise ValueError('ROCm requires process-start PYTORCH_MIOPEN_SUGGEST_NHWC=1')
        torch.cuda.set_device(device)
    torch.set_num_threads(runtime['cpu_threads']);seed_all(identity['training']['seed'])
    sample=BraTSDataset(dict(manifest,records=records),**identity['preprocessing'])[0]
    model=BrainTumorSegmentor(BrainTumorConfig(**identity['model']),convolution_backend=backend)
    model.load_state_dict(state['model'],strict=True)
    del state['optimizer']  # CPU deserialized state only; no optimizer is constructed/restored.
    model=model.to(device).eval();image=sample.image.unsqueeze(0).to(device)
    if memory_format=='channels_last_3d':
        model.to(memory_format=torch.channels_last_3d);image=image.contiguous(memory_format=torch.channels_last_3d)
    if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    with torch.inference_mode():
        output=model(image)  # Exactly one image-only full-volume forward.
        if device.type=='cuda':torch.cuda.synchronize(device)
        forward_seconds=time.perf_counter()-started
        memory={k:getattr(torch.cuda,k)(device) for k in ('max_memory_allocated','max_memory_reserved')} if device.type=='cuda' else None
        output=type(output)(*(x.cpu() for x in output));label=sample.label.unsqueeze(0)
        del model,image
        criterion=RegionJointLoss(**identity['loss'])
        diagnostic=region_diagnostics(output,label,criterion,identity['threshold'])
        coarse=criterion.align_coarse(output.coarse_logits,label)
        branches={name:probability_statistics(logits,label,identity['threshold']) for name,logits in (('final',output.final_logits),('coarse',coarse))}
        report=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),run_id=run['run_id'],global_step=state['progress']['global_step'],
            case=args.case,shape=list(label.shape[1:]),full_volume_voxels=label.numel(),threshold=identity['threshold'],
            execution=dict(device=str(device),dtype='float32',memory_format=memory_format,convolution_backend=backend,forward_calls=1,optimizer_steps=0),
            training_commit=identity['provenance']['git'],diagnostic_commit=git_state(),
            package_sources_match=source_compatibility=='identical',source_compatibility=source_compatibility,
            torch=str(torch.__version__),rocm=torch.version.hip,backend_environment={k:v for k,v in os.environ.items() if k.startswith(('MIOPEN_','PYTORCH_MIOPEN_','PYTORCH_CUDA_','PYTORCH_ALLOC_'))},
            geometry=sample.metadata,forward_seconds=forward_seconds,forward_memory_bytes=memory,
            loss_and_soft_dice=diagnostic,**branches,
            coarse_native_shape=list(output.coarse_logits.shape),
            coarse_native_predicted_voxels=[int(v) for v in (output.coarse_logits.sigmoid()>=identity['threshold']).sum((0,2,3,4)).tolist()])
        try:
            report['monitor_comparison']=compare_monitor(run_dir,report['global_step'],args.case,run['run_id'],diagnostic,branches['final']['mean_hard_dice'])
        except (OSError,ValueError,KeyError,TypeError) as exc:
            report['monitor_comparison']=dict(status='unavailable',reason=str(exc))
        # Always emit complete statistics, even if a recorded comparable value differs.
        print(json.dumps(report,indent=2,allow_nan=False),flush=True)
        if args.overlay:
            render_overlay(args.overlay,sample.image,label,output.final_logits,coarse,identity['threshold'])
            print('Overlay: '+str(args.overlay.resolve()),file=sys.stderr)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True,help='Trusted local checkpoint, with adjacent run.json')
    parser.add_argument('--case',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--source',type=Path,help='Defaults to exact source saved in checkpoint; metadata must match')
    parser.add_argument('--overlay',type=Path,help='Optional new PNG outside source/run; never overwrite')
    args=parser.parse_args(argv)
    try:execute(args)
    except (ValueError,RuntimeError,OSError,KeyError,ImportError) as exc:
        print('BraTS diagnostic stopped: '+str(exc),file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())

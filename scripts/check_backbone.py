"""One synthetic forward/backward probe. Not a training or CT-data entry point."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from organ_relation.backbone import UNetBackbone3D
from organ_relation.backbone_config import BackboneConfig
from organ_relation.backbone_diagnostics import DeviceMemoryMonitor,parameter_inventory
from organ_relation.ct_stats import PROJECT_ROOT,code_hashes,git_state,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/backbone_micro.json')
    parser.add_argument('--device',default='cpu',help='explicit cpu or cuda:N; ROCm also uses cuda:N')
    parser.add_argument('--output',type=Path,required=True,help='new JSON path inside project reports/')
    args=parser.parse_args()
    path=args.output.resolve()
    if not path.is_relative_to((PROJECT_ROOT/'reports').resolve()):
        raise ValueError('diagnostic output must stay inside project reports/')
    if path.exists():raise ValueError('refuse overwriting an existing diagnostic report')
    raw=json.loads(args.config.read_text(encoding='utf-8'))
    if raw.get('schema_version')!=1 or raw.get('purpose')!='cpu_synthetic_test_not_formal_experiment':
        raise ValueError('use an explicitly nonformal synthetic-test configuration')
    config=BackboneConfig(**raw['model']);probe=raw['probe']
    shape=tuple(probe['input_shape_bcdhw'])
    if len(shape)!=5 or any(type(n) is not int or n<1 for n in shape) or shape[1]!=1:
        raise ValueError('probe shape must be positive [B,1,D,H,W]')
    if probe['dtype']!='float32':raise ValueError('stage-1 diagnostic is float32 only; no implicit AMP')
    if type(probe['cpu_threads']) is not int or probe['cpu_threads']<1:
        raise ValueError('cpu_threads must be a positive integer')
    config.validate_spatial(shape[2:])
    device=torch.device(args.device)
    monitor=DeviceMemoryMonitor(device)
    # Avoid accidental full-volume CPU execution by this micro probe.
    if device.type=='cpu' and (shape[0]*shape[2]*shape[3]*shape[4]>262144 or max(config.channels)>64):
        raise ValueError('CPU diagnostic accepts micro tensors only; no silent shrinking')
    torch.set_num_threads(probe['cpu_threads']);torch.manual_seed(probe['seed'])
    model=UNetBackbone3D(config).to(device=device,dtype=torch.float32)
    x=torch.randn(shape,device=device,requires_grad=True)
    monitor.begin();measurements=[monitor.snapshot('parameters_and_input')]
    started=time.perf_counter()
    features=model.encoder(x)
    logits=model.decoder(features.deepest,features.skips)
    measurements.append(monitor.snapshot('forward'))
    # A scalar autograd probe, NOT METHOD_SPEC joint segmentation objective.
    scalar=logits.square().mean()+0.01*logits.mean()
    scalar.backward();measurements.append(monitor.snapshot('backward'))
    inventory=parameter_inventory(model)
    if tuple(logits.shape)!=(shape[0],16,*shape[2:]):raise AssertionError('wrong output extent')
    if inventory['missing_gradients'] or inventory['nonfinite_gradients']:
        raise AssertionError('missing/nonfinite trainable parameter gradients')
    if not bool(torch.isfinite(logits).all()) or x.grad is None or not bool(torch.isfinite(x.grad).all()) or not bool(x.grad.abs().sum()>0):
        raise AssertionError('nonfinite output or absent input gradient')
    result={'utc_time':datetime.now(timezone.utc).isoformat(),'git':git_state(),
            'source_sha256':code_hashes(),'config':raw,'config_sha256':sha256(args.config),
            'python':platform.python_version(),'platform':platform.platform(),
            'torch':torch.__version__,'torch_hip':torch.version.hip,'torch_cuda':torch.version.cuda,
            'dependencies':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
            'device':str(device),'input_shape':list(shape),'skip_shapes':[list(s.shape) for s in features.skips],
            'F_shape':list(features.deepest.shape),'logits_shape':list(logits.shape),
            'parameter_inventory':inventory,'input_gradient_abs_sum':float(x.grad.abs().sum()),
            'memory':measurements,'elapsed_seconds':time.perf_counter()-started,
            'scope':'one synthetic backbone forward/backward; no labels, graph, joint loss, optimizer or training',
            'status':'passed'}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','device','F_shape','skip_shapes','logits_shape','elapsed_seconds']},ensure_ascii=False))
    print(path)
    return 0


if __name__=='__main__':raise SystemExit(main())

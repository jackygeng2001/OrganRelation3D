"""Pure metadata shape audit: no torch import, volume loading or allocation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from organ_relation.models.backbone_config import BackboneConfig
from organ_relation.provenance import PROJECT_ROOT,git_state,sha256


def audit_shapes(records,config):
    results=[];seen=set()
    for record in records:
        key=(record['case_id'],record['candidate_id'])
        if key in seen:raise ValueError('duplicate case/candidate')
        seen.add(key)
        pyramid=config.validate_spatial(record['shape_dhw'])
        results.append({'case_id':key[0],'candidate_id':key[1],
                        'input_dhw':list(pyramid[0]),'encoder_dhw':[list(s) for s in pyramid],
                        'F_channels':config.channels[-1],
                        'decoder_dhw':[list(s) for s in reversed(pyramid[:-1])],
                        'output_dhw':list(pyramid[0]),'input_padding_voxels':0})
    if not results:raise ValueError('empty estimate list')
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--estimates',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/backbone_micro.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to((PROJECT_ROOT/'reports').resolve()) or output.exists():
        raise ValueError('use a new output file inside project reports/')
    raw=json.loads(args.config.read_text(encoding='utf-8'));config=BackboneConfig(**raw['model'])
    records=json.loads(args.estimates.read_text(encoding='utf-8'))['cases']
    rows=audit_shapes(records,config)
    result={'scope':'shape algebra only; micro config is not formal architecture; no tensor/GPU memory claim',
            'config':raw,'git':git_state(),'estimates_sha256':sha256(args.estimates),
            'config_sha256':sha256(args.config),'script_sha256':sha256(Path(__file__)),
            'shape_module_sha256':sha256(PROJECT_ROOT/'src/organ_relation/models/backbone_config.py'),
            'case_count':len({r['case_id'] for r in rows}),'case_candidate_count':len(rows),'rows':rows}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'PASS: {result["case_count"]} cases / {len(rows)} candidate shapes; no tensor allocation')
    return 0


if __name__=='__main__':raise SystemExit(main())

"""Summarize completed pilot JSON without rereading any CT or label volume."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import sys
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from organ_relation.provenance import git_state,sha256


def summarize(report):
    result_path=report/'results.json'
    result=json.loads(result_path.read_text(encoding='utf-8'));cases=result['cases']
    config=result['provenance']['config']
    if len(cases)!=len(config['cases']) or any(len(c['candidates'])!=3 for c in cases):
        raise ValueError('require completed pilot, not partial results')
    lines=['# 分层保真度复核','',
           '有目的抽样；不能当作 200 例总体估计。几何 Dice、体积及灰度指标无预设合格阈值。',
           '', '|原始层间距组|病例数|方案|左右肾上腺 Dice 均值|肾上腺目标体积绝对变化均值 %|前景 CT MAE HU 均值|原生 xyz 前景梯度比均值|',
           '|---|---|---|---|---|---|---|']
    for name,group in [('5 mm',[c for c in cases if c['source_spacing'][2]>=4.9]),
                       ('1.25–2.5 mm',[c for c in cases if c['source_spacing'][2]<4.9])]:
        for cid in ['A','B','C']:
            candidates=[next(r for r in c['candidates'] if r['id']==cid) for c in group]
            if not candidates:continue
            organs=[r for c in candidates for r in c['organs'] if r['label'] in (11,12) and r['source_present']]
            mean=lambda vals:statistics.mean(vals)
            gradient=[round(mean(c['ct_global']['foreground_gradient_abs_ratio_native_axes'][a] for c in candidates),3) for a in range(3)]
            lines.append(f'|{name}|{len(group)}|{cid}|{mean(r["roundtrip_dice"] for r in organs):.4f}|{mean(abs(r["target_volume_change_percent"]) for r in organs):.2f}|{mean(c["ct_global"]["original_foreground_mae_hu"] for c in candidates):.2f}|{gradient}|')
    lines+=['','## 原始缺失与采样消失','',
            '“原始缺失”仅表示原始标签中该类计数为零；不能单凭标签判定器官解剖上不存在，也可能涉及扫描覆盖范围或标注情况。此状态不进入模型。','']
    for case in cases:
        absent=[r['organ'] for r in case['candidates'][0]['organs'] if r['source_absent']]
        lost=[f'{c["id"]}:{r["organ"]}' for c in case['candidates'] for r in c['organs'] if r['disappeared_in_target'] or r['disappeared_in_roundtrip']]
        lines.append(f'- {case["case_id"]}：原始缺失 {absent or "无"}；目标/往返采样消失 {lost or "无"}。')
    lines+=['','## 各方案最低肾上腺 Dice 与体积变化','']
    for cid in ['A','B','C']:
        items=[(case['case_id'],r) for case in cases for c in case['candidates'] if c['id']==cid for r in c['organs'] if r['label'] in (11,12) and r['source_present']]
        worst=min(items,key=lambda pair:pair[1]['roundtrip_dice']);volume=max(items,key=lambda pair:abs(pair[1]['target_volume_change_percent']))
        lines.append(f'- {cid}：最低 Dice {worst[1]["roundtrip_dice"]:.4f}（{worst[0]}，{worst[1]["organ"]}）；最大目标体积相对变化 {volume[1]["target_volume_change_percent"]:+.2f}%（{volume[0]}，{volume[1]["organ"]}）。')
    lines+=['','## CT 对比度复核','',
            '|方案|肾上腺原始局部对比度 HU 范围|往返绝对对比度比均值 / 最低|往返器官内标准差比均值|','|---|---|---|---|']
    for cid in ['A','B','C']:
        organs=[r for case in cases for c in case['candidates'] if c['id']==cid for r in c['ct_organs'] if r['label'] in (11,12) and r['source_present']]
        ratios=[r['local_contrast_abs_ratio'] for r in organs if r.get('local_contrast_abs_ratio') is not None]
        contrasts=[r['source_local_contrast_hu'] for r in organs if 'source_local_contrast_hu' in r]
        stds=[r['std_ratio'] for r in organs if r.get('std_ratio') is not None]
        lines.append(f'|{cid}|{min(contrasts):.2f} … {max(contrasts):.2f}|{statistics.mean(ratios):.3f} / {min(ratios):.3f}|{statistics.mean(stds):.3f}|')
    lines+=['','背景环组织不均匀，以上比值是描述指标；往返包含两次插值，必须结合直接目标网格图判断。','','## 直接目标网格概览','']
    for start in range(0,len(cases),5):
        subset=cases[start:start+5];canvas=Image.new('RGB',(1620,len(subset)*185),'#15191f')
        for row,case in enumerate(subset):
            for col,cid in enumerate(['A','B','C']):
                with Image.open(report/'visuals'/f'{case["case_id"]}_{cid}_target.png') as im:
                    tile=im.crop((0,0,1080,355)).resize((540,178),Image.Resampling.LANCZOS)
                canvas.paste(tile,(col*540,row*185))
        name=f'target_contact_{start//5+1}.png';canvas.save(report/'visuals'/name)
        lines.append(f'![{name}](visuals/{name})')
    transforms=[]
    for case in cases:
        for c in case['candidates']:
            target=np.array(c['grid']['affine_ras_mm'])
            row={'case_id':case['case_id'],'candidate':c['id']}
            for role in ['image','label']:
                source=np.array(case[f'source_{role}_affine_ras_mm'])
                row[f'target_to_original_{role}_index']=np.linalg.solve(source,target).tolist()
                row[f'original_{role}_to_target_index']=np.linalg.solve(target,source).tolist()
            transforms.append(row)
    provenance={'utc_time':datetime.now(timezone.utc).isoformat(),'git':git_state(),
                'summary_script_sha256':sha256(Path(__file__)),'input_results_sha256':sha256(result_path)}
    (report/'spatial_transforms.json').write_text(json.dumps({'provenance':provenance,'transforms':transforms},indent=2),encoding='utf-8')
    (report/'review_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report/'review_summary.md'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report-dir',type=Path,required=True)
    print(summarize(parser.parse_args().report_dir))

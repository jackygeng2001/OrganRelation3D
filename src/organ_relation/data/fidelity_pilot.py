"""Execute and preserve a small full-scan fidelity pilot on CPU."""
import argparse
import csv
from datetime import datetime,timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import psutil

from ..provenance import PROJECT_ROOT, code_hashes, git_state, sha256
from .ct_stats import contained_path
from .nifti_header import read_header
from .fidelity import (LABELS,affine4,antialias_sigmas,build_target_grid,
                      ct_metrics,label_metrics,load_pair,resample,chunks)
from .fidelity_visuals import capture,save_review,save_target_review,view_specs


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def csv_rows(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def fingerprint(path):
    st=path.stat()
    return {'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'header':read_header(path)}


def memory_check(config):
    info=psutil.Process().memory_info()
    rss=info.rss
    if rss>config['memory_working_limit_gib']*1024**3:
        raise MemoryError(f'CPU process RSS {rss/1024**3:.2f} GiB exceeds declared limit; stop and report')
    return {'rss_gib':rss/1024**3,'process_peak_rss_gib':getattr(info,'peak_wset',rss)/1024**3}


def summarize(out,manifest,results):
    rows=[{'case_id':case['case_id'],'candidate':c['id'],**r} for case in results for c in case['candidates'] for r in c['organs']]
    ctrows=[{'case_id':case['case_id'],'candidate':c['id'],**r} for case in results for c in case['candidates'] for r in c['ct_organs']]
    csv_rows(out/'organ_metrics.csv',rows);csv_rows(out/'ct_organ_metrics.csv',ctrows)
    summaries=[]
    for candidate in ['A','B','C']:
        for label in range(1,16):
            subset=[r for r in rows if r['candidate']==candidate and r['label']==label]
            present=[r for r in subset if r['source_present']]
            dice=[r['roundtrip_dice'] for r in present]
            summaries.append({'candidate':candidate,'label':label,'organ':LABELS[label],
                'present_cases':len(present),'absent_cases':len(subset)-len(present),
                'mean_dice':float(np.mean(dice)) if dice else None,'min_dice':min(dice) if dice else None,
                'worst_case':min(present,key=lambda r:r['roundtrip_dice'])['case_id'] if present else None,
                'mean_abs_target_volume_change_percent':float(np.mean([abs(r['target_volume_change_percent']) for r in present])) if present else None,
                'max_abs_target_volume_change_percent':max([abs(r['target_volume_change_percent']) for r in present],default=None),
                'target_disappearances':sum(r['disappeared_in_target'] for r in present),
                'roundtrip_disappearances':sum(r['disappeared_in_roundtrip'] for r in present)})
    csv_rows(out/'organ_summary.csv',summaries)
    anomalies=[]
    for r in rows:
        if r['source_absent'] or r['disappeared_in_target'] or r['disappeared_in_roundtrip']:
            anomalies.append(r)
    # Descriptive review priorities, not pass/fail thresholds.
    for c in ['A','B','C']:
        for label in [11,12]:
            selected=[r for r in rows if r['candidate']==c and r['label']==label and r['source_present']]
            if selected:
                anomalies.append({**min(selected,key=lambda r:r['roundtrip_dice']),'review_reason':'lowest observed adrenal roundtrip Dice; not a quality cutoff'})
    csv_rows(out/'review_cases.csv',anomalies)
    dump(out/'results.json',{'provenance':manifest,'cases':results,'organ_summary':summaries})
    lines=['# 完整扫描重采样保真度 CPU 小样本报告','',
           f'已完成 {len(results)} 例 × 3 候选。仅采样几何/图像保真度，不是模型分割性能；未冻结 spacing。',
           '','## 实际病例','', '|病例|原尺寸 xyz|原 spacing mm|选择依据|','|---|---|---|---|']
    for case in results:
        lines.append(f'|{case["case_id"]}|{case["source_shape"]}|{case["source_spacing"]}|{case["selection_reason"]}|')
    lines+=['','## 全部前景类别','',
            'Dice 均值仅对原始标签存在的病例计算；空类记 NA，单列缺失，不是正式评估空类规则。体积比较使用 affine 行列式。',
            '', '|候选|器官|存在/缺失|往返 Dice 均值 / 最低|目标体积绝对变化均值|目标/往返消失数|最低病例|',
            '|---|---|---|---|---|---|---|']
    for s in summaries:
        fmt=lambda x:'NA' if x is None else f'{x:.4f}'
        lines.append(f'|{s["candidate"]}|{s["label"]} {s["organ"]}|{s["present_cases"]}/{s["absent_cases"]}|{fmt(s["mean_dice"])} / {fmt(s["min_dice"])}|{fmt(s["mean_abs_target_volume_change_percent"])}%|{s["target_disappearances"]}/{s["roundtrip_disappearances"]}|{s["worst_case"]}|')
    lines+=['','## CT 细节与局部对比度','',
            'CT 使用真实强度缩放、降采样方向 Gaussian 预滤波及三线性插值。返回原空间再插值不额外滤波。下表包含整个往返处理的变化，不将其归因于单次降采样。',
            '', '|候选|前景 MAE HU 均值|原生 xyz 前景相邻梯度幅度比均值|', '|---|---|---|']
    for cid in ['A','B','C']:
        values=[c['ct_global'] for case in results for c in case['candidates'] if c['id']==cid]
        lines.append(f'|{cid}|{np.mean([v["original_foreground_mae_hu"] for v in values]):.2f}|{np.mean([v["foreground_gradient_abs_ratio_native_axes"] for v in values],axis=0).round(3).tolist()}|')
    lines+=['','局部对比度：器官原标签内均值减去外侧 2–5 mm 背景标签环均值；仅描述性指标，组织组成/增强程度影响其含义。逐器官 HU、标准差、对比度和 MAE 见 ct_organ_metrics.csv。显示窗 [-160,240] HU 仅用于图像；未对测量值截断或归一化。',
            '', '## 实现与复核','',
            '- 物理目标网格：RAS 正向、完整体素单元边界、ceil 尺寸、保持中心，边界总扩展小于一目标体素；最近边界复制。记录每个网格 affine、边界、逆映射所需原 affine。无网络 padding 或 mask。',
            '- 每例通过原网格 CT/标签恒等变换、独立头解析与 NiBabel affine 核对、原始强度样本缩放核对及输入文件大小/mtime/头信息前后复核。',
            '- 最近邻输出仅原有合法类别；原空间不存在、目标网格消失和往返消失分别报告。边界器官体积变化可能受有限网格扩展及边缘复制影响。',
            '- 叠加图使用相同原空间切面和毫米纵横比；肾上腺 ROI 仅用于评估显示，不参与模型输入或重采样裁剪。',
            '- 标签往返 Dice 无法证明 CT 纹理保留；薄层变厚存在真实细节损失，5 mm→3 mm 插值不增加真实解剖信息。',
            f'- 源码 Git：{manifest["git"]}；各源码哈希与 CPU 依赖见 results.json。',
            f'- 实际总耗时（含逐例读取/恒等/三候选测量/可视化）：{sum(r["elapsed_seconds"] for r in results)/60:.2f} min；进程峰值工作集：{max(c["memory"]["process_peak_rss_gib"] for r in results for c in r["candidates"]):.2f} GiB。不是 GPU 显存。',
            '', '## 可视化','']
    for case in results:
        lines.append(f'- {case["case_id"]}: '+', '.join(f'[{p}](visuals/{p})' for p in case['visuals']))
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/fidelity_pilot.json')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    data=args.data_root.resolve();out=args.output_dir.resolve()
    if out.is_relative_to(data) or data.is_relative_to(out):raise ValueError('output must be separate from raw data')
    if out.exists():raise ValueError('use a fresh output directory; refuse overwrite')
    if not 8<=len(config['cases'])<=12:raise ValueError('pilot requires 8..12 cases')
    if config['network_padding']:raise ValueError('network padding not authorized')
    expected={'schema_version':1,'status':'pilot_not_frozen','grid':'ceil_cell_coverage_centered_extension',
              'ct_interpolation':'trilinear','label_interpolation':'nearest','boundary':'nearest_edge_replication',
              'ct_downsample_antialias':'gaussian_sigma_max((target/source-1)/2,0)_source_voxels'}
    if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('unsupported pilot protocol')
    if config['candidates']!=[{'id':c,'spacing_ras_mm':s} for c,s in [('A',[1.5,1.5,3.]),('B',[2.,2.,3.]),('C',[2.,2.,5.])]]:
        raise ValueError('this pilot must retain the authorized A/B/C candidates')
    metadata=json.loads(args.metadata.read_text(encoding='utf-8'))
    records={r['case_id']:r for r in metadata['cases']}
    selected=[records[s['id']] for s in config['cases']]
    if len({r['case_id'] for r in selected})!=len(selected):raise ValueError('duplicate selection')
    out.mkdir(parents=True);(out/'visuals').mkdir()
    manifest={'utc_time':datetime.now(timezone.utc).isoformat(),'git':git_state(),
              'source_sha256':code_hashes(),'config':config,'config_sha256':sha256(args.config),
              'metadata_sha256':sha256(args.metadata),'platform':platform.platform(),
              'python':platform.python_version(),'device':'CPU only',
              'dependencies':{p:importlib.metadata.version(p) for p in ['numpy','scipy','nibabel','Pillow','psutil']}}
    dump(out/'manifest.json',manifest)
    results=[]
    for selection,record in zip(config['cases'],selected):
        start=time.perf_counter();case_id=selection['id'];header=record['image_header']
        max_target=max(math.prod(build_target_grid(header,c['spacing_ras_mm'])[0]) for c in config['candidates'])
        estimate=header['voxel_count']*20+max_target*16+512*1024**2
        if estimate>config['memory_working_limit_gib']*1024**3 or estimate*config['available_memory_safety_factor']>psutil.virtual_memory().available:
            raise MemoryError(f'{case_id}: conservative working estimate {estimate/1024**3:.2f}GiB exceeds resource budget; stop, no reduction')
        ip,lp=contained_path(data,record['image']),contained_path(data,record['label'])
        before=[fingerprint(p) for p in (ip,lp)]
        for observed,key in zip(before,['image_header','label_header']):
            for field in ['shape_native','affine_mm','scaling_effective']:
                if observed['header'][field]!=record[key][field]:raise ValueError(f'{case_id}: cached metadata changed: {field}')
        print(f'{case_id}: loading, estimated workspace {estimate/1024**3:.2f}GiB',flush=True)
        ct,labels,ih,lh,diagnostic=load_pair(ip,lp)
        ia,la=affine4(ih),affine4(lh)
        identity=resample(ct,ia,ct.shape,ia,1)
        err=max(float(np.max(np.abs(ct[sl]-identity[sl]))) for sl in chunks(ct.shape))
        del identity
        identity=resample(labels,la,labels.shape,la,0)
        equal=bool(np.array_equal(labels,identity));del identity
        if err>1e-4 or not equal:raise AssertionError(f'{case_id}: identity failed {err} {equal}')
        diagnostic.update(identity_ct_max_error_hu=err,identity_labels_exact=equal)
        spacing=ih['spacing_affine_mm'];specs=view_specs(labels,spacing)
        if ih['axis_codes']!='LAS':raise ValueError('review orientation currently requires audited LAS')
        captures={'original':capture(ct,labels,specs)}
        case={'case_id':case_id,'selection_reason':selection['reason'],'source_shape':list(ct.shape),
              'source_spacing':spacing,'source_image_affine_ras_mm':ia.tolist(),
              'source_label_affine_ras_mm':la.tolist(),'checks':diagnostic,'candidates':[]}
        direct_visuals=[]
        for candidate in config['candidates']:
            tick=time.perf_counter();cid=candidate['id']
            shape,ta,grid=build_target_grid(ih,candidate['spacing_ras_mm'])
            target_ct=resample(ct,ia,shape,ta,1,True,config['gaussian_truncate'])
            memory_check(config)
            target_label=resample(labels,la,shape,ta,0)
            direct_visuals.append(save_target_review(out/'visuals',case_id,cid,target_ct,target_label,candidate['spacing_ras_mm'],config['display_window_hu']))
            rt_ct=resample(target_ct,ta,ct.shape,ia,1);del target_ct
            rt_label=resample(target_label,ta,labels.shape,la,0)
            organs=label_metrics(labels,target_label,rt_label,la,ta,config['evaluation_chunk_slices'])
            ct_global,ct_organs=ct_metrics(ct,rt_ct,labels,spacing,config['local_contrast_ring_mm'],config['evaluation_chunk_slices'])
            captures[cid]=capture(rt_ct,rt_label,specs)
            result={'id':cid,'grid':grid,'antialias_sigma_native_voxels':antialias_sigmas(ia,ta).tolist(),
                    'organs':organs,'ct_global':ct_global,'ct_organs':ct_organs,
                    'elapsed_seconds':time.perf_counter()-tick,'memory':memory_check(config)}
            case['candidates'].append(result)
            dump(out/f'{case_id}.json',case)
            print(f'{case_id} {cid}: {result["elapsed_seconds"]:.1f}s, adrenal Dice '+str([round(r['roundtrip_dice'],4) if r['source_present'] else None for r in organs[10:12]]),flush=True)
            del target_label,rt_label,rt_ct;gc.collect()
        case['visuals']=save_review(out/'visuals',case_id,specs,captures,spacing,ia,config['display_window_hu'])+direct_visuals
        after=[fingerprint(p) for p in (ip,lp)]
        if before!=after:raise AssertionError('original source file metadata changed during processing')
        case['original_files_unchanged_stat_and_header']=True
        case['source_fingerprints']=before
        case['elapsed_seconds']=time.perf_counter()-start
        dump(out/f'{case_id}.json',case);results.append(case)
        print(f'{case_id}: completed {case["elapsed_seconds"]:.1f}s',flush=True)
        del ct,labels,captures;gc.collect()
    summarize(out,manifest,results)
    print(f'COMPLETE: {len(results)} cases; {out}',flush=True)
    return 0

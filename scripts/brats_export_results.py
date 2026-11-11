"""Export the committed logs of finished BraTS region runs as plain tables (read only).

把若干 run 目录中已提交的日志（metrics.jsonl、validation/ 逐病例记录、run.json）整理成纯文本表格，
供人或其他 AI 不读原始日志也能分析。不读 checkpoint，不推理，不改动 runs/。

输出（全部写到 --output 目录，已存在的同名文件会被拒绝覆盖）：
  runs.csv          每个 run 一行：实验标签、折、代码版本、配置要点、参数量、轮数、最佳轮与 Dice、最后 5 轮平均等
  epochs.csv        每个 run 每轮一行：训练损失各项与软 Dice、验证 Dice（最终/粗预测，三区域）、验证损失、早停计数、用时、显存
  cases_epochs.csv  每个 run 每轮每个验证病例一行：最终与粗预测的三区域 Dice、预测/真值/正确体素、层级违例、损失各项
  train_steps.csv   每个训练步一行：病例、损失各项、软 Dice、学习率、用时、显存、增强参数
  comparisons.json  按折配对的病例级比较（同一验证病例、各取最佳轮与最后 5 轮平均）：均值差、t 检验、Wilcoxon、bootstrap
用法：python scripts/brats_export_results.py --run E1:0=runs/<...>_fold0 ... --output results/<名称> [--pair E1,A3]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REGIONS = ('WT', 'TC', 'ET')


def parse_run(text):
    head, sep, path = text.partition('=')
    label, colon, fold = head.partition(':')
    if not sep or not colon or not label or not fold.isdigit():
        raise argparse.ArgumentTypeError('--run must be LABEL:FOLD=RUN_DIR')
    return label, int(fold), Path(path)


def rows_of(run_dir):
    with open(run_dir / 'metrics.jsonl', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def ledger(run_dir, step):
    """该 run 在某一步的逐病例验证记录；跳过每轮的完成标记文件。"""
    result = {}
    for path in (run_dir / 'validation' / f'step_{step:08d}').glob('*.json'):
        row = json.loads(path.read_text())
        if 'result' in row:
            result[row['case_id']] = row['result']
    return result


def fmt(value):
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.6g}'
    return value


def write_csv(path, header, rows):
    with path.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in rows:
            writer.writerow([fmt(row.get(k)) for k in header])


def case_mean(dice):
    valid = [d for d in dice if d is not None]
    return sum(valid) / len(valid) if valid else None


def export(runs, output, pairs):
    output.mkdir(parents=True, exist_ok=True)
    run_rows, epoch_rows, case_rows, step_rows = [], [], [], []
    case_scores = {}  # (label, fold) -> {'best': {case: [3]}, 'last5': {case: [3]}}
    for label, fold, run_dir in runs:
        identity = json.loads((run_dir / 'run.json').read_text())['identity']
        dev, train_epoch, steps = [], {}, []
        for row in rows_of(run_dir):
            phase = row.get('phase')
            if phase == 'train':
                steps.append(row)
            elif phase == 'train_epoch':
                train_epoch[row['epoch']] = row
            elif phase == 'internal_dev':
                dev.append(row)
        epoch_seconds, epoch_peak = {}, {}
        for row in steps:
            e = row['epoch']
            epoch_seconds[e] = epoch_seconds.get(e, 0.0) + row['step_seconds']
            peak = row['memory']['peak_allocated_bytes']
            if peak is not None:
                epoch_peak[e] = max(epoch_peak.get(e, 0), peak)
            aug = row.get('augmentation') or {}
            step_rows.append(dict(label=label, fold=fold, global_step=row['global_step'], epoch=e, case_id=row['case_id'],
                                  total_loss=row['total_loss'], final_seg=row['final']['segmentation'],
                                  final_bce=row['final']['bce'], final_dice_loss=row['final']['dice_loss'],
                                  coarse_seg=row['coarse']['segmentation'], hierarchy=row['hierarchy'],
                                  **{f'soft_dice_{r}': v for r, v in zip(REGIONS, row['soft_dice_per_organ'])},
                                  lr=row['lr'], step_seconds=row['step_seconds'],
                                  peak_gib=None if peak is None else peak / 2**30,
                                  shape='x'.join(map(str, row['shape'])),
                                  flip=''.join('1' if f else '0' for f in aug.get('flip', [])),
                                  scale=';'.join(f'{v:.4f}' for v in aug.get('scale', [])),
                                  shift=';'.join(f'{v:.4f}' for v in aug.get('shift', []))))
        for row in dev:
            e = row['epoch']
            te = (train_epoch.get(e) or {}).get('epoch_metrics') or {}
            fin, coa = row['epoch_metrics']['final'], row['epoch_metrics']['coarse']
            es = row.get('early_stopping') or {}
            epoch_rows.append(dict(
                label=label, fold=fold, epoch=e, global_step=row['global_step'],
                train_total_loss=te.get('total_loss'), train_final_loss=(te.get('final') or {}).get('loss'),
                train_coarse_loss=(te.get('coarse') or {}).get('loss'), train_hierarchy=te.get('hierarchy'),
                **{f'train_soft_dice_{r}': v for r, v in zip(REGIONS, (te.get('final') or {}).get('soft_per_region', [None] * 3))},
                val_dice=row['metrics']['mean_case_dice'],
                **{f'val_dice_{o["name"]}': o['mean_dice'] for o in row['metrics']['organs']},
                val_coarse_dice=coa['hard']['mean_case_dice'],
                **{f'val_coarse_dice_{o["name"]}': o['mean_dice'] for o in coa['hard']['organs']},
                val_total_loss=row['epoch_metrics']['total_loss'], val_final_loss=fin['loss'], val_coarse_loss=coa['loss'],
                val_hierarchy=row['epoch_metrics']['hierarchy'],
                **{f'val_soft_dice_{r}': v for r, v in zip(REGIONS, fin['soft_per_region'])},
                best_improved=row.get('best_dev_improved'), es_no_improvement=es.get('no_improvement_count'),
                es_best_epoch=es.get('best_epoch'), es_stopped=es.get('stopped'),
                train_seconds=epoch_seconds.get(e), peak_gib=epoch_peak[e] / 2**30 if e in epoch_peak else None))
            for case, res in ledger(run_dir, row['global_step']).items():
                diag = res['region_diagnostic']
                coarse_hard = diag['coarse']['hard']['organs']
                case_rows.append(dict(
                    label=label, fold=fold, epoch=e, case_id=case,
                    **{f'dice_{o["name"]}': o['dice'] for o in res['organs']},
                    **{f'pred_{o["name"]}': o['predicted_voxels'] for o in res['organs']},
                    **{f'gt_{o["name"]}': o['gt_voxels'] for o in res['organs']},
                    **{f'tp_{o["name"]}': o['true_positive'] for o in res['organs']},
                    **{f'coarse_dice_{o["name"]}': o['dice'] for o in coarse_hard},
                    et_outside_tc=res['hierarchy_violations']['ET_outside_TC'],
                    tc_outside_wt=res['hierarchy_violations']['TC_outside_WT'],
                    final_bce=diag['final']['bce'], final_dice_loss=diag['final']['dice_loss'],
                    coarse_bce=diag['coarse']['bce'], coarse_dice_loss=diag['coarse']['dice_loss'],
                    hierarchy=diag['hierarchy'], total_loss=diag['total_loss'],
                    **{f'soft_dice_{r}': v for r, v in zip(REGIONS, diag['final']['soft_dice'])}))
        if not dev:
            continue
        best = max(dev, key=lambda r: r['metrics']['mean_case_dice'])
        last5 = dev[-5:]
        per_epoch = {}
        for r in case_rows:
            if r['label'] == label and r['fold'] == fold:
                per_epoch.setdefault(r['epoch'], {})[r['case_id']] = [r[f'dice_{x}'] for x in REGIONS]
        cases = per_epoch[best['epoch']]
        last = {c: [np.mean([per_epoch[d['epoch']][c][i] for d in last5 if per_epoch[d['epoch']][c][i] is not None])
                    if any(per_epoch[d['epoch']][c][i] is not None for d in last5) else None for i in range(3)]
                for c in cases}
        case_scores[(label, fold)] = dict(best=cases, last5=last)
        model, loss, training = identity['model'], identity['loss'], identity['training']
        stopped = (dev[-1].get('early_stopping') or {}).get('stopped')
        run_rows.append(dict(
            label=label, fold=fold, run_dir=run_dir.name, commit=identity['provenance']['git']['commit'][:7],
            variant=model.get('variant') or 'full', graph=model.get('graph') or 'complete', rounds=model['rounds'],
            lambda_hier=loss.get('lambda_hier', 0.0), lambda_c=loss['lambda_c'], max_epochs=training['max_epochs'],
            train_cases=len(identity['data']['split']['train']), val_cases=len(identity['data']['split']['dev']),
            epochs_run=dev[-1]['epoch'], status='early_stopped' if stopped else 'incomplete',
            best_epoch=best['epoch'], best_dice=best['metrics']['mean_case_dice'],
            **{f'best_{o["name"]}': o['mean_dice'] for o in best['metrics']['organs']},
            last5_mean_dice=float(np.mean([d['metrics']['mean_case_dice'] for d in last5])),
            train_hours=sum(r['step_seconds'] for r in steps) / 3600))
    header = lambda rows: list(dict.fromkeys(k for r in rows for k in r))
    write_csv(output / 'runs.csv', header(run_rows), run_rows)
    write_csv(output / 'epochs.csv', header(epoch_rows), epoch_rows)
    write_csv(output / 'cases_epochs.csv', header(case_rows), case_rows)
    write_csv(output / 'train_steps.csv', header(step_rows), step_rows)
    comparisons = compare(case_scores, pairs, {(r['label'], r['fold']): r['status'] for r in run_rows})
    with (output / 'comparisons.json').open('x') as stream:
        json.dump(comparisons, stream, indent=1)
    return run_rows, comparisons


def paired(diffs, rng):
    from scipy import stats
    d = np.asarray(diffs, float)
    boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return dict(n=len(d), mean=float(d.mean()), median=float(np.median(d)), sd=float(d.std(ddof=1)),
                ci95_bootstrap=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                t_p=float(stats.ttest_1samp(d, 0).pvalue), wilcoxon_p=float(stats.wilcoxon(d).pvalue) if np.any(d) else 1.0,
                better=int((d > 0).sum()), worse=int((d < 0).sum()))


def compare(scores, pairs, status):
    """按 (A, B) 对比同一折同一批验证病例：B − A，病例平均 Dice 与各区域；只用两边都完成的折。"""
    rng = np.random.default_rng(0)
    result = {}
    for a, b in pairs:
        folds = sorted(f for (l, f) in scores if l == a and (b, f) in scores
                       and status[(a, f)] == status[(b, f)] == 'early_stopped')
        entry = dict(definition=f'{b} minus {a}, same validation cases, each run at its best epoch or the mean of its last 5 validations',
                     folds=folds)
        for which in ('best', 'last5'):
            pooled = {k: [] for k in ('mean', *REGIONS)}
            per_fold = {}
            for f in folds:
                sa, sb = scores[(a, f)][which], scores[(b, f)][which]
                diffs = []
                for case in sorted(sa):
                    ma, mb = case_mean(sa[case]), case_mean(sb[case])
                    if ma is not None and mb is not None:
                        diffs.append(mb - ma)
                    for i, r in enumerate(REGIONS):
                        if sa[case][i] is not None and sb[case][i] is not None:
                            pooled[r].append(sb[case][i] - sa[case][i])
                pooled['mean'] += diffs
                per_fold[f] = paired(diffs, rng)
            entry[which] = dict(pooled={k: paired(v, rng) for k, v in pooled.items() if v}, per_fold=per_fold)
        result[f'{b}_vs_{a}'] = entry
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=parse_run, action='append', required=True, help='LABEL:FOLD=RUN_DIR')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pair', action='append', default=[], help='A,B: compare B against A per case')
    args = parser.parse_args(argv)
    try:
        pairs = [tuple(p.split(',')) for p in args.pair]
        if any(len(p) != 2 for p in pairs):
            raise ValueError('--pair must be A,B')
        runs, _ = export(args.run, args.output, pairs)
        print(f'{len(runs)} runs exported to {args.output}')
    except (ValueError, OSError, KeyError) as exc:
        print('export stopped: ' + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Compact text report of finished or running BraTS runs: last-validation Dice, empty-ET errors, gate, paired tests."""
# 用法：
#   python scripts/brats_report.py --runs-dir runs --prefix brats2018_p2_            # 每个 run 一行
#   python scripts/brats_report.py --runs-dir runs --prefix brats2018_p2_ --compare r2_nodes r4_full [--compare ...]
# 只读 runs/ 下的 run.json、metrics.jsonl 与最近一次验证的逐病例记录，不加载模型、不写任何文件。
# 每个 run 取"最近一次验证"；--compare A B 在两个方案都已跑满的折上做逐病例配对（B − A）。
# "无 ET 误报"指真值没有 ET 而预测了 ET 的病例数；"小 ET"指真值 ET 在 1–999 体素之间。
import argparse
import json
import math
from pathlib import Path
import re
import statistics


def load_run(run_dir):
    try:
        identity = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))['identity']
        steps = sorted(p for p in (run_dir / 'validation').glob('step_*') if p.is_dir())
    except (OSError, ValueError, KeyError):
        return None
    if not steps:
        return None
    training = identity['training']
    cases = {}
    for path in steps[-1].glob('*.json'):
        row = json.loads(path.read_text(encoding='utf-8'))
        if 'result' not in row:
            continue
        organs = row['result']['organs']
        cases[row['case_id']] = dict(dice=[o['dice'] for o in organs], pred=[o['predicted_voxels'] for o in organs],
                                     gt=[o['gt_voxels'] for o in organs],
                                     readout=row['result'].get('region_diagnostic', {}).get('containment_logits'))
    step = int(steps[-1].name.split('_')[1])
    train_cases = len(identity['data']['split']['train'])
    epoch = step // train_cases
    return dict(name=run_dir.name, epoch=epoch, max_epochs=training['max_epochs'], cases=cases,
                complete=epoch >= training['max_epochs'] and len(cases) == len(identity['data']['split']['dev']))


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float('nan')


def case_mean(case):
    return mean(case['dice'])


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x > -700 else 0.0


def existence(case):
    """存在性门控的 ET 存在概率 pi_TC * pi_ET；没有该读出时为 None。"""
    readout = case['readout']
    if not readout or not isinstance(readout[0], list):
        return None
    return sigmoid(readout[0][1]) * sigmoid(readout[1][1])


def describe(run):
    cases = list(run['cases'].values())
    no_et = [c for c in cases if c['gt'][2] == 0]
    small = [c for c in cases if 0 < c['gt'][2] < 1000]
    text = (f"{run['name']:44s} ep {run['epoch']:3d}/{run['max_epochs']:<3d} n={len(cases):3d} "
            f"Dice {mean(case_mean(c) for c in cases):.4f}  WT {mean(c['dice'][0] for c in cases):.4f} "
            f"TC {mean(c['dice'][1] for c in cases):.4f} ET {mean(c['dice'][2] for c in cases):.4f} | "
            f"no-ET FP {sum(c['pred'][2] > 0 for c in no_et)}/{len(no_et)} | small-ET Dice {mean(c['dice'][2] for c in small):.3f} (n={len(small)})")
    gate = [(existence(c), c['gt'][2]) for c in cases if existence(c) is not None]
    if gate:
        absent = sorted(round(p, 2) for p, gt in gate if gt == 0)
        present = [p for p, gt in gate if gt > 0]
        text += (f" | gate: pi_ET on no-ET cases {absent}; with-ET cases closed (pi<0.5) {sum(p < .5 for p in present)}/{len(present)}, "
                 f"min pi {min(present):.2f}")
    return text


def paired(label, before, after):
    """逐病例配对：after − before 的均值、标准误与 t 检验 p 值（正态近似用于 n 很大时的展示）。"""
    differences = [b - a for a, b in zip(before, after) if not (math.isnan(a) or math.isnan(b))]
    n = len(differences)
    if n < 3:
        return f'    {label:16s} n={n} (too few)'
    average, sd = statistics.fmean(differences), statistics.stdev(differences)
    se = sd / math.sqrt(n)
    t = average / se if se else float('inf')
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), n - 1)
    except ImportError:
        p = math.erfc(abs(t) / math.sqrt(2))
    return f'    {label:16s} n={n:3d}  diff {average:+.4f}  se {se:.4f}  sd {sd:.4f}  p {p:.3f}  better/worse {sum(d > 0 for d in differences)}/{sum(d < 0 for d in differences)}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', type=Path, required=True)
    parser.add_argument('--prefix', default='brats2018_p2_', help='run directory prefix, e.g. brats2018_p2_')
    parser.add_argument('--compare', nargs=2, action='append', default=[], metavar=('BEFORE', 'AFTER'),
                        help='config names without prefix and fold, e.g. r2_nodes r4_full')
    args = parser.parse_args(argv)
    runs = {}
    for path in sorted(args.runs_dir.glob(f'{args.prefix}*_fold[0-9]')):
        run = load_run(path)
        if run is not None:
            runs[path.name] = run
            print(describe(run))
    for before, after in args.compare:
        folds = [int(re.search(r'_fold(\d)$', name).group(1)) for name in runs
                 if name == f'{args.prefix}{before}_fold{name[-1]}' and runs[name]['complete']
                 and runs.get(f'{args.prefix}{after}_fold{name[-1]}', {}).get('complete')]
        print(f'\n{after} minus {before}: complete folds {sorted(folds)}')
        a_cases, b_cases = {}, {}
        for fold in folds:
            a_cases.update(runs[f'{args.prefix}{before}_fold{fold}']['cases'])
            b_cases.update(runs[f'{args.prefix}{after}_fold{fold}']['cases'])
        common = sorted(set(a_cases) & set(b_cases))
        if not common:
            continue
        value = lambda case, index: float('nan') if case['dice'][index] is None else case['dice'][index]
        print(paired('mean Dice', [case_mean(a_cases[c]) for c in common], [case_mean(b_cases[c]) for c in common]))
        for index, region in enumerate(('WT', 'TC', 'ET')):
            print(paired(region, [value(a_cases[c], index) for c in common], [value(b_cases[c], index) for c in common]))
        subgroup = [c for c in common if a_cases[c]['gt'][2] < 1000]
        print(paired('ET<1000 or none', [case_mean(a_cases[c]) for c in subgroup], [case_mean(b_cases[c]) for c in subgroup]))
        rest = [c for c in common if a_cases[c]['gt'][2] >= 1000]
        print(paired('ET>=1000', [case_mean(a_cases[c]) for c in rest], [case_mean(b_cases[c]) for c in rest]))
        empty = [c for c in common if a_cases[c]['gt'][2] == 0]
        print(f"    no-ET false positives: {before} {sum(a_cases[c]['pred'][2] > 0 for c in empty)}/{len(empty)}, "
              f"{after} {sum(b_cases[c]['pred'][2] > 0 for c in empty)}/{len(empty)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

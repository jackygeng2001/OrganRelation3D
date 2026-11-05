"""Read-only side-by-side of two BraTS region runs (e.g. FP32 vs bf16 arms).

Prints per-epoch internal-dev hard Dice (mean/WT/TC/ET) for both runs, their
difference, and mean training step time. Never writes to either run.
"""
import argparse
import json
from pathlib import Path
import statistics


def load(run_dir):
    rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    dev = {}
    for row in rows:
        if row.get('phase') == 'internal_dev':
            metrics = row['metrics']
            dev[row['epoch']] = [metrics['mean_case_dice']] + [r['mean_dice'] for r in metrics['organs']]
    steps = [r['step_seconds'] for r in rows if r.get('phase') == 'train']
    identity = json.loads((run_dir / 'run.json').read_text()) if (run_dir / 'run.json').exists() else {}
    return dev, steps, identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--tolerance', type=float, default=0.01,
                        help='Mean-Dice difference treated as no practical change (default 0.01)')
    args = parser.parse_args(argv)
    a, a_steps, _ = load(args.first)
    b, b_steps, _ = load(args.second)
    print(f'A = {args.first}\nB = {args.second}\n')
    print(f"{'epoch':>5} | {'A mean':>7} {'WT':>6} {'TC':>6} {'ET':>6} | {'B mean':>7} {'WT':>6} {'TC':>6} {'ET':>6} | {'B-A':>7}")
    common = sorted(set(a) & set(b))
    for epoch in sorted(set(a) | set(b)):
        fa = ' '.join(f'{v:6.4f}' if v is not None else '  None' for v in a.get(epoch, [None] * 4))
        fb = ' '.join(f'{v:6.4f}' if v is not None else '  None' for v in b.get(epoch, [None] * 4))
        diff = f'{b[epoch][0] - a[epoch][0]:+7.4f}' if epoch in common else '      -'
        print(f'{epoch:>5} | {fa} | {fb} | {diff}')
    for name, steps in (('A', a_steps), ('B', b_steps)):
        if steps:
            print(f'{name}: {len(steps)} training steps, median step {statistics.median(steps):.2f} s')
    if common:
        last = common[-1]
        window = common[-5:]
        best = max(a[e][0] for e in common), max(b[e][0] for e in common)
        mean_gap = statistics.fmean(b[e][0] - a[e][0] for e in window)
        print(f'\nlast common epoch {last}: B-A = {b[last][0] - a[last][0]:+.4f}')
        print(f'mean B-A over epochs {window[0]}-{window[-1]}: {mean_gap:+.4f}')
        print(f'best mean Dice: A {best[0]:.4f}, B {best[1]:.4f} (B-A {best[1] - best[0]:+.4f})')
        verdict = 'within' if abs(mean_gap) <= args.tolerance and abs(best[1] - best[0]) <= args.tolerance else 'OUTSIDE'
        print(f'{verdict} the +/-{args.tolerance} tolerance (one run per arm: rules out large changes only)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

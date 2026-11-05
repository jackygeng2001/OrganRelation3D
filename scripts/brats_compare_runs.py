"""Read-only side-by-side of BraTS region runs (e.g. FP32 / conv-only bf16 / mixed bf16 arms).

The first run is the reference. Prints per-epoch internal-dev hard Dice
(mean/WT/TC/ET) for every run, each run's difference from the reference, and
training step time (median, mean, and summed per completed epoch; validation
time is not logged). Never writes to any run.
"""
import argparse
import json
from pathlib import Path
import statistics


def load(run_dir):
    rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    dev, epoch_seconds = {}, {}
    for row in rows:
        if row.get('phase') == 'internal_dev':
            metrics = row['metrics']
            dev[row['epoch']] = [metrics['mean_case_dice']] + [r['mean_dice'] for r in metrics['organs']]
        elif row.get('phase') == 'train':
            epoch_seconds[row['epoch']] = epoch_seconds.get(row['epoch'], 0.0) + row['step_seconds']
    steps = [r['step_seconds'] for r in rows if r.get('phase') == 'train']
    return dev, steps, epoch_seconds


def cells(values):
    return ' '.join(f'{v:6.4f}' if v is not None else '  None' for v in values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+', help='Two or more run directories; the first is the reference')
    parser.add_argument('--tolerance', type=float, default=0.01,
                        help='Mean-Dice difference treated as no practical change (default 0.01)')
    args = parser.parse_args(argv)
    if len(args.runs) < 2:
        parser.error('give at least two run directories')
    loaded = [load(run) for run in args.runs]
    names = [chr(ord('A') + i) for i in range(len(args.runs))]
    for name, run in zip(names, args.runs):
        print(f'{name} = {run}')
    print()
    header = ' | '.join(f"{name + ' mean':>7} {'WT':>6} {'TC':>6} {'ET':>6}" for name in names)
    diffs = ' '.join(f"{name + '-A':>7}" for name in names[1:])
    print(f"{'epoch':>5} | {header} | {diffs}")
    reference = loaded[0][0]
    for epoch in sorted(set().union(*(dev for dev, _, _ in loaded))):
        row = ' | '.join(cells(dev.get(epoch, [None] * 4)) for dev, _, _ in loaded)
        gaps = ' '.join(f'{dev[epoch][0] - reference[epoch][0]:+7.4f}' if epoch in dev and epoch in reference
                        else '      -' for dev, _, _ in loaded[1:])
        print(f'{epoch:>5} | {row} | {gaps}')
    print()
    base = statistics.median(loaded[0][1]) if loaded[0][1] else None
    for name, (_, steps, epochs) in zip(names, loaded):
        if not steps:
            continue
        median = statistics.median(steps)
        speed = f', {base / median:.2f}x the reference' if base else ''
        per_epoch = ', '.join(f'{e}: {s / 60:.1f} min' for e, s in sorted(epochs.items()))
        print(f'{name}: {len(steps)} steps, median {median:.2f} s, mean {statistics.fmean(steps):.2f} s{speed}; '
              f'summed step time per epoch {per_epoch}')
    for name, (dev, _, _) in zip(names[1:], loaded[1:]):
        common = sorted(set(reference) & set(dev))
        if not common:
            continue
        window = common[-5:]
        mean_gap = statistics.fmean(dev[e][0] - reference[e][0] for e in window)
        best = max(reference[e][0] for e in common), max(dev[e][0] for e in common)
        verdict = 'within' if abs(mean_gap) <= args.tolerance and abs(best[1] - best[0]) <= args.tolerance else 'OUTSIDE'
        print(f'\n{name} vs A over epochs {window[0]}-{window[-1]}: mean {name}-A {mean_gap:+.4f}; '
              f'best mean Dice A {best[0]:.4f}, {name} {best[1]:.4f} ({best[1] - best[0]:+.4f})')
        print(f'{verdict} the +/-{args.tolerance} tolerance (one run per arm: rules out large changes only)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

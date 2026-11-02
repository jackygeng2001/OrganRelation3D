"""Create immutable, local year-specific patient folds from legacy training metadata."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from organ_relation.data.brats import index_cases, make_folds, validate_partition
from organ_relation.training.state import digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cohort', choices=('brats2018','brats2019','brats2020'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.resolve().is_relative_to(args.source.expanduser().resolve()):
            raise ValueError('fold output must be outside original source')
        manifest = index_cases(args.source, cohort=args.cohort, partition='training')
        splits = make_folds(manifest, seed=args.seed)
        paths = [args.output_dir/f'{args.cohort}_fold_{i}.local.json' for i in range(5)]
        if any(p.exists() for p in paths):
            raise ValueError('refusing to overwrite an established fold')
        for split in splits:
            validate_partition(manifest, split)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path, split in zip(paths, splits):
            with path.open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(split, indent=2)+'\n')
        print(json.dumps(dict(cohort=args.cohort, manifest_hash=digest(manifest),
            folds=[dict(path=str(p), train=len(s['train']), dev=len(s['dev']),
                        split_hash=digest(s)) for p,s in zip(paths,splits)]), indent=2))
    except (ValueError, OSError) as exc:
        print(f'BraTS folds stopped: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

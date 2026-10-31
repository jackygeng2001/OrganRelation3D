"""Prepare a local patient-disjoint pilot split using ZIP/directory metadata only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from organ_relation.data.brats import index_cases, make_pilot_partition, validate_partition
from organ_relation.training.state import digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-subjects', type=int, required=True)
    parser.add_argument('--dev-subjects', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--exclude-subject', action='append', default=[])
    args = parser.parse_args(argv)
    try:
        manifest = index_cases(args.source, cohort='brats2023_gli')
        split = make_pilot_partition(manifest, train_subjects=args.train_subjects,
            dev_subjects=args.dev_subjects, seed=args.seed, excluded_subjects=args.exclude_subject)
        validate_partition(manifest, split)
        if args.output.resolve().is_relative_to(args.source.expanduser().resolve()):
            raise ValueError('split output must be outside the read-only source')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:  # Never replace an established split.
            stream.write(json.dumps(split, indent=2)+'\n')
        print(json.dumps(dict(split=str(args.output), split_hash=digest(split),
            manifest_hash=split['manifest_hash'], train=split['train'], dev=split['dev'],
            unused_scans=len(split['unused']), excluded_subjects=split['excluded_subjects']), indent=2))
    except (ValueError, OSError) as exc:
        print(f'Pilot split stopped: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

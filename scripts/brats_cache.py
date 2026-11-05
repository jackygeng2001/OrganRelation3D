"""Pre-build the exact .npy loader cache for every case of one cohort/config.

Each case is read once from the read-only source with the config's
preprocessing (normalization, optional spatial_crop) and stored under
--data-cache. Existing complete entries are verified and reused. Training
then passes the same --data-cache; it reads bitwise-identical arrays.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None):
    from organ_relation.data.brats import BraTSDataset, index_cases
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--source-partition', choices=('all', 'training'), default='all')
    parser.add_argument('--data-cache', type=Path, required=True)
    parser.add_argument('--split', type=Path, help='Optional: only its train+dev cases')
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text())
    manifest = index_cases(args.source, cohort=args.cohort, partition=args.source_partition)
    records = manifest['records']
    if args.split:
        split = json.loads(args.split.read_text())
        wanted = set(split['train']) | set(split['dev'])
        records = [r for r in records if r['case_id'] in wanted]
    dataset = BraTSDataset(dict(manifest, records=records), **config['preprocessing'], cache_dir=args.data_cache)
    started, failed = time.perf_counter(), []
    for index, record in enumerate(records):
        try:
            sample = dataset[index]
        except ValueError as exc:
            failed.append(record['case_id'])
            print(f"[{index + 1}/{len(records)}] {record['case_id']} FAILED: {exc}", flush=True)
            continue
        print(f"[{index + 1}/{len(records)}] {record['case_id']} {list(sample.image.shape)}", flush=True)
    size = sum(p.stat().st_size for p in (args.data_cache.expanduser() / args.cohort).rglob('*') if p.is_file())
    print(json.dumps(dict(cases=len(records), failed=failed, cache_gib=round(size / 2**30, 2),
                          seconds=round(time.perf_counter() - started, 1)), indent=2))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())

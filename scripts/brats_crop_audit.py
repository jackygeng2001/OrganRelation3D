"""Read-only audit of the image-only nonzero brain box for every BraTS case.

For each case: the tight box of nonzero voxels across T1/T1ce/T2/FLAIR, whether
all labeled tumor voxels lie inside it, the tumor-to-box-edge clearance, and
the box size after optional margin/multiple-of rounding. Labels never define
the box; they are only checked. No run, cache or model is created.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None):
    from organ_relation.data.brats import BraTSDataset, expand_box, index_cases
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--source-partition', choices=('all', 'training'), default='all')
    parser.add_argument('--split', type=Path, help='Optional split JSON; audit only its train+dev cases')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, required=True, help='JSON lines, one row per case')
    parser.add_argument('--roundings', type=int, nargs='+', default=(1, 8, 16))
    args = parser.parse_args(argv)

    manifest = index_cases(args.source, cohort=args.cohort, partition=args.source_partition)
    records = manifest['records']
    if args.split:
        split = json.loads(args.split.read_text())
        wanted = set(split['train']) | set(split['dev'])
        records = [r for r in records if r['case_id'] in wanted]
    records = records[:args.limit] if args.limit else records
    dataset = BraTSDataset(dict(manifest, records=records), normalization='nonzero_zscore',
                           normalization_epsilon=1e-8,
                           spatial_crop=dict(kind='nonzero_bbox', margin=0, multiple_of=1))
    rows = []
    with args.output.open('w') as handle:
        for index, record in enumerate(records):
            started = time.perf_counter()
            try:
                sample = dataset[index]
            except ValueError as exc:
                row = dict(case_id=record['case_id'], status='failed', error=str(exc))
            else:
                crop = sample.metadata['spatial_crop']
                shape, start, stop = crop['original_shape_dhw'], crop['start_dhw'], crop['stop_dhw']
                tumor = sample.label.numpy() > 0
                clearance = None
                if tumor.any():
                    clearance = []
                    for axis in range(3):
                        hit = np.flatnonzero(tumor.any(axis=tuple(a for a in range(3) if a != axis)))
                        clearance.append(min(int(hit[0]), tumor.shape[axis] - int(hit[-1]) - 1))
                rounded = {}
                for multiple in args.roundings:
                    s, e = expand_box(list(zip(start, stop)), shape, margin=0, multiple_of=multiple)
                    rounded[str(multiple)] = [b - a for a, b in zip(s, e)]
                row = dict(case_id=record['case_id'], status='passed', original_shape_dhw=shape,
                           start_dhw=start, stop_dhw=stop, tight_shape_dhw=list(sample.label.shape),
                           voxel_fraction=float(np.prod(sample.label.shape) / np.prod(shape)),
                           tumor_voxels=int(tumor.sum()), tumor_min_clearance_dhw=clearance,
                           rounded_shape_dhw=rounded)
            row['load_seconds'] = time.perf_counter() - started
            rows.append(row)
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            print(f"[{index + 1}/{len(records)}] {row['case_id']} {row['status']} "
                  f"{row.get('tight_shape_dhw')} {row['load_seconds']:.2f}s", flush=True)

    passed = [r for r in rows if r['status'] == 'passed']
    summary = dict(cases=len(rows), passed=len(passed), failed=[r['case_id'] for r in rows if r['status'] != 'passed'],
                   load_seconds_mean=float(np.mean([r['load_seconds'] for r in rows])))
    if passed:
        fractions = np.array([r['voxel_fraction'] for r in passed])
        summary['voxel_fraction'] = dict(mean=float(fractions.mean()), median=float(np.median(fractions)),
                                         min=float(fractions.min()), max=float(fractions.max()))
        summary['tight_shape_max_dhw'] = np.max([r['tight_shape_dhw'] for r in passed], axis=0).tolist()
        summary['tight_shape_median_dhw'] = np.median([r['tight_shape_dhw'] for r in passed], axis=0).tolist()
        clear = [min(r['tumor_min_clearance_dhw']) for r in passed if r['tumor_min_clearance_dhw']]
        summary['tumor_min_clearance_voxels'] = min(clear) if clear else None
        for multiple in args.roundings:
            shapes = [tuple(r['rounded_shape_dhw'][str(multiple)]) for r in passed]
            original = np.prod(passed[0]['original_shape_dhw'])
            summary[f'multiple_of_{multiple}'] = dict(
                distinct_shapes=len(set(shapes)),
                voxel_fraction_mean=float(np.mean([np.prod(s) / original for s in shapes])))
    print(json.dumps(summary, indent=2))
    return 0 if not summary['failed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

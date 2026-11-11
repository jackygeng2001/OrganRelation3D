"""结果导出：从合成的 run 目录读出表格与配对比较，只读不改（CPU）。"""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from organ_relation.provenance import sha256

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('brats_export_results', ROOT / 'scripts/brats_export_results.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)
CASES = ['c1', 'c2', 'c3']


def make_run(root, name, offset, epochs=6):
    run = root / name
    run.mkdir()
    identity = dict(model=dict(rounds=2), loss=dict(lambda_c=.5, lambda_hier=1.0), training=dict(max_epochs=150),
                    data=dict(split=dict(train=['t'] * 4, dev=CASES)), provenance=dict(git=dict(commit='abcdef1234')))
    (run / 'run.json').write_text(json.dumps(dict(identity=identity, run_id=name)))
    rows = []
    for epoch in range(1, epochs + 1):
        step = epoch * 2
        for s in (step - 1, step):
            rows.append(dict(phase='train', epoch=epoch, global_step=s, case_id='t', total_loss=1.0, lr=3e-4,
                             final=dict(segmentation=.6, bce=.1, dice_loss=.5), coarse=dict(segmentation=.4),
                             hierarchy=0.0, soft_dice_per_organ=[.5, .4, .3], step_seconds=2.0, shape=[8, 8, 8],
                             memory=dict(peak_allocated_bytes=2**30), augmentation=dict(flip=[True, False, False])))
        rows.append(dict(phase='train_epoch', epoch=epoch, epoch_metrics=dict(total_loss=1.0, hierarchy=0.0,
                         final=dict(loss=.6, soft_per_region=[.5, .4, .3]), coarse=dict(loss=.4))))
        dice = [round(.5 + offset + .01 * epoch + .001 * i, 6) for i in range(3)]
        organs = [dict(name=r, mean_dice=d) for r, d in zip(('WT', 'TC', 'ET'), dice)]
        hard = dict(mean_case_dice=sum(dice) / 3, organs=organs)
        rows.append(dict(phase='internal_dev', epoch=epoch, global_step=step, best_dev_improved=True,
                         metrics=dict(mean_case_dice=sum(dice) / 3, organs=organs),
                         epoch_metrics=dict(total_loss=1.0, hierarchy=0.0, final=dict(loss=.6, hard=hard, soft_per_region=dice),
                                            coarse=dict(loss=.4, hard=hard)),
                         early_stopping=dict(no_improvement_count=0, best_epoch=epoch, stopped=epoch == epochs)))
        folder = run / 'validation' / f'step_{step:08d}'
        folder.mkdir(parents=True)
        (folder / 'marker.json').write_text(json.dumps(dict(completed=True)))
        for i, case in enumerate(CASES):
            d = [x + .01 * i for x in dice]
            organs = [dict(name=r, dice=v, predicted_voxels=10, gt_voxels=10, true_positive=5) for r, v in zip(('WT', 'TC', 'ET'), d)]
            result = dict(organs=organs, hierarchy_violations=dict(ET_outside_TC=1, TC_outside_WT=0),
                          region_diagnostic=dict(coarse=dict(bce=.1, dice_loss=.3, hard=dict(organs=organs)),
                                                 final=dict(bce=.1, dice_loss=.3, soft_dice=d), hierarchy=0.0, total_loss=.9))
            (folder / f'{case}.json').write_text(json.dumps(dict(case_id=case, result=result)))
    (run / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return run


class ExportTests(unittest.TestCase):
    def test_tables_and_paired_comparison_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a, b = make_run(root, 'a', 0.0), make_run(root, 'b', 0.02)
            before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            out = root / 'out'
            with contextlib.redirect_stdout(io.StringIO()):
                status = export.main(['--run', f'E1:0={a}', '--run', f'A3:0={b}', '--pair', 'E1,A3', '--output', str(out)])
            self.assertEqual(status, 0)
            self.assertEqual(before, {str(p): sha256(p) for p in root.rglob('*') if p.is_file() and out not in p.parents})
            runs = list(csv.DictReader((out / 'runs.csv').open()))
            self.assertEqual([(r['label'], r['best_epoch'], r['status']) for r in runs],
                             [('E1', '6', 'early_stopped'), ('A3', '6', 'early_stopped')])
            self.assertEqual(len(list(csv.DictReader((out / 'epochs.csv').open()))), 12)
            self.assertEqual(len(list(csv.DictReader((out / 'cases_epochs.csv').open()))), 2 * 6 * 3)
            self.assertEqual(len(list(csv.DictReader((out / 'train_steps.csv').open()))), 24)
            comparison = json.loads((out / 'comparisons.json').read_text())['A3_vs_E1']
            self.assertEqual(comparison['folds'], [0])
            self.assertAlmostEqual(comparison['best']['pooled']['mean']['mean'], .02, places=9)
            self.assertEqual(comparison['best']['pooled']['WT']['n'], 3)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(export.main(['--run', f'E1:0={a}', '--output', str(out)]), 2)  # 拒绝覆盖


if __name__ == '__main__':
    unittest.main()

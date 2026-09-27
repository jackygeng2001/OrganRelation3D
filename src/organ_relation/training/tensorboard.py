"""Best-effort scalar observer. JSONL/checkpoints remain the source of truth."""
import json
from pathlib import Path
import sys
import time


# METHOD_SPEC labels 1..15; no dependency on exploratory imaging code.
ORGAN_NAMES = ('Spleen', 'RightKidney', 'LeftKidney', 'Gallbladder', 'Esophagus',
               'Liver', 'Stomach', 'Aorta', 'InferiorVenaCava', 'Pancreas',
               'RightAdrenal', 'LeftAdrenal', 'Duodenum', 'Bladder', 'ProstateOrUterus')
MODULE_NAMES = dict(encoder='Encoder', coarse_head='CoarseHead', relation='Relation',
                    node_to_space='NodeToSpace', fusion='Fusion', decoder='Decoder', network='MONAI_UNet')


def scalar_values(row):
    """Translate scalar log rows only; never touch live model tensors."""
    if row['phase'] == 'train':
        values = {'Train/Total_Loss': row['total_loss'],
                  'Train/Final_CE': row['final']['ce'],
                  'Train/Final_DiceLoss': row['final']['dice_loss'],
                  'Train/SoftDice_Mean': sum(row['soft_dice_per_organ']) / 15,
                  'Optimizer/LR': row['lr'], 'System/Step_Time': row['step_seconds']}
        if 'coarse' in row:
            values.update({'Train/Coarse_CE': row['coarse']['ce'],
                           'Train/Coarse_DiceLoss': row['coarse']['dice_loss']})
        if row.get('mode') in ('monai_reference_unet', 'monai_relation_unet', 'monai_coarse_aux'):
            for name, value in zip(ORGAN_NAMES, row['soft_dice_per_organ']):
                values['TrainSoftDice/' + name] = value
        for name in ('allocated', 'reserved'):
            value = row['memory'][f'peak_{name}_bytes']
            if value is not None:
                values[f'System/GPU_Peak_{name.title()}_GiB'] = value / 2**30
        for name, stats in (row['diagnostics'] or {}).items():
            if name in MODULE_NAMES:
                values['GradNorm/' + MODULE_NAMES[name]] = stats['gradient_norm']
        return values
    if row['phase'] in ('train_monitor', 'internal_dev'):
        prefix = 'Monitor' if row['phase'] == 'train_monitor' else 'Val'
        metrics = row['metrics']
        values = {f'{prefix}/HardDice_Mean': metrics['mean_case_dice']}
        if row.get('diagnostic_cases'):
            cases = row['diagnostic_cases'].values()
            values[f'{prefix}/FinalSoftDice_Mean'] = sum(c['final_soft_dice'] for c in cases) / len(cases)
        if row.get('mode') in ('monai_reference_unet', 'monai_relation_unet', 'monai_coarse_aux') and row.get('diagnostic_cases'):
            cases = list(row['diagnostic_cases'].values())
            for key in ('total_loss', 'ce_loss', 'dice_loss', 'predicted_foreground_voxels',
                        'foreground_true_positive_voxels', 'gt_foreground_voxels',
                        'gt_foreground_true_class_mean_probability', 'gt_foreground_background_mean_probability'):
                present = [c[key] for c in cases if c[key] is not None]
                if present:
                    values[f'{prefix}/' + key] = sum(present) / len(present)
            for i, name in enumerate(ORGAN_NAMES):
                values[f'{prefix}SoftDice/' + name] = sum(c['soft_dice_per_organ'][i] for c in cases) / len(cases)
        if row.get('mode') in ('monai_relation_unet', 'monai_coarse_aux') and row.get('diagnostic_cases'):
            coarse = [c['coarse'] for c in row['diagnostic_cases'].values()]
            for key in ('total_loss', 'ce_loss', 'dice_loss', 'final_soft_dice'):
                values[f'{prefix}Coarse/' + key] = sum(c[key] for c in coarse) / len(coarse)
            present = [c['hard_metrics']['mean_dice'] for c in coarse if c['hard_metrics']['mean_dice'] is not None]
            if present:
                values[f'{prefix}Coarse/HardDice_Mean'] = sum(present) / len(present)
            if row.get('mode') == 'monai_coarse_aux':
                for i, name in enumerate(ORGAN_NAMES):
                    values[f'{prefix}CoarseSoftDice/' + name] = sum(c['soft_dice_per_organ'][i] for c in coarse) / len(coarse)
                    present = [c['hard_metrics']['organs'][i]['dice'] for c in coarse
                               if c['hard_metrics']['organs'][i]['dice'] is not None]
                    if present:
                        values[f'{prefix}CoarseDice/' + name] = sum(present) / len(present)
        for organ in metrics['organs']:
            values[f'{prefix}Dice/{ORGAN_NAMES[organ["label"]-1]}'] = organ['mean_dice']
        return {k: v for k, v in values.items() if v is not None}
    return {}


class TensorBoardObserver:
    def __init__(self, run_dir, *, enabled=True):
        self.directory = Path(run_dir) / 'tensorboard'
        self.enabled = enabled
        self.writer = None
        self.failed = False

    def _failure(self, exc):
        self.failed = True
        self.enabled = False
        writer, self.writer = self.writer, None
        try:
            print(f'[TensorBoard disabled] {type(exc).__name__}: {exc}. '
                  'Training continues; metrics.jsonl/checkpoint remain authoritative. '
                  'TensorBoard curves may be incomplete.', file=sys.stderr, flush=True)
        except Exception:
            pass
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

    def start(self, committed_step, log_path):
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            # EventAccumulator reads filenames lexicographically. Default names
            # use SECOND-resolution time followed by host/PID/unpadded counter;
            # .10 can sort before .9 on a quick restart. Advance at most one
            # second before creating a new file, so purge is always read last.
            existing = list(self.directory.glob('events.out.tfevents.*'))
            if existing:
                latest = max(int(path.name.split('.')[3]) for path in existing)
                delay = latest + 1 - time.time()
                if delay > 1.1:
                    raise OSError('event timestamp is in the future; refusing out-of-order restart')
                if delay > 0:
                    time.sleep(delay + .01)
                if int(time.time()) <= latest:
                    raise OSError('clock moved backwards; refusing out-of-order restart')
            # Purge K itself and replay only committed rows at K, as validation
            # at that SAME step may have crashed before its checkpoint commit.
            self.writer = SummaryWriter(log_dir=str(self.directory),
                                        purge_step=committed_step if committed_step else None)
            if committed_step:
                with Path(log_path).open(encoding='utf-8') as stream:
                    for line in stream:
                        row = json.loads(line)
                        if row['global_step'] == committed_step:
                            self._record(row)
                self.writer.flush()
        except Exception as exc:
            self._failure(exc)

    def _record(self, row):
        for tag, value in scalar_values(row).items():
            self.writer.add_scalar(tag, value, global_step=row['global_step'])

    def record(self, row):
        if self.writer is not None:
            try:
                self._record(row)
            except Exception as exc:
                self._failure(exc)

    def flush(self):
        if self.writer is not None:
            try:
                self.writer.flush()
            except Exception as exc:
                self._failure(exc)

    def close(self):
        self.flush()
        if self.writer is not None:
            writer, self.writer = self.writer, None
            try:
                writer.close()
            except Exception as exc:
                self._failure(exc)

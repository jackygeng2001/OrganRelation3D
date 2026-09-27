"""Presentation of committed scalar rows; no model, RNG or ETA calculations."""
from functools import wraps
import shutil
import sys


def duration(seconds):
    if seconds is None:
        return 'warming up'
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (f'{days}d {hours:02d}h' if days else f'{hours}h {minutes:02d}m' if hours
            else f'{minutes}m {seconds:02d}s' if minutes else f'{seconds}s')


def columns(rows, title=None):
    """Fixed label/value widths in both columns; long values are abbreviated."""
    def cell(pair):
        if pair is None:
            return ' ' * 47
        key, value = pair
        value = str(value)
        if len(value) > 30:
            value = value[:27] + '...'
        return f' {key:<14}: {value:<30}'
    lines = ['=' * 97 if title else '-' * 97]
    if title:
        lines.append(title.center(97))
        lines.append('=' * 97)
    lines.extend(cell(left) + '   ' + cell(right) for left, right in rows)
    lines.append(lines[0])
    return '\n'.join(lines)


def _presentation(method):
    @wraps(method)
    def safe(self, *args, **kwargs):
        if not self.enabled:
            return
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            self.enabled = False
            for bar in (self.train_bar, self.val_bar):
                try:
                    if bar is not None:
                        bar.close()
                except Exception:
                    pass
            try:
                print(f'[Console disabled] {type(exc).__name__}: {exc}; training continues.', file=sys.stderr)
            except Exception:
                pass
    return safe


class TrainingConsole:
    def __init__(self, *, enabled=True, stream=None):
        self.enabled = enabled
        self.stream = sys.stdout if stream is None else stream
        self.train_bar = self.val_bar = None
        self.last = None
        self.last_summary_step = None

    def _bar(self, total, initial, desc):
        from tqdm import tqdm
        return tqdm(total=total, initial=initial, desc=desc, file=self.stream,
                    disable=not self.stream.isatty(), leave=False, dynamic_ncols=True,
                    bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}')

    def _write(self, message):
        from tqdm import tqdm
        tqdm.write(message, file=self.stream)

    @_presentation
    def start(self, identity, options, cases, total, epochs, state, *, resume=None, next_case=None, last=None):
        self.total, self.epochs, self.count = total, epochs, len(cases)
        self.global_bar = self.count <= 2
        self.last = last
        self.last_summary_step = None
        model, prep = identity.get('model', {}), identity.get('preprocessing', {})
        model = model if isinstance(model, dict) else {}
        prep = prep if isinstance(prep, dict) else {}
        channels = model.get('backbone', {}).get('channels', model.get('channels', []))
        candidate = prep.get('candidate', '?')
        spacing = prep.get('spacing_candidates', {}).get(candidate)
        spacing_text = ' x '.join(f'{v:g}' for v in spacing) + ' mm' if spacing else '?'
        optimizer = identity.get('optimizer', {})
        optimizer = optimizer if isinstance(optimizer, dict) else {}
        env = identity.get('environment', {})
        runtime = identity.get('runtime', {})
        self._write(columns([
            (('Model', f'{len(channels)}-stage {channels}' if channels else 'configured model'),
             ('Device', env.get('gpu_name') or runtime.get('device', 'CPU'))),
            (('Spacing', f'{candidate} ({spacing_text})'),
             ('Precision', {'float32': 'FP32'}.get(runtime.get('dtype', 'float32'), runtime.get('dtype')))),
            (('Train Cases', len(cases)), ('Optimizer', optimizer.get('name', 'configured'))),
            (('LR', f"{optimizer['lr']:.2e}" if 'lr' in optimizer else '?'), ('Batch Size', options['batch_size'])),
            (('Max Steps', total), ('Resume', 'Checkpoint' if resume else 'Fresh run')),
            (('Checkpoint', f"every {options['checkpoint_every']} steps"),
             ('Monitor' if options['validation_role'] == 'train_monitor' else 'Validation',
              f"every {options['validation_every']} steps" if options['validation_every'] else 'off')),
        ], 'External MONAI reference' if identity.get('mode') == 'monai_reference_unet' else 'OrganRelation3D'))
        if resume:
            self._write('[RESUME]\n' + f' Checkpoint : {resume}\n'
                        f' Epoch      : {min(state["epoch"]+1, epochs)} / {epochs}\n'
                        f' Step       : {state["global_step"]} / {total}\n'
                        f' Next Case  : {next_case}')
        if self.global_bar:
            self.train_bar = self._bar(total, state['global_step'], '[Train]')

    @_presentation
    def begin_step(self, epoch, cursor):
        if not self.global_bar and self.train_bar is None:
            total = min(self.count, self.total - (epoch - 1) * self.count)
            self.train_bar = self._bar(total, cursor, f'[Train {epoch}/{self.epochs}]')

    @_presentation
    def train_step(self, row):
        self.last = row  # Only Python scalars/lists from the machine log.
        peak = row['memory']['peak_allocated_bytes']
        memory = f'{peak / 2**30:.1f}G' if peak is not None else 'CPU'
        postfix = (f"loss={row['total_loss']:.3f} case={row['case_id']} mem={memory} "
                   f"ETA={duration(row['progress']['eta_seconds'])}")
        if shutil.get_terminal_size().columns >= 110:
            postfix += f" lr={row['lr']:.1e}"
        if self.train_bar is not None:
            self.train_bar.set_postfix_str(postfix, refresh=False)
            self.train_bar.update(1)

    @_presentation
    def epoch_end(self, *, monitored=False):
        if not self.global_bar:
            self.train_bar.close()
            self.train_bar = None
            if not monitored:
                self._summary()

    def _summary(self, metrics=None, phase=None):
        row = self.last
        if row is None:
            return
        peak = row['memory']['peak_allocated_bytes']
        dice = None
        if metrics is not None:
            value = metrics['mean_case_dice']
            dice = ('Monitor Dice' if phase == 'train_monitor' else 'Val Dice',
                    f'{value:.4f}' if value is not None else 'n/a (all empty)')
        self._write(columns([
            (('Epoch', f"{row['epoch']} / {self.epochs}"), ('Step', f"{row['global_step']} / {self.total}")),
            (('Last Loss', f"{row['total_loss']:.4f}"), dice),
            (('LR', f"{row['lr']:.2e}"), ('GPU Peak', f'{peak/2**30:.2f} GiB' if peak is not None else 'n/a (CPU)')),
            (('Step Mean', duration(row['progress']['rolling_seconds'])), ('Train ETA', duration(row['progress']['eta_seconds']))),
        ]))
        self.last_summary_step = row['global_step']

    @_presentation
    def validation_start(self, total, phase):
        self.val_bar = self._bar(total, 0, '[Monitor]' if phase == 'train_monitor' else '[Val]')

    @_presentation
    def validation_case(self, case, eta):
        self.val_bar.set_postfix(dict(case=case, ETA=duration(eta['eta_seconds'])), refresh=False)
        self.val_bar.update(1)

    @_presentation
    def validation_end(self):
        if self.val_bar is not None:
            self.val_bar.close()
            self.val_bar = None

    @_presentation
    def validation_summary(self, row):
        self._summary(row['metrics'], row['phase'])

    @_presentation
    def finish(self):
        if self.last and self.last_summary_step != self.last['global_step']:
            self._summary()

    @_presentation
    def close(self):
        for bar in (self.train_bar, self.val_bar):
            if bar is not None:
                bar.close()
        self.train_bar = self.val_bar = None

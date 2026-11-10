"""控制台输出：只呈现已写入日志的标量，不碰模型、随机数或训练状态。

格式（2026-11-10 用户确定）：
- 启动时一份 Pre-flight Checklist（逐项 [INFO]）；
- 按轮训练时，每次验证后输出一块固定 64 列宽的报告：上下 `=` 包裹，三行数据（[Epoch] 悬挂缩进：
  训练损失/验证 Dice、学习率/显存峰值、折号/本轮用时），`-` 分隔后一行 SAVE/WARN 状态与 ETA，
  每块之间空一行；所有行在同一列结束；
- 早停时一块 STOP 报告；进度条只在终端显示、不留在输出里。
按步运行的开发配置（无每轮验证）每 console_every_steps 步输出一行。
任何显示错误只会关闭控制台，训练照常进行。
"""
from functools import wraps
import math
import sys
import time

WIDTH = 64


def duration(seconds):
    if seconds is None:
        return 'warming up'
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (f'{days}d {hours:02d}h' if days else f'{hours}h {minutes:02d}m' if hours
            else f'{minutes}m {seconds:02d}s' if minutes else f'{seconds}s')


def rule(char):
    return char * WIDTH


def fit(text, width=WIDTH):
    """截断到宽度；保证任何一行都不超过报告宽度。"""
    text = str(text)
    return text if len(text) <= width else text[:width - 3] + '...'


def spread(left, right):
    """左边说明、右边数值，右端落在第 WIDTH 列；说明过长时截断，两段之间至少两个空格。"""
    right = fit(right, WIDTH - 4)
    return fit(left, WIDTH - len(right) - 2).ljust(WIDTH - len(right)) + right


def report_row(head, left_key, left_value, right_key, right_value):
    """报告的一行：[Epoch] 悬挂缩进 | 左栏（键 10 宽，值 10 宽）| 右栏（键 8 宽，值靠右对齐到第 WIDTH 列）。"""
    prefix = f'{head:<16}| {left_key:<10} : {fit(left_value, 10):<10} | {right_key:<8} : '
    return prefix + fit(right_value, WIDTH - len(prefix)).rjust(WIDTH - len(prefix))


def _gib(value):
    return f'{value / 2**30:.2f} GiB' if value is not None else 'CPU'


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


def _mapping(value):
    """身份中的某一节不是字典（例如测试夹具里的字符串）时当作空；控制台只做展示，不做校验。"""
    return value if isinstance(value, dict) else {}


def _fold(identity):
    split = _mapping(_mapping(identity.get('data')).get('split'))
    fold, folds = split.get('fold'), split.get('folds')
    return f'{fold}/{folds}' if isinstance(fold, int) and isinstance(folds, int) else None


def _model_text(model):
    variant = model.get('variant')
    if variant is None:
        return f"full method, {model.get('graph') or 'complete'} graph, K={model.get('rounds', '?')}"
    return {'no_relation': 'nodes without relation reasoning',
            'drf_backbone': 'DRF encoder + plain decoder',
            'unet': '3D U-Net (plain encoder and decoder)'}.get(variant, variant)


class TrainingConsole:
    def __init__(self, *, enabled=True, stream=None):
        self.enabled = enabled
        self.stream = sys.stdout if stream is None else stream
        self.train_bar = self.val_bar = None
        self.fold = None
        self.last = None
        self.best = None
        self.epoch_loss = None
        self.epoch_peak = None
        self.trained_seconds = 0.0
        self.console_every_steps = None
        self.epoch_clock = None
        self.patience, self.min_epochs = None, 0

    def _bar(self, total, initial, desc):
        from tqdm import tqdm
        return tqdm(total=total, initial=initial, desc=desc, file=self.stream,
                    disable=not self.stream.isatty(), leave=False, dynamic_ncols=True,
                    bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}')

    def _write(self, message):
        from tqdm import tqdm
        tqdm.write(message, file=self.stream)

    def _label(self, text):
        return f'Fold {self.fold} {text}' if self.fold else text

    @_presentation
    def start(self, identity, options, cases, total, epochs, state, *, resume=None, next_case=None, last=None,
              run_name=None, parameters=None, best=None, trained_seconds=0.0):
        self.total, self.epochs, self.count = total, epochs, len(cases)
        self.epoch_mode = options.get('cadence_unit') == 'epoch'
        self.global_bar = self.count <= 2
        self.last, self.best, self.trained_seconds = last, best, trained_seconds or 0.0
        self.console_every_steps = None if self.epoch_mode else options.get('console_every_steps')
        self.fold = _fold(identity)
        self.epoch_clock = time.monotonic()
        model, prep, loss = (_mapping(identity.get(k)) for k in ('model', 'preprocessing', 'loss'))
        optimizer, env, runtime = (_mapping(identity.get(k)) for k in ('optimizer', 'environment', 'runtime'))
        data = _mapping(identity.get('data'))
        manifest, split = _mapping(data.get('manifest')), _mapping(data.get('split'))
        git = _mapping(_mapping(identity.get('provenance')).get('git'))
        crop = _mapping(prep.get('spatial_crop'))
        early = _mapping(options.get('early_stopping'))
        # 早停计数以验证次数计；显示为 计数/需要的次数，与 patience_epochs、验证间隔一致
        self.patience = math.ceil(early['patience_epochs'] / (options.get('validation_every') or 1)) if early else None
        self.min_epochs = early['min_epochs'] if early else 0
        augmentation = _mapping(options.get('augmentation'))
        rocm, torch_version = env.get('rocm'), env.get('torch')
        device = env.get('gpu_name') or runtime.get('device', 'CPU')
        device += f", ROCm {'.'.join(str(rocm).split('.')[:2])}" if rocm else (f", CUDA {env['cuda']}" if env.get('cuda') else '')
        precision = ('bf16 whole network' if options.get('mixed_precision') == 'bfloat16' else
                     'bf16 convolutions' if options.get('convolution_compute_dtype') == 'bfloat16' else 'FP32')
        cadence = 'epoch' if self.epoch_mode else 'step'
        validation = (f"every {options['validation_every']} {cadence}" if options.get('validation_every') else 'off')
        if resume:
            resume_text = f"epoch {min(state['epoch'] + 1, epochs)}/{epochs}, step {state['global_step']}/{total}"
        items = [
            ('Run', run_name or '-'),
            ('Fold', f"{self.fold} (seed {options.get('seed', '?')})" if self.fold else '- (not a fold run)'),
            ('Device', device),
            ('Code', (f"{git['commit'][:7]} ({'dirty' if git.get('dirty') else 'clean'})" if git and git.get('commit') else '-')
                     + (f", PyTorch {str(torch_version).split('+')[0]}" if torch_version else '')),
            ('Dataset', f"{manifest.get('cohort', '-')} {manifest.get('source_partition', '')}".rstrip()
                        + f", {len(cases)} train / {len(split.get('dev') or []) or '-'} val"),
            ('Data', (f"crop {crop['kind']} x{crop.get('multiple_of', 1)}" if crop.get('kind') else 'crop none')
                     + (', cache' if runtime.get('data_cache') else '') + (' + prefetch' if runtime.get('data_prefetch') else '')),
            ('Model', _model_text(model)),
            ('Params', f'{parameters:,}' if isinstance(parameters, int) else '-'),
            ('Loss', f"final + {loss.get('lambda_c', '?')} coarse"
                     + (f" + {loss['lambda_hier']} hierarchy" if loss.get('lambda_hier') else '')),
            ('Precision', precision + (f", {runtime['convolution_backend']}" if runtime.get('convolution_backend') else '')),
            ('Optimizer', f"{optimizer.get('name', 'configured')}"
                          + (f", lr {optimizer['lr']:.2e}" if 'lr' in optimizer else '')
                          + (f", wd {optimizer['weight_decay']:g}" if 'weight_decay' in optimizer else '')
                          + f", batch {options['batch_size']}"),
            ('Epochs', f"max {epochs}, validation {validation}" if self.epoch_mode
                       else f"max {total} steps, validation {validation}"),
            ('Early Stop', f"min {early['min_epochs']}, patience {early['patience_epochs']}, delta {early['min_delta']:g}"
                           if early else 'off'),
            ('Augment', f"flip p={augmentation['flip_probability']:g}, intensity +-{augmentation['intensity_scale']:g}"
                        if augmentation and augmentation.get('kind') == 'flip_intensity_v1' else 'off'),
            ('Checkpoint', f"every {options['checkpoint_every_steps']} steps + epoch end"
                           if options.get('checkpoint_every_steps') else f"every {options['checkpoint_every']} {cadence}"),
            ('Resume', resume_text if resume else 'fresh run'),
        ]
        if resume:
            items.append(('Next Case', next_case))
        lines = [rule('='), 'Pre-flight Checklist'.center(WIDTH).rstrip(), rule('-')]
        lines += [fit(f'[INFO] {key:<10} : {value}') for key, value in items]
        lines += [rule('='), '', 'Training Started'.center(WIDTH).rstrip(), '']
        self._write('\n'.join(lines))
        if self.global_bar:
            self.train_bar = self._bar(total, state['global_step'], self._label('Train'))

    @_presentation
    def begin_step(self, epoch, cursor):
        if not self.global_bar and self.train_bar is None:
            total = min(self.count, self.total - (epoch - 1) * self.count)
            self.train_bar = self._bar(total, cursor, self._label(f'Epoch {epoch}/{self.epochs}'))

    @_presentation
    def train_step(self, row):
        self.last = row  # 只用机器日志里的 Python 标量。
        self.trained_seconds += row.get('step_seconds') or 0.0
        peak = row['memory']['peak_allocated_bytes']
        if peak is not None:
            self.epoch_peak = peak if self.epoch_peak is None else max(self.epoch_peak, peak)
        if self.train_bar is not None:
            self.train_bar.set_postfix_str(f"loss={row['total_loss']:.3f} ETA={duration(row['progress']['eta_seconds'])}",
                                           refresh=False)
            self.train_bar.update(1)
        if self.console_every_steps and row['global_step'] % self.console_every_steps == 0:
            self._write(fit(f"[Step {row['global_step']}/{self.total}] loss {row['total_loss']:.4f} | "
                            f"lr {row['lr']:.2e} | gpu {_gib(peak)} | ETA {duration(row['progress']['eta_seconds'])}"))

    @_presentation
    def train_epoch(self, row):
        """本轮训练结束时引擎写入日志的 train_epoch 行：取全轮平均总损失。"""
        self.epoch_loss = row['epoch_metrics']['total_loss']

    @_presentation
    def epoch_end(self, *, monitored=False):
        if not self.global_bar and self.train_bar is not None:
            self.train_bar.close()
            self.train_bar = None
        if self.epoch_mode and not monitored and self.last is not None:
            self._report(None, None, '[INFO] No validation this epoch')

    def _report(self, dice, phase, status):
        row = self.last
        epoch = row['epoch'] if self.epoch_mode else None
        head = f'[Epoch {epoch:03d}/{self.epochs:03d}]' if epoch is not None else f"[Step {row['global_step']}]"
        dice_key = 'Mon Dice' if phase == 'train_monitor' else 'Val Dice'
        loss = f'{self.epoch_loss:.4f}' if self.epoch_loss is not None else '-'
        seconds = time.monotonic() - self.epoch_clock
        lines = [rule('='),
                 report_row(head, 'Train Loss', loss, dice_key, '-' if dice is None else f'{dice:.4f}'),
                 report_row('', 'Lr', f"{row['lr']:.2e}", 'GPU Peak', _gib(self.epoch_peak)),
                 report_row('', 'Fold', self.fold or '-', 'Time', duration(seconds)),
                 rule('-'),
                 spread(status, f"ETA {duration(row['progress']['eta_seconds'])}"),
                 rule('='), '']
        self._write('\n'.join(lines))
        self.epoch_loss = self.epoch_peak = None
        self.epoch_clock = time.monotonic()

    @_presentation
    def validation_start(self, total, phase):
        self.val_bar = self._bar(total, 0, self._label('Monitor' if phase == 'train_monitor' else 'Val'))

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
        if self.last is None:
            return
        dice = row['metrics']['mean_case_dice']
        early = row.get('early_stopping')
        improved = row.get('best_dev_improved')
        previous = self.best['score'] if isinstance(self.best, dict) else None
        if improved:
            gain = f' (+{dice - previous:.4f})' if previous is not None else ''
            # 新的原始最佳但累计提升未达 min_delta 时，早停计数不清零，一并显示
            status = (f"[SAVE] Best {dice:.4f}{gain}, saved; no gain {early['no_improvement_count']}/{self.patience}"
                      if early and early['no_improvement_count'] else f'[SAVE] Best {dice:.4f}{gain}, checkpoint saved')
            self.best = dict(score=dice, epoch=row['epoch'])
        elif dice is None:
            status = '[WARN] Dice undefined (all regions empty)'
        elif early:
            count, patience, minimum = early['no_improvement_count'], self.patience, self.min_epochs
            if count >= patience and row['epoch'] < minimum:
                status = f'[WARN] No gain {count}/{patience}, min epochs {minimum} not reached'
            else:
                status = f"[WARN] No gain {count}/{patience}, best {early['best_metric']:.4f} @ epoch {early['best_epoch']}"
        else:
            status = (f"[WARN] No gain, best {self.best['score']:.4f} @ epoch {self.best['epoch']}"
                      if isinstance(self.best, dict) else '[WARN] No gain')
        self._report(dice, row['phase'], status)

    @_presentation
    def early_stop(self, state):
        self._write('\n'.join([
            rule('='),
            spread(f"[STOP] Early stopping at epoch {state['last_validation_epoch']}", f'Fold {self.fold or "-"}'),
            spread(f"       Best Val Dice {state['best_metric']:.4f} @ epoch {state['best_epoch']}",
                   f'Train {duration(self.trained_seconds)}'),
            rule('='), '']))

    @_presentation
    def finish(self):
        if self.last is not None and self.last['global_step'] >= self.total:
            self._write('\n'.join([rule('='), spread(f'[DONE] Reached the maximum of {self.epochs} epochs'
                                                     if self.epoch_mode else f'[DONE] Reached {self.total} steps',
                                                     f'Fold {self.fold or "-"}'),
                                   spread('', f'Train {duration(self.trained_seconds)}'), rule('='), '']))

    @_presentation
    def close(self):
        for bar in (self.train_bar, self.val_bar):
            if bar is not None:
                bar.close()
        self.train_bar = self.val_bar = None

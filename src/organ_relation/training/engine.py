"""One serial full-scan loop for smoke, overfit, pilot and final training."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import torch
from torch.utils.data import DataLoader

from ..evaluation.progress import CaseLedger
from ..metrics import METRIC_PROTOCOL, hard_dice, summarize_dice
from ..losses import (JointLoss, SegmentationLoss, foreground_background_ce_means,
                      foreground_class_ce_means, resolve_ce_weights)
from .console import TrainingConsole
from .progress import ProgressClock
from .extension import resolve_horizon
from .state import (ScalarLog, atomic_json, capture_rng, digest, load_checkpoint,
                    restore_rng, save_checkpoint)
from .tensorboard import TensorBoardObserver
from .development import (cadence_due, summarize_branches, validation_branches,
                          validate_early_stopping, new_early_state, advance_early_stopping)
from .monai_relation import observed_forward


def validate_options(options):
    if options.get('loss_observation', 'legacy') not in ('legacy', 'monitor_only'):
        raise ValueError('invalid loss_observation')
    if options.get('cadence_unit', 'step') not in ('step', 'epoch'):
        raise ValueError('invalid cadence_unit')
    for key in ('checkpoint_every', 'diagnostics_every', 'validation_every'):
        if type(options[key]) is not int or options[key] < (1 if key == 'checkpoint_every' else 0):
            raise ValueError(f'invalid {key}')
    for key in ('checkpoint_every_steps', 'console_every_steps'):
        if key in options and (type(options[key]) is not int or options[key] < 1):
            raise ValueError(f'{key} must be a positive integer when specified')
    for key in ('max_steps', 'max_epochs'):
        if options[key] is not None and (type(options[key]) is not int or options[key] < 1):
            raise ValueError(f'{key} must be positive or null')
    if options['max_steps'] is None and options['max_epochs'] is None:
        raise ValueError('max_steps or max_epochs required')
    if options['batch_size'] != 1 or options['num_workers'] != 0:
        raise ValueError('training v1 requires batch_size=1 and num_workers=0')
    if options['scheduler'] is not None:
        raise ValueError('scheduler not implemented')
    if options['memory_format'] not in ('contiguous', 'channels_last_3d'):
        raise ValueError('invalid memory format')
    if type(options['seed']) is not int or type(options['shuffle']) is not bool:
        raise ValueError('explicit seed and shuffle required')
    ProgressClock(options['eta_window'], options['eta_warmup'])
    validate_early_stopping(options)


def weights_hash(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def diagnostic_parameter_groups(model):
    from ..models.monai_relation import MonaiRelationUNet
    from ..models.monai_coarse_aux import MonaiCoarseAuxUNet
    if isinstance(model, (MonaiRelationUNet, MonaiCoarseAuxUNet)):
        return model.diagnostic_parameter_groups()
    return {name: [(f'{name}.{n}', p) for n, p in module.named_parameters()]
            for name, module in model.named_children()}


def gradient_diagnostics(model):
    result = {}
    for name, named in diagnostic_parameter_groups(model).items():
        parameters = [p for _, p in named if p.requires_grad]
        if not parameters:  # e.g. SpaceToNode has no learnable parameters.
            continue
        if any(p.grad is None for p in parameters):
            raise ValueError(f'missing gradients in {name}')
        if any(not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError(f'nonfinite gradients in {name}')
        norm = torch.stack([p.grad.detach().double().square().sum() for p in parameters]).sum().sqrt()
        result[name] = dict(gradient_norm=norm.item(), finite=True)
    return result


def foreground_ce_diagnostics(voxel_loss, indices, criterion):
    """Single-case monitor statistics; never retain tensors in logs."""
    means, macro, count = foreground_class_ce_means(voxel_loss, indices)
    values = means[0].detach().cpu().tolist()
    return dict(foreground_ce_reduction=criterion.foreground_ce_reduction,
                present_foreground_class_count=count[0].item(),
                per_class_ce_mean=[v if math.isfinite(v) else None for v in values],
                CE_fg_macro=macro[0].item() if torch.isfinite(macro[0]) else None), macro


def final_diagnostic_metrics(logits, label, criterion, hard_metrics):
    """Single-case monitor only. TP requires the correct foreground CLASS.

    Probability means are restricted to GT foreground for observation only;
    they never enter model.forward or alter the supervised loss.
    """
    soft_dice = criterion(logits, label).final.dice_per_class[0].mean().item()
    probabilities = logits.softmax(dim=1)
    foreground = label > 0
    count = foreground.sum().item()
    true_probability = probabilities.gather(1, label.unsqueeze(1)).squeeze(1)
    voxel_loss = -torch.log(true_probability.flatten(1) + criterion.epsilon)
    bg, fg = foreground_background_ce_means(
        voxel_loss, foreground.flatten(1))
    fg_stats, macro = foreground_ce_diagnostics(voxel_loss, label.flatten(1), criterion)
    selected_fg = macro if criterion.foreground_ce_reduction == 'class_macro_mean' else fg
    bg_mean = bg[0].item() if torch.isfinite(bg[0]) else None
    fg_mean = fg[0].item() if torch.isfinite(fg[0]) else None
    return dict(**fg_stats, final_soft_dice=soft_dice,
                ce_reduction_mode=criterion.ce_reduction_mode,
                ce_background_weight=criterion.ce_background_weight,
                ce_foreground_weight=criterion.ce_foreground_weight,
                CE_bg_mean=bg_mean, CE_fg_mean=fg_mean,
                balanced_ce=(0.5 * bg[0] + 0.5 * fg[0]).item() if bg_mean is not None and fg_mean is not None else None,
                weighted_ce=(criterion.ce_background_weight * bg[0] + criterion.ce_foreground_weight * selected_fg[0]).item()
                if bg_mean is not None and fg_mean is not None else None,
                predicted_foreground_voxels=sum(o['predicted_voxels'] for o in hard_metrics['organs']),
                foreground_true_positive_voxels=sum(o['true_positive'] for o in hard_metrics['organs']),
                gt_foreground_voxels=count,
                gt_foreground_true_class_mean_probability=true_probability[foreground].mean().item() if count else None,
                gt_foreground_background_mean_probability=probabilities[:, 0][foreground].mean().item() if count else None)


def joint_ce_diagnostic_metrics(output, label, criterion):
    """Single-case scalar monitor; process branches sequentially to bound memory."""
    def branch(logits):
        matched = logits.softmax(1).gather(1, label.unsqueeze(1)).squeeze(1)
        voxel_loss = -torch.log(matched.flatten(1) + criterion.epsilon)
        bg, fg = foreground_background_ce_means(
            voxel_loss, label.flatten(1) > 0)
        fg_stats, macro = foreground_ce_diagnostics(voxel_loss, label.flatten(1), criterion)
        selected_fg = macro if criterion.foreground_ce_reduction == 'class_macro_mean' else fg
        return dict(**fg_stats, CE_bg_mean=bg[0].item(), CE_fg_mean=fg[0].item(),
                    ce_background_weight=criterion.ce_background_weight,
                    ce_foreground_weight=criterion.ce_foreground_weight,
                    weighted_ce=(criterion.ce_background_weight * bg[0] + criterion.ce_foreground_weight * selected_fg[0]).item(),
                    balanced_ce=(0.5 * bg[0] + 0.5 * fg[0]).item())

    coarse = branch(torch.nn.functional.interpolate(
        output.coarse_logits, size=tuple(label.shape[1:]), mode='trilinear',
        align_corners=criterion.align_corners))
    final = branch(output.final_logits)
    return dict(coarse=coarse, final=final)


class Trainer:
    def __init__(self, model, criterion, optimizer, dataset, case_ids, *, device,
                 options, identity, run_dir, validation_dataset=None, validation_case_ids=(), resume=None,
                 console=None, tensorboard=False, extend_to=None):
        validate_options(options)
        if identity.get('training') != options:
            raise ValueError('training options must match the strict run identity')
        self.mode = identity.get('mode', 'organ_relation_joint')
        if self.mode in ('monai_reference_unet', 'monai_relation_unet', 'monai_coarse_aux'):
            from .monai_reference import MonaiReferenceLoss
            from .monai_relation import MonaiRelationLoss
            expected = MonaiReferenceLoss if self.mode == 'monai_reference_unet' else MonaiRelationLoss
            if not isinstance(criterion, expected) or criterion.constructor != identity.get('loss'):
                raise ValueError('MONAI criterion must match reference identity')
            if any(k in identity for k in ('ce_reduction_mode', 'foreground_ce_reduction',
                                           'ce_background_weight', 'ce_foreground_weight')):
                raise ValueError('MONAI reference cannot use custom CE options')
            if self.mode == 'monai_relation_unet':
                from ..models.monai_relation import MonaiRelationUNet
                if (not isinstance(model, MonaiRelationUNet) or not model.relation_enabled
                        or model.relation_config != identity.get('relation')
                        or identity.get('relation_enabled') is not True
                        or identity.get('coarse_supervision') != dict(lambda_c=criterion.lambda_c, align_corners=criterion.align_corners)):
                    raise ValueError('MONAI relation model/loss must match identity')
            if self.mode == 'monai_coarse_aux':
                from ..models.monai_coarse_aux import MonaiCoarseAuxUNet
                if (not isinstance(model, MonaiCoarseAuxUNet)
                        or model.coarse_config != identity.get('coarse_head')
                        or any(key in identity for key in ('relation', 'relation_enabled'))
                        or identity.get('coarse_supervision') != dict(lambda_c=criterion.lambda_c, align_corners=criterion.align_corners)):
                    raise ValueError('MONAI coarse-only model/loss must match identity')
            self.ce_reduction_mode = 'monai_cross_entropy'
            self.foreground_ce_reduction = 'voxel_mean'
            self.ce_weights = {}
        else:
            expected_loss = {'organ_relation_joint': JointLoss, 'backbone_only_final': SegmentationLoss}.get(self.mode)
            if expected_loss is None or not isinstance(criterion, expected_loss):
                raise ValueError('unsupported or mismatched model/loss mode')
            self.ce_reduction_mode = getattr(criterion, 'ce_reduction_mode', 'voxel_mean')
            if identity.get('ce_reduction_mode', 'voxel_mean') != self.ce_reduction_mode:
                raise ValueError('criterion ce_reduction_mode must match checkpoint/run identity')
            self.foreground_ce_reduction = criterion.foreground_ce_reduction
            if identity.get('foreground_ce_reduction', 'voxel_mean') != self.foreground_ce_reduction:
                raise ValueError('criterion foreground_ce_reduction must match checkpoint/run identity')
            bg, fg = resolve_ce_weights(identity.get('ce_background_weight', 0.5),
                                        identity.get('ce_foreground_weight', 0.5))
            self.ce_weights = dict(ce_background_weight=bg, ce_foreground_weight=fg)
            if (criterion.ce_background_weight, criterion.ce_foreground_weight) != (bg, fg):
                raise ValueError('criterion CE weights must match checkpoint/run identity')
        if extend_to is not None and (resume is None or type(extend_to) is not int or extend_to < 1):
            raise ValueError('--extend-to requires --resume and a positive integer total')
        if not case_ids or len(dataset) != len(case_ids) or len(set(case_ids)) != len(case_ids):
            raise ValueError('invalid training case list')
        if options['validation_every'] and (validation_dataset is None or not validation_case_ids):
            raise ValueError('validation requested without validation cases')
        if validation_dataset is not None and len(validation_dataset) != len(validation_case_ids):
            raise ValueError('validation case count mismatch')
        self.model, self.criterion, self.optimizer = model, criterion, optimizer
        self.dataset, self.case_ids = dataset, list(case_ids)
        self.validation_dataset, self.validation_case_ids = validation_dataset, list(validation_case_ids)
        self.device, self.options, self.identity = torch.device(device), options, identity
        self.epoch_mode = options.get('cadence_unit') == 'epoch'
        self.scalar_only = options.get('loss_observation') == 'monitor_only'
        self.observation_profile = 'development_v2' if self.scalar_only else 'development_v1'
        if self.scalar_only and (not self.epoch_mode or self.mode not in ('monai_reference_unet', 'monai_relation_unet')):
            raise ValueError('monitor-only optimization requires epoch MONAI A/C')
        if self.epoch_mode and self.mode not in ('monai_reference_unet', 'monai_relation_unet'):
            raise ValueError('epoch observation currently requires MONAI A/C')
        self.run_dir = Path(run_dir)
        self.console = console if console is not None else TrainingConsole(enabled=False)
        self.board = TensorBoardObserver(self.run_dir, enabled=tensorboard)
        self.resume_path = resume
        self.clock = ProgressClock(options['eta_window'], options['eta_warmup'])
        self.sampler_generator = torch.Generator().manual_seed(options['seed'])
        self.loader_generator = torch.Generator().manual_seed(options['seed'] + 1)
        self.state = dict(global_step=0, epoch=0, order=[], cursor=0, pending_validation=False)
        if not resume and self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise ValueError('new run directory must be empty; use --resume')
        checkpoint = load_checkpoint(resume, identity, extension=extend_to is not None) if resume else None
        self.horizon = resolve_horizon(options, len(dataset), checkpoint, extend_to, identity, resume)
        self.total = self.horizon['total_steps']
        self.epochs = math.ceil(self.total / len(dataset))
        self.extension_pending = extend_to is not None
        self.origin_identity = checkpoint.get('origin_identity', checkpoint['identity']) if checkpoint else identity
        self.development_state = dict(epoch_cases=[], validation_history=[], best_dev=None)
        self.early_state = new_early_state() if options.get('early_stopping') else None
        if checkpoint and self.early_state is not None:
            if 'early_stopping' not in checkpoint:
                raise ValueError('missing early stopping state')
            self.early_state = checkpoint['early_stopping']
        if checkpoint and self.epoch_mode:
            if 'development_state' not in checkpoint:
                raise ValueError('missing development aggregate/checkpoint history')
            self.development_state = checkpoint['development_state']
            if len(self.development_state['epoch_cases']) != checkpoint['progress']['cursor']:
                raise ValueError('epoch aggregate does not match committed case position')
            if self.early_state is not None:
                expected = new_early_state()
                for event in self.development_state['validation_history']:
                    expected = advance_early_stopping(expected, event['monitor_score'], event['epoch'], options)
                if expected != self.early_state or expected['last_validation_epoch'] > checkpoint['progress']['epoch']:
                    raise ValueError('early stopping state does not match validation history')
        if checkpoint:
            self._validate_progress(checkpoint)
            run = json.loads((self.run_dir / 'run.json').read_text(encoding='utf-8'))
            if run != dict(run_id=checkpoint['run_id'], identity=self.origin_identity):
                raise ValueError('resume requires the original run identity')
            if not (self.run_dir / 'metrics.jsonl').exists():
                raise ValueError('resume requires the original run log')
        self.log = ScalarLog(self.run_dir / 'metrics.jsonl')
        if checkpoint:
            self.model.load_state_dict(checkpoint['model'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.state = checkpoint['progress']
            self.run_id = checkpoint['run_id']
            self.log.recover(checkpoint['log'])
            self.sampler_generator.set_state(checkpoint['sampler_generator'])
            self.loader_generator.set_state(checkpoint['loader_generator'])
            # Last initialization step, before any data iteration/random sampling.
            restore_rng(checkpoint['rng'], self.device)
        else:
            self.run_id = str(uuid.uuid4())
            atomic_json(self.run_dir / 'run.json', dict(run_id=self.run_id, identity=identity))
        self.model.train()

    def _validate_progress(self, checkpoint):
        s = checkpoint['progress']
        if set(s) != {'global_step', 'epoch', 'order', 'cursor', 'pending_validation'}:
            raise ValueError('incomplete training progress')
        if any(type(s[k]) is not int or s[k] < 0 for k in ('global_step', 'epoch', 'cursor')):
            raise ValueError('invalid progress counters')
        n = len(self.case_ids)
        if (s['global_step'] > self.total or s['cursor'] >= n
                or s['global_step'] != s['epoch'] * n + s['cursor']
                or (s['order'] and sorted(s['order']) != list(range(n)))
                or (s['cursor'] and not s['order']) or type(s['pending_validation']) is not bool
                or checkpoint['log']['global_step'] != s['global_step']):
            raise ValueError('inconsistent checkpoint progress')

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _memory(self):
        if self.device.type != 'cuda':
            return dict(peak_allocated_bytes=None, peak_reserved_bytes=None)
        return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))

    def _batch(self, dataset, index, *, validation=False):
        # One index per iterator makes loader RNG consumption exactly one per
        # committed sample, including mid-epoch resume. No worker/prefetch state.
        generator = torch.Generator().manual_seed(0) if validation else self.loader_generator
        batch = next(iter(DataLoader(dataset, batch_size=1, sampler=[index], num_workers=0,
                                     generator=generator)))
        image, label = batch.image.to(self.device), batch.label.to(self.device)
        if self.options['memory_format'] == 'channels_last_3d':
            image = image.contiguous(memory_format=torch.channels_last_3d)
        return image, label

    def checkpoint(self):
        if not self.state['global_step']:
            return  # No completed optimizer step yet.
        payload = dict(
            schema_version=1, run_id=self.run_id, identity=self.identity,
            origin_identity=self.origin_identity, horizon=self.horizon,
            model=self.model.state_dict(), optimizer=self.optimizer.state_dict(), scheduler=None,
            progress=self.state, rng=capture_rng(self.device),
            sampler_generator=self.sampler_generator.get_state(),
            loader_generator=self.loader_generator.get_state(),
            log=self.log.position(self.state['global_step']))
        if self.epoch_mode:
            payload['development_state'] = self.development_state
            if self.early_state is not None:
                payload['early_stopping'] = self.early_state
            best = self.development_state['best_dev']
            # Best first, then last: a crash before last commits replays pending
            # validation from its ledger and idempotently writes this best again.
            if best and best['global_step'] == self.state['global_step'] and not self.state['pending_validation']:
                save_checkpoint(self.run_dir / 'best-dev.ckpt', payload)
        save_checkpoint(self.run_dir / 'last.ckpt', payload)
        self.board.flush()  # Observer failure never invalidates a saved checkpoint.

    def _validation(self):
        step = self.state['global_step']
        ledger = CaseLedger(self.run_dir / 'validation' / f'step_{step:08d}', dict(
            run_id=self.run_id, model_hash=weights_hash(self.model),
            run_identity_hash=digest(self.origin_identity), metric_protocol=METRIC_PROTOCOL,
            preprocessing_hash=digest(self.identity['preprocessing']),
            manifest_hash=self.identity['data']['manifest_hash'],
            split_hash=self.identity['data'].get('split_hash'),
            cases=self.validation_case_ids))
        clock = ProgressClock(self.options['eta_window'], self.options['eta_warmup'])
        records = []
        training_rng = capture_rng(self.device)
        self.console.validation_start(len(self.validation_case_ids), self.options['validation_role'])
        self.model.eval()
        try:
            for index, case in enumerate(self.validation_case_ids):
                data_identity = dict(case_id=case, manifest_hash=self.identity['data']['manifest_hash'],
                                     source=self.identity['data'].get('case_identity', {}).get(case))
                record = ledger.read(case, data_identity)
                if record is None:
                    self._sync()
                    started = time.perf_counter()
                    with torch.no_grad():
                        image, label = self._batch(self.validation_dataset, index, validation=True)
                        output, relation_stats = observed_forward(self.model, image)
                        record = hard_dice(output.final_logits.argmax(1)[0], label[0])
                        if relation_stats:
                            record['relation_scale'] = relation_stats
                        if self.mode in ('monai_reference_unet', 'monai_relation_unet', 'monai_coarse_aux'):
                            from .monai_reference import reference_diagnostics
                            if self.mode in ('monai_relation_unet', 'monai_coarse_aux'):
                                from .monai_relation import relation_diagnostics
                                record['diagnostic'] = relation_diagnostics(output, label, self.criterion, record)
                            else:
                                record['diagnostic'] = reference_diagnostics(
                                    output.final_logits, label, self.criterion, record)
                        elif self.mode == 'backbone_only_final':
                            record['diagnostic'] = final_diagnostic_metrics(
                                output.final_logits, label, self.criterion, record)
                        elif self.ce_reduction_mode == 'foreground_background_balanced':
                            record['ce_branches'] = joint_ce_diagnostic_metrics(output, label, self.criterion)
                        del image, label, output
                    ledger.commit(case, data_identity, record)
                    self._sync()
                    clock.add(time.perf_counter() - started)
                records.append(record)
                eta = clock.estimate(len(self.validation_case_ids) - index - 1)
                self.console.validation_case(case, eta)
        finally:
            self.model.train()
            restore_rng(training_rng, self.device)
            self.console.validation_end()
        row = dict(run_id=self.run_id, phase=self.options['validation_role'],
                   global_step=step, epoch=self.state['epoch'], metrics=summarize_dice(records))
        row['mode'] = self.mode
        row['ce_reduction_mode'] = self.ce_reduction_mode
        row['foreground_ce_reduction'] = self.foreground_ce_reduction
        row.update(self.ce_weights)
        if any('relation_scale' in record for record in records):
            row['relation_scale_cases'] = {case: record['relation_scale']
                for case, record in zip(self.validation_case_ids, records)}
        if self.mode in ('backbone_only_final', 'monai_reference_unet', 'monai_relation_unet', 'monai_coarse_aux'):
            row['diagnostic_cases'] = {case: record['diagnostic']
                                       for case, record in zip(self.validation_case_ids, records)}
        elif self.ce_reduction_mode == 'foreground_background_balanced':
            row['ce_diagnostic_cases'] = {case: record['ce_branches']
                                          for case, record in zip(self.validation_case_ids, records)}
        if self.epoch_mode:
            row['observation_profile'] = self.observation_profile
            row['epoch_metrics'] = summarize_branches([validation_branches(r) for r in records])
            score = row['metrics']['mean_case_dice']
            best = self.development_state['best_dev']
            improved = score is not None and (best is None or score > best['score'])
            row['best_dev_improved'] = improved
            if improved:
                self.development_state['best_dev'] = dict(score=score, epoch=self.state['epoch'], global_step=step)
            self.development_state['validation_history'].append(dict(
                epoch=self.state['epoch'], global_step=step, epoch_metrics=row['epoch_metrics'], improved=improved))
            if self.early_state is not None:
                self.development_state['validation_history'][-1]['monitor_score'] = score
                self.early_state = advance_early_stopping(self.early_state, score, self.state['epoch'], self.options)
                row['early_stopping'] = dict(self.early_state)
        self.log.append(row)
        self.board.record(row)
        self.state['pending_validation'] = False
        self.checkpoint()  # Same completed optimizer boundary, validation now committed.
        self.console.validation_summary(row)
        if self.early_state and self.early_state['stopped']:
            self.console.early_stop(self.early_state)

    def _start_observers(self):
        rng = capture_rng(self.device)
        try:
            last = None
            if self.console.enabled:
                with self.log.path.open(encoding='utf-8') as stream:
                    for line in stream:
                        row = json.loads(line)
                        if row['phase'] == 'train':
                            last = row
                next_case = 'complete'
                if self.state['global_step'] < self.total:
                    order = self.state['order']
                    if not order:
                        preview = torch.Generator().set_state(self.sampler_generator.get_state())
                        order = (torch.randperm(len(self.case_ids), generator=preview).tolist()
                                 if self.options['shuffle'] else list(range(len(self.case_ids))))
                    next_case = self.case_ids[order[self.state['cursor']]]
                self.console.start(self.identity, self.options, self.case_ids, self.total, self.epochs,
                                   self.state, resume=self.resume_path, next_case=next_case, last=last)
            self.board.start(self.state['global_step'], self.log.path)
        finally:
            # Lazy observer imports/initialization must not alter training RNG.
            restore_rng(rng, self.device)

    def run(self, *, stop_after=None):
        """Observers start after state restoration and close on every exit path."""
        try:
            if self.extension_pending:
                # Persist the new horizon at the already-completed optimizer
                # boundary before observers, pending validation or another step.
                self.checkpoint()
                self.extension_pending = False
            self._start_observers()
            start_step = self.state['global_step']
            self.report_steps = self.scalar_only and stop_after == 5
            result = self._run(stop_after=stop_after)
            if self.report_steps:
                from .console import short_run_summary
                with self.log.path.open(encoding='utf-8') as stream:
                    rows = [row for line in stream if (row := json.loads(line))['phase'] == 'train'
                            and row['global_step'] > start_step]
                if rows:
                    summary = dict(run_id=self.run_id, phase='short_run_summary', global_step=self.state['global_step'],
                                   **short_run_summary(rows))
                    self.log.append(summary)
                    self.checkpoint()
                    self.console.short_summary(summary)
            self.console.finish()
            return result
        finally:
            self.board.close()
            self.console.close()

    def _run(self, *, stop_after=None):
        """Optional invocation budget stops safely without changing the run identity."""
        if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
            raise ValueError('stop_after must be positive')
        start_step = self.state['global_step']
        if self.state['pending_validation']:
            self._validation()
        while self.state['global_step'] < self.total and not (self.early_state and self.early_state['stopped']):
            if not self.state['order']:
                self.state['order'] = (torch.randperm(len(self.dataset), generator=self.sampler_generator).tolist()
                                       if self.options['shuffle'] else list(range(len(self.dataset))))
            index = self.state['order'][self.state['cursor']]
            case, epoch = self.case_ids[index], self.state['epoch'] + 1
            step = self.state['global_step'] + 1
            self.console.begin_step(epoch, self.state['cursor'])
            self._sync()
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            image, label = self._batch(self.dataset, index)
            output, relation_stats = observed_forward(self.model, image)  # No supervision enters forward.
            loss_function = self.criterion.objective if self.scalar_only else self.criterion
            loss = (loss_function(output.coarse_logits, output.final_logits, label)
                    if self.mode in ('organ_relation_joint', 'monai_relation_unet', 'monai_coarse_aux') else loss_function(output.final_logits, label))
            loss.total.backward()
            epoch_complete = self.state['cursor'] + 1 == len(self.dataset)
            check = cadence_due(self.options, 'diagnostics_every', step, epoch, epoch_complete)
            diagnostics = gradient_diagnostics(self.model) if check else None
            before = {n: p.detach().clone() for n, p in self.model.named_parameters()} if check else None
            lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.step()
            if check:
                for name, parameters in diagnostic_parameter_groups(self.model).items():
                    if not parameters:
                        continue
                    if any(not torch.isfinite(p).all() for _, p in parameters):
                        raise ValueError(f'nonfinite updated parameters in {name}; failed step not committed')
                    change = sum((p.detach().double() - before[n].double()).square().sum()
                                 for n, p in parameters).sqrt()
                    diagnostics[name]['update_norm'] = change.item()
            row = dict(run_id=self.run_id, phase='train', epoch=epoch, total_epochs=self.epochs,
                       global_step=step, total_steps=self.total, case_id=case,
                       shape=list(image.shape[2:]), lr=lr, total_loss=loss.total.detach().item(),
                       final=({'segmentation': loss.final.detach().item()} if self.scalar_only else
                              {k: getattr(loss.final, k).detach().item() for k in ('ce', 'dice_loss', 'segmentation')}),
                       diagnostics=diagnostics)
            if not self.scalar_only:
                row['soft_dice_per_organ'] = loss.final.dice_per_class.detach().cpu()[0].tolist()
            if relation_stats:
                row['relation_scale'] = relation_stats  # Same forward's gamma, before optimizer.step.
                if self.scalar_only:
                    row['gamma_after_step'] = self.model.bottleneck.fusion.relation_scale.detach().item()
            if self.mode in ('monai_reference_unet', 'monai_relation_unet', 'monai_coarse_aux'):
                from ..models.monai_reference import padding_geometry
                row['geometry'] = padding_geometry(image.shape[2:])
            row['mode'] = self.mode
            row['ce_reduction_mode'] = self.ce_reduction_mode
            row['foreground_ce_reduction'] = self.foreground_ce_reduction
            row.update(self.ce_weights)
            if self.mode in ('organ_relation_joint', 'monai_relation_unet', 'monai_coarse_aux'):
                row['coarse'] = ({'segmentation': loss.coarse.detach().item()} if self.scalar_only else
                                {k: getattr(loss.coarse, k).detach().item() for k in ('ce', 'dice_loss', 'segmentation')})
            if self.epoch_mode:
                # Observe this same forward, before freeing it; no extra model
                # call or stochastic sampling. Store only case scalar records.
                with torch.no_grad():
                    branches = dict(total_loss=row['total_loss'], final=dict(
                        loss=row['final']['segmentation'],
                        hard=hard_dice(output.final_logits.argmax(1)[0], label[0])))
                    if not self.scalar_only:
                        branches['final']['soft_per_organ'] = row['soft_dice_per_organ']
                    if self.mode == 'monai_relation_unet':
                        branches['coarse'] = dict(loss=row['coarse']['segmentation'])
                        if not self.scalar_only:
                            coarse = self.criterion.align_coarse(output.coarse_logits, label)
                            branches['coarse'].update(soft_per_organ=loss.coarse.dice_per_class[0].detach().cpu().tolist(),
                                hard=hard_dice(coarse.argmax(1)[0], label[0]))
                            del coarse
                    if self.scalar_only and relation_stats:
                        branches['relation_scale'] = dict(relation_stats)
                        branches['gamma_after_step'] = row['gamma_after_step']
                row['branch_metrics'] = branches
                row['observation_profile'] = self.observation_profile
                self.development_state['epoch_cases'].append(branches)
            del image, label, output, loss, before
            if self.scalar_only:
                self.optimizer.zero_grad(set_to_none=True)
            self._sync()
            elapsed = time.perf_counter() - started
            row.update(step_seconds=elapsed, memory=self._memory())
            self.clock.add(elapsed)
            self.state['global_step'] = step
            self.state['cursor'] += 1
            if self.state['cursor'] == len(self.dataset):
                self.state.update(epoch=epoch, order=[], cursor=0)
            remaining = self.total - step
            epoch_remaining = min(remaining, len(self.dataset)-self.state['cursor']) if self.state['cursor'] else 0
            row['progress'] = self.clock.estimate(remaining, epoch_remaining)
            row['progress']['scope'] = 'training steps only; validation/checkpoint time excluded'
            self.log.append(row)
            self.board.record(row)
            self.console.train_step(row)
            if self.report_steps:
                self.console.short_step(row)
            if self.epoch_mode and epoch_complete:
                summary = dict(run_id=self.run_id, phase='train_epoch', epoch=epoch, global_step=step,
                    mode=self.mode, observation_profile=self.observation_profile,
                    epoch_metrics=summarize_branches(self.development_state['epoch_cases']))
                if self.scalar_only and relation_stats:
                    cases = self.development_state['epoch_cases']
                    summary['relation_scale'] = dict(gamma=row['gamma_after_step'],
                        **{key: sum(c['relation_scale'][key] for c in cases)/len(cases) for key in
                           ('writeback_to_feature_norm', 'scaled_writeback_to_feature_norm')})
                self.log.append(summary)
                self.board.record(summary)
                if self.scalar_only:
                    self.console.epoch_summary(summary)
                self.development_state['epoch_cases'] = []
            due = cadence_due(self.options, 'validation_every', step, epoch, epoch_complete)
            self.state['pending_validation'] = due
            invocation_done = stop_after is not None and step-start_step >= stop_after
            step_interval = self.options.get('checkpoint_every_steps')
            step_save = step_interval is not None and (step % step_interval == 0 or epoch_complete)
            if step_save or cadence_due(self.options, 'checkpoint_every', step, epoch, epoch_complete) or due or not remaining or invocation_done:
                self.checkpoint()
            if due:
                self._validation()
            if self.state['cursor'] == 0 or not remaining:
                self.console.epoch_end(monitored=due or (self.scalar_only and epoch_complete))
            if invocation_done:
                break
        return dict(self.state)

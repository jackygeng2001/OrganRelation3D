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
from .progress import ProgressClock
from .state import (ScalarLog, atomic_json, capture_rng, digest, load_checkpoint,
                    restore_rng, save_checkpoint)


def validate_options(options):
    for key in ('checkpoint_every', 'diagnostics_every', 'validation_every'):
        if type(options[key]) is not int or options[key] < (1 if key == 'checkpoint_every' else 0):
            raise ValueError(f'invalid {key}')
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


def weights_hash(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def gradient_diagnostics(model):
    result = {}
    for name, module in model.named_children():
        parameters = [p for p in module.parameters() if p.requires_grad]
        if not parameters:  # e.g. SpaceToNode has no learnable parameters.
            continue
        if any(p.grad is None for p in parameters):
            raise ValueError(f'missing gradients in {name}')
        if any(not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError(f'nonfinite gradients in {name}')
        norm = torch.stack([p.grad.detach().double().square().sum() for p in parameters]).sum().sqrt()
        result[name] = dict(gradient_norm=norm.item(), finite=True)
    return result


class Trainer:
    def __init__(self, model, criterion, optimizer, dataset, case_ids, *, device,
                 options, identity, run_dir, validation_dataset=None, validation_case_ids=(), resume=None):
        validate_options(options)
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
        self.run_dir = Path(run_dir)
        self.total = min(v for v in (options['max_steps'],
                         options['max_epochs'] * len(dataset) if options['max_epochs'] else None) if v is not None)
        self.epochs = math.ceil(self.total / len(dataset))
        self.clock = ProgressClock(options['eta_window'], options['eta_warmup'])
        self.sampler_generator = torch.Generator().manual_seed(options['seed'])
        self.loader_generator = torch.Generator().manual_seed(options['seed'] + 1)
        self.state = dict(global_step=0, epoch=0, order=[], cursor=0, pending_validation=False)
        if not resume and self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise ValueError('new run directory must be empty; use --resume')
        checkpoint = load_checkpoint(resume, identity) if resume else None
        if checkpoint:
            self._validate_progress(checkpoint)
            run = json.loads((self.run_dir / 'run.json').read_text(encoding='utf-8'))
            if run != dict(run_id=checkpoint['run_id'], identity=identity):
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
        save_checkpoint(self.run_dir / 'last.ckpt', dict(
            schema_version=1, run_id=self.run_id, identity=self.identity,
            model=self.model.state_dict(), optimizer=self.optimizer.state_dict(), scheduler=None,
            progress=self.state, rng=capture_rng(self.device),
            sampler_generator=self.sampler_generator.get_state(),
            loader_generator=self.loader_generator.get_state(),
            log=self.log.position(self.state['global_step'])))

    def _validation(self):
        step = self.state['global_step']
        ledger = CaseLedger(self.run_dir / 'validation' / f'step_{step:08d}', dict(
            run_id=self.run_id, model_hash=weights_hash(self.model),
            run_identity_hash=digest(self.identity), metric_protocol=METRIC_PROTOCOL,
            preprocessing_hash=digest(self.identity['preprocessing']),
            manifest_hash=self.identity['data']['manifest_hash'],
            split_hash=self.identity['data'].get('split_hash'),
            cases=self.validation_case_ids))
        clock = ProgressClock(self.options['eta_window'], self.options['eta_warmup'])
        records = []
        training_rng = capture_rng(self.device)
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
                        output = self.model(image)
                        record = hard_dice(output.final_logits.argmax(1)[0], label[0])
                        del image, label, output
                    ledger.commit(case, data_identity, record)
                    self._sync()
                    clock.add(time.perf_counter() - started)
                records.append(record)
                eta = clock.estimate(len(self.validation_case_ids) - index - 1)
                print(f'validation {index+1}/{len(self.validation_case_ids)} case={case} ETA={eta}', flush=True)
        finally:
            self.model.train()
            restore_rng(training_rng, self.device)
        self.log.append(dict(run_id=self.run_id, phase=self.options['validation_role'],
                             global_step=step, epoch=self.state['epoch'], metrics=summarize_dice(records)))
        self.state['pending_validation'] = False
        self.checkpoint()  # Same completed optimizer boundary, validation now committed.

    def run(self, *, stop_after=None):
        """Optional invocation budget stops safely without changing the run identity."""
        if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
            raise ValueError('stop_after must be positive')
        start_step = self.state['global_step']
        if self.state['pending_validation']:
            self._validation()
        while self.state['global_step'] < self.total:
            if not self.state['order']:
                self.state['order'] = (torch.randperm(len(self.dataset), generator=self.sampler_generator).tolist()
                                       if self.options['shuffle'] else list(range(len(self.dataset))))
            index = self.state['order'][self.state['cursor']]
            case, epoch = self.case_ids[index], self.state['epoch'] + 1
            step = self.state['global_step'] + 1
            self._sync()
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            image, label = self._batch(self.dataset, index)
            output = self.model(image)  # Supervision never enters model.forward.
            loss = self.criterion(output.coarse_logits, output.final_logits, label)
            loss.total.backward()
            check = bool(self.options['diagnostics_every'] and step % self.options['diagnostics_every'] == 0)
            diagnostics = gradient_diagnostics(self.model) if check else None
            before = {n: p.detach().clone() for n, p in self.model.named_parameters()} if check else None
            lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.step()
            if check:
                for name, module in self.model.named_children():
                    parameters = list(module.named_parameters())
                    if not parameters:
                        continue
                    if any(not torch.isfinite(p).all() for _, p in parameters):
                        raise ValueError(f'nonfinite updated parameters in {name}; failed step not committed')
                    change = sum((p.detach().double() - before[f'{name}.{n}'].double()).square().sum()
                                 for n, p in parameters).sqrt()
                    diagnostics[name]['update_norm'] = change.item()
            row = dict(run_id=self.run_id, phase='train', epoch=epoch, total_epochs=self.epochs,
                       global_step=step, total_steps=self.total, case_id=case,
                       shape=list(image.shape[2:]), lr=lr, total_loss=loss.total.detach().item(),
                       coarse={k: getattr(loss.coarse, k).detach().item() for k in ('ce', 'dice_loss', 'segmentation')},
                       final={k: getattr(loss.final, k).detach().item() for k in ('ce', 'dice_loss', 'segmentation')},
                       soft_dice_per_organ=loss.final.dice_per_class.detach().cpu()[0].tolist(),
                       diagnostics=diagnostics)
            del image, label, output, loss, before
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
            print(f'epoch {epoch}/{self.epochs} step {step}/{self.total} case={case} shape={row["shape"]} '
                  f'lr={lr:g} loss={row["total_loss"]:.6f} time={elapsed:.3f}s '
                  f'progress={row["progress"]} memory={row["memory"]}', flush=True)
            due = bool(self.options['validation_every'] and step % self.options['validation_every'] == 0)
            self.state['pending_validation'] = due
            invocation_done = stop_after is not None and step-start_step >= stop_after
            if step % self.options['checkpoint_every'] == 0 or due or not remaining or invocation_done:
                self.checkpoint()
            if due:
                self._validation()
            if invocation_done:
                break
        return dict(self.state)

"""BraTS 三区域整例训练循环：串行、可严格续训，覆盖过拟合、pilot 与五折正式训练。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import torch
from torch.utils.data import DataLoader

from ..data import augmentation as augment
from ..evaluation.progress import CaseLedger
from .console import TrainingConsole
from .progress import ProgressClock
from .extension import resolve_horizon
from .state import (ScalarLog, atomic_json, capture_rng, digest, load_checkpoint,
                    restore_rng, save_checkpoint)
from .tensorboard import TensorBoardObserver
from .development import cadence_due, validate_early_stopping, new_early_state, advance_early_stopping
from . import brain_tumor as regions  # 验证打分经模块属性调用，测试可以替换它
from .brain_tumor import (PROTOCOL, region_training_branches, region_validation_branches,
                          summarize_region_branches)


def prefetch_safe(dataset):
    """预读只用于"读一例是纯函数、不消耗任何随机数"的数据集：BraTSDataset，或显式声明 prefetch_safe 的数据集。"""
    from ..data.brats import BraTSDataset
    return isinstance(dataset, BraTSDataset) or getattr(dataset, 'prefetch_safe', False) is True


class _Prefetched(torch.utils.data.Dataset):
    """把已预读的一例交给读取器；读取器照常创建，随机数消耗与现读时相同。"""

    def __init__(self, index, sample):
        self.index, self.sample = index, sample

    def __getitem__(self, index):
        if index != self.index:
            raise IndexError('prefetched sample requested for another index')
        return self.sample


def _collate_one(items):
    # batch=1：等价于默认 collate 的 stack（数值、形状、连续性相同），但不复制，保留锁页内存。
    (sample,) = items
    return sample._replace(image=sample.image.unsqueeze(0), label=sample.label.unsqueeze(0))


class NextCase:
    """单槽预读：GPU 计算当前这一例时，后台线程提前读出下一例。

    只提前做"读出这一例"这件纯函数的工作；何时取用、读取器随机数、送入模型的数值都与不预读时逐位相同，
    checkpoint 不含任何预读状态。后台读取出错时，错误在主线程取用这一例时原样抛出。
    """

    def __init__(self, *, pin_memory):
        self.pin_memory = pin_memory
        self.pool = None  # 首次预读时创建，close 后可再次创建
        self.key = self.future = None

    def _load(self, dataset, index):
        sample = dataset[index]
        if self.pin_memory:  # 锁页内存让拷进显存的那一步可以异步进行
            sample = sample._replace(image=sample.image.pin_memory(), label=sample.label.pin_memory())
        return sample

    def submit(self, dataset, index):
        self.discard()
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='next-case')
        self.key = (id(dataset), index)
        self.future = self.pool.submit(self._load, dataset, index)

    def take(self, dataset, index):
        """命中返回预读结果；未命中返回 None，由调用方现读。"""
        if self.future is None or self.key != (id(dataset), index):
            self.discard()
            return None
        future, self.key, self.future = self.future, None, None
        return future.result()

    def discard(self):
        if self.future is not None and not self.future.cancel():
            # 已在运行的读取无法取消：等它结束再丢弃，保证同一时刻只有一个线程读数据集；
            # 它的异常也丢弃，需要这一例时主线程会现读并重新抛出。
            try:
                self.future.result()
            except Exception:
                pass
        self.key = self.future = None

    def close(self):
        self.discard()
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None


def validate_options(options):
    if options.get('loss_observation', 'legacy') != 'legacy':
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
    if options.get('convolution_compute_dtype') not in (None, 'bfloat16'):
        # float16 would underflow per-voxel gradients without loss scaling, which is not implemented.
        raise ValueError('convolution_compute_dtype must be null or bfloat16 for training')
    if options.get('mixed_precision') not in (None, 'bfloat16'):
        raise ValueError('mixed_precision must be null or bfloat16 for training')
    if options.get('mixed_precision') is not None and options.get('convolution_compute_dtype') is not None:
        raise ValueError('choose either convolution_compute_dtype or whole-network mixed_precision')
    if type(options['seed']) is not int or type(options['shuffle']) is not bool:
        raise ValueError('explicit seed and shuffle required')
    augment.validate_augmentation(options.get('augmentation'))
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


class Trainer:
    def __init__(self, model, criterion, optimizer, dataset, case_ids, *, device,
                 options, identity, run_dir, validation_dataset=None, validation_case_ids=(), resume=None,
                 console=None, tensorboard=False, extend_to=None, prefetch=False):
        validate_options(options)
        if type(prefetch) is not bool:
            raise ValueError('prefetch must be a boolean')
        if prefetch and not all(prefetch_safe(d) for d in (dataset, validation_dataset) if d is not None):
            raise ValueError('prefetch requires datasets whose loading is pure and consumes no random numbers')
        if identity.get('training') != options:
            raise ValueError('training options must match the strict run identity')
        self.mode = identity.get('mode')
        if self.mode != 'brats_regions':
            raise ValueError('training is defined only for BraTS region volumes (mode brats_regions)')
        self.augmentation = options.get('augmentation')
        from ..models.brain_tumor import BrainTumorSegmentor
        from ..losses import RegionJointLoss
        expected_loss = (dict(epsilon=criterion.epsilon,lambda_c=criterion.lambda_c,align_corners=criterion.align_corners)
                         if isinstance(criterion, RegionJointLoss) else None)
        if expected_loss is not None and criterion.lambda_hier is not None:
            expected_loss['lambda_hier'] = criterion.lambda_hier  # Eq. (33); absent keeps legacy identity.
        if (not isinstance(model, BrainTumorSegmentor) or not isinstance(criterion, RegionJointLoss)
                or model.config.to_dict() != identity.get('model')
                or identity.get('loss') != expected_loss
                or not 0 < identity.get('threshold',0) < 1):
            raise ValueError('brain model/loss/threshold must match strict identity')
        # 以下三个日志字段沿用历史格式，保证新旧 run 的日志可直接对比。
        self.ce_reduction_mode, self.foreground_ce_reduction = 'region_probability_bce', 'not_applicable'
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
        self.next_case = NextCase(pin_memory=self.device.type == 'cuda') if prefetch else None
        self.epoch_mode = options.get('cadence_unit') == 'epoch'
        self.observation_profile = 'development_v1'
        if options.get('early_stopping') and options['early_stopping']['monitor'] != 'dev_mean_region_hard_dice':
            raise ValueError('early stopping monitor does not match model region semantics')
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
        # 每步只建一个读取器：读取器随机数对每个已提交样本恰好消耗一次（含中途续跑）。
        # 没有多进程读取；可选的预读（NextCase）不进 checkpoint，也不改变这一点。
        generator = torch.Generator().manual_seed(0) if validation else self.loader_generator
        if self.next_case is None:
            batch = next(iter(DataLoader(dataset, batch_size=1, sampler=[index], num_workers=0,
                                         generator=generator)))
            image, label = batch.image.to(self.device), batch.label.to(self.device)
        else:
            # 预读命中时读取器照常创建（随机数消耗不变），只是不再重复读盘；数值与上面的分支逐位相同。
            sample = self.next_case.take(dataset, index)
            source = dataset if sample is None else _Prefetched(index, sample)
            batch = next(iter(DataLoader(source, batch_size=1, sampler=[index], num_workers=0,
                                         generator=generator, collate_fn=_collate_one)))
            image = batch.image.to(self.device, non_blocking=True)
            label = batch.label.to(self.device, non_blocking=True)
        if self.options['memory_format'] == 'channels_last_3d':
            image = image.contiguous(memory_format=torch.channels_last_3d)
        return image, label

    def checkpoint(self, *, save_best=True):
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
            if save_best and best and best['global_step'] == self.state['global_step'] and not self.state['pending_validation']:
                save_checkpoint(self.run_dir / 'best-dev.ckpt', payload)
        save_checkpoint(self.run_dir / 'last.ckpt', payload)
        self.board.flush()  # Observer failure never invalidates a saved checkpoint.

    def _validation(self):
        step = self.state['global_step']
        ledger = CaseLedger(self.run_dir / 'validation' / f'step_{step:08d}', dict(
            run_id=self.run_id, model_hash=weights_hash(self.model),
            run_identity_hash=digest(self.origin_identity), metric_protocol=PROTOCOL,
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
                        if self.next_case is not None and index + 1 < len(self.validation_case_ids):
                            self.next_case.submit(self.validation_dataset, index + 1)
                        output = self.model(image)
                        record = regions.region_dice(output.final_logits,label,self.identity['threshold'])
                        record['region_diagnostic'] = regions.region_diagnostics(output,label,self.criterion,self.identity['threshold'])
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
        summary = regions.summarize_regions(records)
        row = dict(run_id=self.run_id, phase=self.options['validation_role'],
                   global_step=step, epoch=self.state['epoch'], metrics=summary)
        row['mode'] = self.mode
        row['region_diagnostic_cases'] = {case:record['region_diagnostic'] for case,record in zip(self.validation_case_ids,records)}
        row['ce_reduction_mode'] = self.ce_reduction_mode
        row['foreground_ce_reduction'] = self.foreground_ce_reduction
        if self.epoch_mode:
            row['observation_profile'] = self.observation_profile
            row['epoch_metrics'] = summarize_region_branches([region_validation_branches(r) for r in records])
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
            last, trained_seconds = None, 0.0
            if self.console.enabled:
                with self.log.path.open(encoding='utf-8') as stream:
                    for line in stream:
                        row = json.loads(line)
                        if row['phase'] == 'train':
                            last = row
                            trained_seconds += row.get('step_seconds') or 0.0
                next_case = 'complete'
                if self.state['global_step'] < self.total:
                    order = self.state['order']
                    if not order:
                        preview = torch.Generator().set_state(self.sampler_generator.get_state())
                        order = (torch.randperm(len(self.case_ids), generator=preview).tolist()
                                 if self.options['shuffle'] else list(range(len(self.case_ids))))
                    next_case = self.case_ids[order[self.state['cursor']]]
                # 以下只用于显示：运行目录名、参数量、已有最佳与累计训练时间（全部来自已提交的状态与日志）
                self.console.start(self.identity, self.options, self.case_ids, self.total, self.epochs,
                                   self.state, resume=self.resume_path, next_case=next_case, last=last,
                                   run_name=self.run_dir.name, parameters=sum(p.numel() for p in self.model.parameters()),
                                   best=self.development_state.get('best_dev'), trained_seconds=trained_seconds)
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
            result = self._run(stop_after=stop_after)
            self.console.finish()
            return result
        finally:
            if self.next_case is not None:
                self.next_case.close()
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
            following = self.state['cursor'] + 1
            if self.next_case is not None and following < len(self.state['order']) and step < self.total:
                # 本轮内的下一例已确定，后台先读；跨轮的顺序要在下一步才抽取，那一例照常现读。
                self.next_case.submit(self.dataset, self.state['order'][following])
            augmentation = None
            if self.augmentation is not None:  # Training cases only; drawn from (seed, step), never from labels.
                augmentation = augment.draw(self.augmentation, augment.step_generator(self.options['seed'], step),
                                            image.shape[1])
                image, label = augment.apply(image, label, augmentation)
                if self.options['memory_format'] == 'channels_last_3d':
                    image = image.contiguous(memory_format=torch.channels_last_3d)
            output = self.model(image)  # No supervision enters forward.
            loss = self.criterion(output.coarse_logits, output.final_logits, label)
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
                       final={k: getattr(loss.final, k).detach().item() for k in ('ce', 'dice_loss', 'segmentation')},
                       diagnostics=diagnostics)
            row['soft_dice_per_organ'] = loss.final.dice_per_class.detach().cpu()[0].tolist()  # 依次为 WT/TC/ET（字段名沿用历史格式）
            row['mode'] = self.mode
            if augmentation is not None:
                row['augmentation'] = augmentation
            row['ce_reduction_mode'] = self.ce_reduction_mode
            row['foreground_ce_reduction'] = self.foreground_ce_reduction
            row['coarse'] = {k: getattr(loss.coarse, k).detach().item() for k in ('ce', 'dice_loss', 'segmentation')}
            for branch in ('final','coarse'):
                row[branch]['bce'] = row[branch].pop('ce')
            if getattr(loss, 'hierarchy', None) is not None:  # Eq. (32), before lambda_hier.
                row['hierarchy'] = loss.hierarchy.detach().item()
            if self.epoch_mode:
                # Observe this same forward, before freeing it; no extra model
                # call or stochastic sampling. Store only case scalar records.
                branches = region_training_branches(loss)
                row['branch_metrics'] = branches
                row['observation_profile'] = self.observation_profile
                self.development_state['epoch_cases'].append(branches)
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
            self.board.record(row)
            self.console.train_step(row)
            if self.epoch_mode and epoch_complete:
                summary = dict(run_id=self.run_id, phase='train_epoch', epoch=epoch, global_step=step,
                    mode=self.mode, observation_profile=self.observation_profile,
                    epoch_metrics=summarize_region_branches(self.development_state['epoch_cases']))
                self.log.append(summary)
                self.board.record(summary)
                self.console.train_epoch(summary)
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
                self.console.epoch_end(monitored=due)
            if invocation_done:
                break
        return dict(self.state)

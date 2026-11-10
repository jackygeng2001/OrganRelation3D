"""真实 TensorBoard 事件文件与控制台输出；观测层失败或开关不改变训练轨迹（CPU）。"""
import contextlib
import io
import json
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import support as training_fixtures
from organ_relation.training.console import WIDTH, TrainingConsole, duration, report_row, spread
from organ_relation.training.state import load_checkpoint, save_checkpoint, capture_rng
from organ_relation.training.tensorboard import TensorBoardObserver, scalar_values


class Terminal(io.StringIO):
    def __init__(self, tty=False):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


CASE = 'Brats18_CBICA_AAA_1'


def branch(bce, dice):
    return {'bce': bce, 'dice_loss': dice, 'segmentation': bce + dice}


def train_row(step=1):
    return dict(phase='train', mode='brats_regions', global_step=step, epoch=step, total_epochs=100, total_steps=100,
        case_id=CASE, total_loss=3.284, lr=3e-4, shape=[10, 11, 12],
        coarse=branch(.2, .8), final=branch(.15, .7), soft_dice_per_organ=[.3, .2, .1], step_seconds=60.,
        memory={'peak_allocated_bytes': 8 * 2**30, 'peak_reserved_bytes': 9 * 2**30},
        progress={'eta_seconds': 3600., 'rolling_seconds': 60.},
        diagnostics={'encoder': {'gradient_norm': .25, 'finite': True, 'update_norm': .01}})


def validation_row(step=1, phase='train_monitor'):
    case = {name: dict(bce=.1, dice_loss=.3, loss=.4, soft_dice=[.7, .6, .5]) for name in ('final', 'coarse')}
    return dict(phase=phase, mode='brats_regions', global_step=step, epoch=step,
        metrics={'mean_case_dice': .7, 'organs': [dict(label=i + 1, name=n, mean_dice=.7 if n == 'WT' else None)
                                                for i, n in enumerate(('WT', 'TC', 'ET'))]},
        region_diagnostic_cases={CASE: case})


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.fixture = training_fixtures.TrainingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def events(self, directory):
        return EventAccumulator(str(directory), size_guidance={'scalars': 0}).Reload()

    def test_20_to_40_extension_exact_trajectory_rng_and_unique_tensorboard_steps(self):
        options = dict(max_steps=40, checkpoint_every=5, diagnostics_every=5, validation_every=5)
        whole = self.fixture.trainer(self.root / 'whole', validation=True, options=options)
        self.fixture.quiet_run(whole)
        expected = load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)
        options['max_steps'] = 20
        first = self.fixture.trainer(self.root / 'extended', validation=True, options=options)
        first.board = TensorBoardObserver(first.run_dir)
        self.fixture.quiet_run(first)
        path = first.run_dir / 'last.ckpt'
        middle = load_checkpoint(path, first.identity)
        # Exercise the real pre-extension checkpoint schema as well.
        middle.pop('horizon'); middle.pop('origin_identity')
        save_checkpoint(path, middle)
        original_run = (first.run_dir / 'run.json').read_bytes()
        original_log = first.log.path.read_bytes()
        resumed = self.fixture.trainer(first.run_dir, resume=True, validation=True,
                                       options=options, extend_to=40)
        self.assertEqual(resumed.run_id, first.run_id)
        self.assertEqual(resumed.total, 40)
        self.assertEqual(resumed.options['max_steps'], 20)
        self.assertEqual(len(resumed.clock.samples), 0)
        self.fixture.assert_nested_equal(middle['model'], resumed.model.state_dict())
        self.fixture.assert_nested_equal(middle['optimizer'], resumed.optimizer.state_dict())
        self.fixture.assert_nested_equal(middle['rng'], capture_rng(resumed.device))
        self.fixture.assert_nested_equal(middle['progress'], resumed.state)
        self.fixture.assert_nested_equal(middle['sampler_generator'], resumed.sampler_generator.get_state())
        self.fixture.assert_nested_equal(middle['loader_generator'], resumed.loader_generator.get_state())
        resumed.board = TensorBoardObserver(resumed.run_dir)
        self.fixture.quiet_run(resumed)
        actual = load_checkpoint(path, resumed.identity)
        for key in ('model', 'optimizer', 'progress', 'rng', 'sampler_generator', 'loader_generator'):
            self.fixture.assert_nested_equal(expected[key], actual[key])
        self.assertEqual((first.run_dir / 'run.json').read_bytes(), original_run)
        self.assertTrue(first.log.path.read_bytes().startswith(original_log))
        rows, reference = self.fixture.rows(first.run_dir), self.fixture.rows(whole.run_dir)
        self.assertEqual(len(rows), len(reference))
        for a, b in zip(rows, reference):
            for key in ('phase', 'epoch', 'global_step', 'case_id', 'total_loss', 'coarse', 'final',
                        'soft_dice_per_organ', 'metrics', 'diagnostics'):
                self.assertEqual(a.get(key), b.get(key), key)
        training = [r for r in rows if r['phase'] == 'train']
        self.assertEqual([r['global_step'] for r in training], list(range(1, 41)))
        self.assertEqual(training[20]['total_steps'], 40)
        self.assertEqual(training[20]['progress']['status'], 'warming up')
        self.assertAlmostEqual(training[25]['progress']['eta_seconds'],
                               training[25]['progress']['rolling_seconds'] * 14)
        self.assertEqual(training[-1]['progress']['eta_seconds'], 0)
        history = actual['horizon']
        self.assertEqual((history['original_total_steps'], history['total_steps']), (20, 40))
        self.assertEqual(len(history['extensions']), 1)
        event = history['extensions'][0]
        self.assertEqual((event['at_global_step'], event['previous_total_steps'], event['total_steps']), (20, 20, 40))
        self.assertEqual(event['previous_provenance'], first.identity['provenance'])
        self.assertEqual(event['execution_provenance'], resumed.identity['provenance'])
        self.assertEqual(len(event['parent_checkpoint_sha256']), 64)
        self.assertFalse(resumed.board.failed)
        events = self.events(resumed.board.directory)
        self.assertEqual([e.step for e in events.Scalars('Train/Total_Loss')], list(range(1, 41)))
        self.assertEqual([e.step for e in events.Scalars('Monitor/HardDice_Mean')], list(range(5, 41, 5)))

    def start_console(self, stream, *, resume=False, cases=1, epoch_mode=False):
        console = TrainingConsole(stream=stream)
        identity = dict(model=dict(training_fixtures.config()['model'], graph='nested_chain'),
                        preprocessing={'spatial_crop': {'kind': 'nonzero_bbox', 'margin': 0, 'multiple_of': 8}},
                        loss={'lambda_c': .5, 'lambda_hier': 1.0},
                        optimizer={'name': 'AdamW', 'lr': 3e-4, 'weight_decay': 0},
                        data={'manifest': {'cohort': 'brats2018', 'source_partition': 'training'},
                              'split': {'fold': 2, 'folds': 5, 'dev': ['x'] * 57}},
                        provenance={'git': {'commit': 'f58c4cd8abfe', 'dirty': False}},
                        environment={'gpu_name': 'RX 7900 XTX', 'rocm': '7.2.26015', 'torch': '2.11.0+rocm7.2'},
                        runtime={'dtype': 'float32', 'convolution_backend': 'depth_sliced',
                                 'data_cache': 'brats_npy_v1', 'data_prefetch': 'next_case_thread_v1'})
        options = dict(training_fixtures.config()['training'])
        if epoch_mode:
            options.update(cadence_unit='epoch', validation_every=1, checkpoint_every_steps=25, console_every_steps=25,
                           early_stopping=dict(min_epochs=20, patience_epochs=20, min_delta=.001, monitor='x'))
        console.start(identity, options, [CASE] * cases, 100, 100,
                      dict(global_step=25 if resume else 0, epoch=25 if resume else 0),
                      resume='run/last.ckpt' if resume else None, next_case=CASE,
                      run_name='brats2018_r4_full_fold2', parameters=8486001)
        return console

    def test_report_lines_share_one_right_edge(self):
        lines = [report_row('[Epoch 012/150]', 'Train Loss', '0.8631', 'Val Dice', '0.7712'),
                 report_row('', 'Lr', '3.00e-04', 'GPU Peak', '12.34 GiB'),
                 report_row('', 'Fold', '2/5', 'Time', '4m 21s'),
                 report_row('', 'Fold', 'x' * 30, 'Time', 'y' * 30),
                 spread('[SAVE] Best 0.7712 (+0.0021), checkpoint saved', 'ETA 3h12m'),
                 spread('[WARN] ' + 'z' * 80, 'ETA 3h12m')]
        for line in lines:
            self.assertEqual(len(line), WIDTH, line)
        self.assertTrue(lines[1].endswith(' 12.34 GiB'))
        self.assertTrue(lines[4].startswith('[SAVE] Best') and lines[4].endswith('  ETA 3h12m'))
        self.assertTrue(lines[5].endswith('...  ETA 3h12m'))
        self.assertEqual(len({line.index('|', 10) for line in lines[:3]}), 1)  # 悬挂缩进：各行竖线对齐

    def test_tqdm_tty_and_redirected_output_flags(self):
        for tty in (False, True):
            stream = Terminal(tty)
            with patch('tqdm.tqdm', return_value=MagicMock()) as factory:
                console = self.start_console(stream)
                kwargs = factory.call_args.kwargs
                self.assertIs(kwargs['file'], stream)
                self.assertEqual(kwargs['disable'], not tty)
                self.assertFalse(kwargs['leave'])
                self.assertEqual(kwargs['ascii'], ' =')  # 只用单宽 ASCII 字符
                self.assertLessEqual(kwargs['ncols'], 100)
                self.assertEqual(kwargs['desc'], 'Fold 2/5 Train')
                console.close()

    def test_checklist_resume_and_no_raw_debug_dicts(self):
        stream = Terminal()
        console = self.start_console(stream, resume=True)
        console.train_step(train_row(26))
        console.validation_start(1, 'train_monitor')
        console.validation_case(CASE, {'eta_seconds': 0})
        console.validation_end()
        console.validation_summary(validation_row(26))
        console.close()
        text = stream.getvalue()
        for value in ('Pre-flight Checklist', '[INFO] Run        : brats2018_r4_full_fold2', '[INFO] Fold       : 2/5',
                      'RX 7900 XTX, ROCm 7.2', 'f58c4cd (clean), PyTorch 2.11.0', 'brats2018 training, 1 train / 57 val',
                      'crop nonzero_bbox x8, cache + prefetch', 'full method, nested_chain graph', '8,486,001',
                      'final + 0.5 coarse + 1.0 hierarchy', '[INFO] Resume     : epoch 26/100, step 25/100',
                      f'[INFO] Next Case  : {CASE}', '\n[Training Started]\n', 'Mon Dice', '0.7000', '[Step 26]'):
            self.assertIn(value, text)
        for value in ('Val Dice', 'gradient_norm', 'pending_validation', 'eta_seconds', "{'", '\r'):
            self.assertNotIn(value, text)
        self.assertTrue(all(len(line) <= WIDTH for line in text.splitlines()))

    def test_epoch_reports_save_warn_stop_with_fold_and_one_right_edge(self):
        stream = Terminal()
        console = self.start_console(stream, cases=3, epoch_mode=True)
        early = dict(best_metric=None, best_epoch=None, no_improvement_count=0, last_validation_epoch=0, stopped=False)
        with patch('tqdm.tqdm', return_value=MagicMock()) as factory:
            factory.write.side_effect = lambda text, file: print(text, file=file)
            for epoch, dice, improved in ((1, .70, True), (2, .71, True), (3, .705, False), (4, .7105, True)):
                console.begin_step(epoch, 0)
                self.assertEqual(factory.call_args.kwargs['desc'], f'Fold 2/5 Epoch {epoch}/100')
                for step in range(3):
                    row = train_row(3 * (epoch - 1) + step + 1); row['epoch'] = epoch
                    console.train_step(row)
                console.train_epoch(dict(phase='train_epoch', epoch=epoch, epoch_metrics={'total_loss': 2. - epoch / 10}))
                console.validation_start(57, 'internal_dev')
                self.assertEqual(factory.call_args.kwargs['desc'], 'Fold 2/5 Val')
                console.validation_end()
                count = 0 if improved and dice - (early['best_metric'] or 0) >= .001 else early['no_improvement_count'] + 1
                early = dict(early, best_metric=max(dice, early['best_metric'] or 0), no_improvement_count=count,
                             best_epoch=epoch if improved else early['best_epoch'], last_validation_epoch=epoch)
                row = dict(validation_row(3 * epoch, 'internal_dev'), epoch=epoch, best_dev_improved=improved,
                           early_stopping=early)
                row['metrics'] = dict(row['metrics'], mean_case_dice=dice)
                console.validation_summary(row)
                console.epoch_end(monitored=True)
            console.early_stop(dict(early, stopped=True))
            console.close()
        text = stream.getvalue()
        blocks = [line for line in text.splitlines() if line.startswith('[Epoch ')]
        self.assertEqual([b[:15] for b in blocks], [f'[Epoch 00{e}/100]' for e in range(1, 5)])
        for value in ('Train Loss : 1.9000', 'Val Dice :    0.7000', 'GPU Peak :  8.00 GiB', 'Fold       : 2/5',
                      '[SAVE] Best 0.7000, checkpoint saved', '[SAVE] Best 0.7100 (+0.0100), checkpoint saved',
                      '[WARN] No gain 1/20, best 0.7100 @ epoch 2', '[SAVE] Best 0.7105 (+0.0005), saved; no gain 2/20',
                      '[STOP] Early stopping at epoch 4', 'Fold 2/5', 'Best Val Dice 0.7105 @ epoch 4', 'ETA 1h 00m'):
            self.assertIn(value, text)
        report = text[text.index('[Training Started]'):].splitlines()[2:]
        for line in report:
            if line:  # 报告与分隔线一律正好 WIDTH 列，空行分隔各轮
                self.assertEqual(len(line), WIDTH, line)
        self.assertEqual(text.count('=' * WIDTH + '\n\n' + '=' * WIDTH), 4)

    def test_step_mode_prints_one_line_every_n_steps_and_done(self):
        stream = Terminal()
        console = self.start_console(stream, cases=3)
        console.console_every_steps = 5
        for step in range(1, 11):
            console.train_step(train_row(step))
        self.assertEqual(stream.getvalue().count('[Step '), 2)
        console.total = 10
        console.finish(); console.close()
        self.assertIn('[DONE] Reached 10 steps', stream.getvalue())

    def test_time_format_only_presents_existing_eta(self):
        for value, expected in [(None, 'warming up'), (0, '0s'), (1902, '31m 42s'),
                                (5160, '1h 26m'), (165600, '1d 22h')]:
            self.assertEqual(duration(value), expected)

    def test_real_tensorboard_scalar_mapping_uses_region_names(self):
        observer = TensorBoardObserver(self.root)
        observer.start(0, self.root / 'unused')
        observer.record(train_row())
        observer.record(validation_row())
        observer.record(validation_row(2, 'internal_dev'))
        observer.close()
        events = self.events(observer.directory)
        for row in (train_row(), validation_row(), validation_row(2, 'internal_dev')):
            for tag, value in scalar_values(row).items():
                self.assertAlmostEqual(events.Scalars(tag)[0].value, value, places=6)
        self.assertAlmostEqual(events.Scalars('Train/final/bce')[0].value, .15)
        self.assertAlmostEqual(events.Scalars('Monitor/HardDice_Mean')[0].value, .7)
        self.assertEqual(len(events.Scalars('Monitor/HardDice/WT')), 1)
        self.assertNotIn('Monitor/HardDice/TC', events.Tags()['scalars'])  # 全为空的区域不写入
        self.assertEqual(len(events.Scalars('Val/HardDice/WT')), 1)
        self.assertTrue(all(name in ('WT', 'TC', 'ET') for tag in events.Tags()['scalars'] if 'SoftDice/' in tag
                            for name in [tag.rsplit('/', 1)[1]]))
        for kind in ('images', 'histograms', 'tensors'):
            self.assertFalse(events.Tags()[kind])

    def test_train_rows_do_not_emit_validation_tags(self):
        tags = scalar_values(train_row())
        self.assertFalse(any(t.startswith(('Monitor/', 'Val/')) for t in tags))

    def test_purge_replays_committed_k_removes_future_and_uncommitted_same_step_val(self):
        log = self.root / 'metrics.jsonl'
        committed = [train_row(1), train_row(2)]
        log.write_text(''.join(json.dumps(r) + '\n' for r in committed))
        old = TensorBoardObserver(self.root); old.start(0, log)
        for row in committed + [validation_row(2), train_row(3), train_row(4)]:
            old.record(row)
        old.close()
        old_time = max(int(p.name.split('.')[3]) for p in old.directory.glob('events.out.tfevents.*'))
        resumed = TensorBoardObserver(self.root); resumed.start(2, log)
        new_time = max(int(p.name.split('.')[3]) for p in resumed.directory.glob('events.out.tfevents.*'))
        self.assertGreater(new_time, old_time)
        events = self.events(resumed.directory)
        self.assertEqual([e.step for e in events.Scalars('Train/Total_Loss')], [1, 2])
        self.assertEqual(events.Scalars('Monitor/HardDice_Mean'), [])
        resumed.record(validation_row(2)); resumed.record(train_row(3)); resumed.close()
        events.Reload()
        self.assertEqual([e.step for e in events.Scalars('Train/Total_Loss')], [1, 2, 3])
        self.assertEqual([e.step for e in events.Scalars('Monitor/HardDice_Mean')], [2])

    def test_committed_validation_at_k_is_replayed_once(self):
        log = self.root / 'metrics.jsonl'; rows = [train_row(1), validation_row(1)]
        log.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        for restart in range(3):
            observer = TensorBoardObserver(self.root); observer.start(1 if restart else 0, log)
            if not restart:
                for row in rows:
                    observer.record(row)
            observer.close()
        events = self.events(observer.directory)
        self.assertEqual([e.step for e in events.Scalars('Monitor/HardDice_Mean')], [1])
        self.assertEqual([e.step for e in events.Scalars('Train/Total_Loss')], [1])

    def test_disabled_observer_does_not_create_events(self):
        observer = TensorBoardObserver(self.root, enabled=False)
        with patch('torch.utils.tensorboard.SummaryWriter', side_effect=AssertionError('disabled')):
            observer.start(0, self.root / 'unused'); observer.record(train_row()); observer.flush(); observer.close()
        self.assertFalse(observer.directory.exists())

    def compare_trainers(self, left, right):
        a = load_checkpoint(left.run_dir / 'last.ckpt', left.identity)
        b = load_checkpoint(right.run_dir / 'last.ckpt', right.identity)
        for key in ('model', 'optimizer', 'rng', 'sampler_generator', 'loader_generator', 'progress', 'identity'):
            self.fixture.assert_nested_equal(a[key], b[key])
        rows_a, rows_b = self.fixture.rows(left.run_dir), self.fixture.rows(right.run_dir)
        self.assertEqual(len(rows_a), len(rows_b))
        for x, y in zip(rows_a, rows_b):
            self.assertEqual(x.keys(), y.keys())
            for key in x.keys() - {'run_id', 'step_seconds', 'progress'}:
                self.assertEqual(x[key], y[key], key)

    def test_tty_non_tty_and_tensorboard_enabled_disabled_preserve_training_and_rng(self):
        baseline = self.fixture.trainer(self.root / 'off', validation=True)
        self.fixture.quiet_run(baseline)
        for tty in (False, True):
            observed = self.fixture.trainer(self.root / str(tty), validation=True)
            observed.console = TrainingConsole(stream=Terminal(tty))
            observed.board = TensorBoardObserver(observed.run_dir)
            self.fixture.quiet_run(observed)
            self.compare_trainers(baseline, observed)

    def test_enabled_resume_preserves_trajectory_and_unique_event_steps(self):
        whole = self.fixture.trainer(self.root / 'whole', validation=True)
        self.fixture.quiet_run(whole)
        split = self.fixture.trainer(self.root / 'split', validation=True)
        split.board = TensorBoardObserver(split.run_dir)
        self.fixture.quiet_run(split, stop_after=3)
        resumed = self.fixture.trainer(split.run_dir, resume=True, validation=True)
        resumed.board = TensorBoardObserver(split.run_dir)
        resumed.console = TrainingConsole(stream=Terminal(True))
        self.fixture.quiet_run(resumed)
        self.compare_trainers(whole, resumed)
        events = self.events(resumed.board.directory)
        self.assertEqual([e.step for e in events.Scalars('Train/Total_Loss')], list(range(1, 8)))
        self.assertEqual([e.step for e in events.Scalars('Monitor/HardDice_Mean')], [2, 4, 6])

    def test_flush_at_checkpoint_and_close_on_controlled_stop(self):
        trainer = self.fixture.trainer(self.root / 'run')
        trainer.board = TensorBoardObserver(trainer.run_dir)
        writer = MagicMock()
        with patch('torch.utils.tensorboard.SummaryWriter', return_value=writer):
            self.fixture.quiet_run(trainer, stop_after=2)
        self.assertEqual(writer.flush.call_count, 3)  # two boundaries + final close
        writer.close.assert_called_once()
        self.assertEqual(trainer.state['global_step'], 2)

    def test_crash_after_validation_events_before_checkpoint_reuses_ledger_without_duplicates(self):
        whole = self.fixture.trainer(self.root / 'whole', validation=True)
        self.fixture.quiet_run(whole)
        split = self.fixture.trainer(self.root / 'split', validation=True)
        split.board = TensorBoardObserver(split.run_dir)
        save = split.checkpoint
        def interrupt():
            if split.state['global_step'] == 2 and not split.state['pending_validation']:
                raise OSError('validation checkpoint interrupted')
            save()
        with patch.object(split, 'checkpoint', side_effect=interrupt), self.assertRaises(OSError):
            self.fixture.quiet_run(split)
        checkpoint = load_checkpoint(split.run_dir / 'last.ckpt', split.identity)
        self.assertTrue(checkpoint['progress']['pending_validation'])
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = self.fixture.trainer(split.run_dir, resume=True, validation=True)
        resumed.board = TensorBoardObserver(split.run_dir)
        real_batch = resumed._batch
        def no_repeat(dataset, index, *, validation=False):
            if validation and resumed.state['global_step'] == 2:
                self.fail('committed ledger case should be skipped')
            return real_batch(dataset, index, validation=validation)
        with patch.object(resumed, '_batch', side_effect=no_repeat):
            self.fixture.quiet_run(resumed)
        self.compare_trainers(whole, resumed)
        events = self.events(resumed.board.directory)
        self.assertEqual([e.step for e in events.Scalars('Monitor/HardDice_Mean')], [2, 4, 6])
        self.assertEqual([e.step for e in events.Scalars('Train/Total_Loss')], list(range(1, 8)))

    def test_console_write_failure_does_not_fail_training(self):
        baseline = self.fixture.trainer(self.root / 'baseline'); self.fixture.quiet_run(baseline)
        observed = self.fixture.trainer(self.root / 'observed')
        stream = Terminal()
        observed.console = TrainingConsole(stream=stream)
        with patch.object(stream, 'write', side_effect=BrokenPipeError('pipe closed')), \
                contextlib.redirect_stderr(io.StringIO()) as error:
            self.fixture.quiet_run(observed)
        self.assertIn('Console disabled', error.getvalue())
        self.compare_trainers(baseline, observed)

    def test_writer_failures_do_not_change_checkpoint_or_training_result(self):
        baseline = self.fixture.trainer(self.root / 'baseline'); self.fixture.quiet_run(baseline)
        for failure in ('create', 'add_scalar', 'flush', 'close'):
            trainer = self.fixture.trainer(self.root / failure)
            trainer.board = TensorBoardObserver(trainer.run_dir)
            writer = MagicMock()
            if failure != 'create':
                getattr(writer, failure).side_effect = OSError('disk unavailable')
            with patch('torch.utils.tensorboard.SummaryWriter', return_value=writer,
                       side_effect=OSError('import/disk') if failure == 'create' else None), \
                    contextlib.redirect_stderr(io.StringIO()) as error:
                self.fixture.quiet_run(trainer)
            self.assertIn('Training continues', error.getvalue())
            self.assertTrue(trainer.board.failed)
            self.compare_trainers(baseline, trainer)

    def test_training_error_closes_observers_without_committing_failed_step(self):
        trainer = self.fixture.trainer(self.root / 'run')
        trainer.board = TensorBoardObserver(trainer.run_dir)
        writer = MagicMock()
        with patch('torch.utils.tensorboard.SummaryWriter', return_value=writer), \
                patch.object(trainer.optimizer, 'step', side_effect=RuntimeError('bad step')):
            with self.assertRaises(RuntimeError):
                self.fixture.quiet_run(trainer)
        writer.close.assert_called_once()
        self.assertFalse((trainer.run_dir / 'last.ckpt').exists())

    def test_startup_interrupt_closes_writer_and_restores_rng(self):
        trainer = self.fixture.trainer(self.root / 'run')
        from organ_relation.training.state import capture_rng
        before = capture_rng(trainer.device)
        writer = MagicMock()
        def interrupt(*args):
            trainer.board.writer = writer
            training_fixtures.random.random()
            training_fixtures.np.random.random()
            training_fixtures.torch.rand(1)
            raise KeyboardInterrupt()
        with patch.object(trainer.board, 'start', side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.fixture.quiet_run(trainer)
        writer.close.assert_called_once()
        self.fixture.assert_nested_equal(before, capture_rng(trainer.device))
        self.assertFalse((trainer.run_dir / 'last.ckpt').exists())

    def test_tensorboard_cli_imports_and_exposes_standard_event_loader(self):
        result = subprocess.run([sys.executable, '-B', '-m', 'tensorboard.main', '--help'],
                                capture_output=True, text=True, errors='replace', timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--logdir', result.stdout)
        self.assertIn('--load_fast', result.stdout)


if __name__ == '__main__':
    unittest.main()

"""Real TensorBoard event files and observer-neutral CPU training acceptance."""
import contextlib
import io
import json
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import test_training as training_fixtures
from organ_relation.training.console import TrainingConsole, columns, duration
from organ_relation.training.state import load_checkpoint
from organ_relation.training.tensorboard import TensorBoardObserver, ORGAN_NAMES, scalar_values


class Terminal(io.StringIO):
    def __init__(self, tty=False):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


def train_row(step=1):
    return dict(phase='train', global_step=step, epoch=step, total_epochs=100, total_steps=100,
        case_id='amos_0109', total_loss=3.284, lr=3e-4, shape=[10, 11, 12],
        coarse={'ce': 2., 'dice_loss': .8}, final={'ce': 1.5, 'dice_loss': .7},
        soft_dice_per_organ=[.3] * 15, step_seconds=60.,
        memory={'peak_allocated_bytes': 8 * 2**30, 'peak_reserved_bytes': 9 * 2**30},
        progress={'eta_seconds': 3600., 'rolling_seconds': 60.},
        diagnostics={'encoder': {'gradient_norm': .25, 'finite': True, 'update_norm': .01}})


def validation_row(step=1, phase='train_monitor'):
    return dict(phase=phase, global_step=step, epoch=step,
        metrics={'mean_case_dice': .7, 'organs': [dict(label=i, mean_dice=.7 if i == 1 else None)
                                                for i in range(1, 16)]})


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.fixture = training_fixtures.TrainingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def events(self, directory):
        return EventAccumulator(str(directory), size_guidance={'scalars': 0}).Reload()

    def start_console(self, stream, *, resume=False, cases=1):
        console = TrainingConsole(stream=stream)
        identity = dict(model={'backbone': {'channels': [8, 16, 32]}},
                        preprocessing={'candidate': 'B', 'spacing_candidates': {'B': [2, 2, 3]}},
                        optimizer={'name': 'AdamW', 'lr': 3e-4},
                        environment={'gpu_name': 'RX 7900 XTX'}, runtime={'dtype': 'float32'})
        options = training_fixtures.config()['training']
        console.start(identity, options, ['amos_0109'] * cases, 100, 100,
                      dict(global_step=25 if resume else 0, epoch=25 if resume else 0),
                      resume='run/last.ckpt' if resume else None, next_case='amos_0109')
        return console

    def test_two_columns_have_fixed_colons_and_values(self):
        text = columns([(('Model', '3-stage [8,16,32]'), ('Device', 'RX 7900 XTX')),
                        (('LR', '3e-4'), ('Batch Size', 1))])
        for line in text.splitlines()[1:-1]:
            self.assertEqual(len(line), 97)
            self.assertEqual([line[15], line[65]], [':', ':'])
            self.assertNotEqual(line[17], ' ')
            self.assertNotEqual(line[67], ' ')

    def test_tqdm_tty_and_redirected_output_flags(self):
        for tty in (False, True):
            stream = Terminal(tty)
            with patch('tqdm.tqdm', return_value=MagicMock()) as factory:
                console = self.start_console(stream)
                kwargs = factory.call_args.kwargs
                self.assertIs(kwargs['file'], stream)
                self.assertEqual(kwargs['disable'], not tty)
                self.assertFalse(kwargs['leave'])
                self.assertTrue(kwargs['dynamic_ncols'])
                console.close()

    def test_resume_summary_and_no_raw_debug_dicts(self):
        stream = Terminal()
        console = self.start_console(stream, resume=True)
        console.train_step(train_row(26))
        console.validation_start(1, 'train_monitor')
        console.validation_case('amos_0109', {'eta_seconds': 0})
        console.validation_end()
        console.validation_summary(validation_row(26))
        console.close()
        text = stream.getvalue()
        for value in ('[RESUME]', '25 / 100', 'Next Case  : amos_0109', 'Monitor Dice', '0.7000'):
            self.assertIn(value, text)
        for value in ('Val Dice', 'gradient_norm', 'pending_validation', 'eta_seconds', "{'", '\r'):
            self.assertNotIn(value, text)

    def test_single_case_does_not_print_summary_each_epoch(self):
        stream = Terminal(); console = self.start_console(stream)
        for step in range(1, 10):
            console.train_step(train_row(step)); console.epoch_end()
        self.assertNotIn('Last Loss', stream.getvalue())
        console.finish(); console.close()
        self.assertEqual(stream.getvalue().count('Last Loss'), 1)

    def test_multicase_epoch_and_val_bars_and_summary(self):
        stream = Terminal()
        console = self.start_console(stream, cases=3)
        with patch('tqdm.tqdm', return_value=MagicMock()) as factory:
            factory.write.side_effect = lambda text, file: print(text, file=file)
            console.begin_step(1, 0)
            self.assertEqual(factory.call_args.kwargs['total'], 3)
            console.train_step(train_row(3))
            console.epoch_end()
            self.assertIn('Last Loss', stream.getvalue())
            console.validation_start(40, 'internal_dev')
            self.assertEqual(factory.call_args.kwargs['desc'], '[Val]')
            console.validation_summary(validation_row(3, 'internal_dev'))
            console.close()
        self.assertIn('Val Dice', stream.getvalue())

    def test_time_format_only_presents_existing_eta(self):
        for value, expected in [(None, 'warming up'), (0, '0s'), (1902, '31m 42s'),
                                (5160, '1h 26m'), (165600, '1d 22h')]:
            self.assertEqual(duration(value), expected)

    def test_real_tensorboard_scalar_mapping_and_stable_organ_names(self):
        observer = TensorBoardObserver(self.root)
        observer.start(0, self.root / 'unused')
        observer.record(train_row())
        observer.record(validation_row())
        observer.record(validation_row(2, 'internal_dev'))
        observer.close()
        events = self.events(observer.directory)
        expected = scalar_values(train_row())
        for tag, value in expected.items():
            self.assertAlmostEqual(events.Scalars(tag)[0].value, value, places=6)
        self.assertEqual(events.Scalars('GradNorm/Encoder')[0].value, .25)
        self.assertAlmostEqual(events.Scalars('Monitor/HardDice_Mean')[0].value, .7)
        self.assertEqual(len(events.Scalars('MonitorDice/Spleen')), 1)
        self.assertNotIn('MonitorDice/RightKidney', events.Tags()['scalars'])
        self.assertEqual(len(ORGAN_NAMES), 15)
        self.assertEqual(ORGAN_NAMES[10:12], ('RightAdrenal', 'LeftAdrenal'))
        self.assertEqual(ORGAN_NAMES[14], 'ProstateOrUterus')
        for kind in ('images', 'histograms', 'tensors'):
            self.assertFalse(events.Tags()[kind])

    def test_train_rows_do_not_emit_per_organ_hard_dice(self):
        tags = scalar_values(train_row())
        self.assertFalse(any('Monitor' in t or 'Val' in t for t in tags))
        row = train_row(); row['memory'] = dict(peak_allocated_bytes=None, peak_reserved_bytes=None)
        self.assertFalse(any('GPU_Peak' in t for t in scalar_values(row)))

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

    def test_overfit_config_only_changes_run_length_and_monitor_save_frequency(self):
        path = training_fixtures.ROOT / 'configs/train_single_case_overfit.json'
        actual = json.loads(path.read_text())
        expected = training_fixtures.config()
        expected['purpose'] = actual['purpose']
        expected['training'].update(max_steps=100, checkpoint_every=25, validation_every=25, diagnostics_every=25)
        self.assertEqual(actual, expected)

    def test_tensorboard_cli_imports_and_exposes_standard_event_loader(self):
        result = subprocess.run([sys.executable, '-B', '-m', 'tensorboard.main', '--help'],
                                capture_output=True, text=True, errors='replace', timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--logdir', result.stdout)
        self.assertIn('--load_fast', result.stdout)


if __name__ == '__main__':
    unittest.main()

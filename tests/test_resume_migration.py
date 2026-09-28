"""Audited old engine -> current engine; synthetic CPU data/provenance only."""
import copy
import hashlib
import importlib.util
import io
import subprocess
import types
import unittest
from unittest.mock import patch

import torch
import test_training as fixtures
from test_monai_reference import Cases
from test_training_v2 import config
from organ_relation.models.monai_relation import MonaiRelationUNet
from organ_relation.training.monai_relation import MonaiRelationLoss
from organ_relation.training.engine import Trainer
from organ_relation.training.state import capture_rng, digest, load_checkpoint, seed_all
from organ_relation.training.resume_migration import (
    SOURCE_COMMIT, SOURCE_HASHES, check_migration_identity, MIGRATION_TYPE)


ROOT = fixtures.ROOT
TEST_CURRENT_COMMIT = 'f' * 40  # Explicit synthetic provenance, never a real run artifact.


class ResumeMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def git(*args):
            return subprocess.check_output(['git', '-C', str(ROOT), *args])
        paths = git('ls-tree', '-r', '--name-only', SOURCE_COMMIT).decode().splitlines()
        cls.old_hashes = {p: hashlib.sha256(git('show', f'{SOURCE_COMMIT}:{p}')).hexdigest()
                          for p in paths if p.endswith('.py') and (p.startswith('src/') or
                              (p.startswith('scripts/') and p.count('/') == 1))}
        # Simulate a clean Linux checkout (LF); the real runtime hashes raw bytes.
        files = [*ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/*.py')]
        cls.new_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(
            p.read_text(encoding='utf-8').encode()).hexdigest() for p in files}
        source = git('show', f'{SOURCE_COMMIT}:src/organ_relation/training/engine.py').decode()
        module = types.ModuleType('organ_relation.training._legacy_engine_test')
        module.__package__ = 'organ_relation.training'
        exec(compile(source, '<audited b4e37bf engine>', 'exec'), module.__dict__)
        cls.LegacyTrainer = module.Trainer

    def setUp(self):
        self.f = fixtures.TrainingTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.root = self.f.root
        self.assertEqual(digest(self.old_hashes), SOURCE_HASHES)

    def identity(self, old=False, count=160):
        cfg = config(True)
        training = copy.deepcopy(cfg['training'])
        if old:
            training.pop('checkpoint_every_steps'); training.pop('console_every_steps')
        result = {k: copy.deepcopy(cfg[k]) for k in ('model', 'relation', 'relation_enabled', 'loss',
                  'coarse_supervision', 'optimizer', 'runtime', 'mode', 'protocol')}
        result.update(training=training, preprocessing={'candidate': 'A', 'spacing': [1.5,1.5,3],
                'input_processing': cfg['input_processing']},
            data={'manifest_hash': 'synthetic', 'split_hash': 'synthetic',
                  'actual_train': [f't{i}' for i in range(count)], 'actual_validation': [f'v{i}' for i in range(40)]},
            frozen_split={'split_hash': 'synthetic', 'file_sha256': 'synthetic'},
            randomness={'model_seed': training['seed'], 'sampler_seed': training['seed'], 'loader_seed': training['seed']+1},
            early_stopping=copy.deepcopy(training['early_stopping']), lr_policy='fixed_no_scheduler',
            environment={'platform': 'synthetic CPU', 'dtype': 'float32'},
            provenance={'git': {'commit': SOURCE_COMMIT if old else TEST_CURRENT_COMMIT, 'dirty': False},
                        'source_hashes': self.old_hashes if old else self.new_hashes})
        return result

    def trainer(self, name, *, old=False, resume=False, migrate=False, count=160):
        identity = self.identity(old, count); cfg = config(True)
        seed_all(identity['training']['seed'])
        model = MonaiRelationUNet(cfg['model'], cfg['input_processing'], relation_enabled=True, relation_config=cfg['relation'])
        optimizer = torch.optim.AdamW(model.parameters(), **{k: v for k, v in cfg['optimizer'].items() if k != 'name'})
        class TrainCases(Cases):
            def __len__(self): return count
            def __getitem__(self, index): return super().__getitem__(index % 2)
        class DevCases(TrainCases):
            def __len__(self): return 40
        path = self.root/name
        options = dict(device='cpu', options=identity['training'], identity=identity, run_dir=path,
            resume=path/'last.ckpt' if resume else None, validation_dataset=DevCases(),
            validation_case_ids=[f'v{i}' for i in range(40)], tensorboard=True)
        if not old:
            options['engineering_resume_migration'] = migrate
        cls = self.LegacyTrainer if old else Trainer
        return cls(model, MonaiRelationLoss(cfg['loss'], **cfg['coarse_supervision']), optimizer,
                   TrainCases(), [f't{i}' for i in range(count)], **options)

    def checkpoint(self, trainer):
        return load_checkpoint(trainer.run_dir/'last.ckpt', trainer.identity)

    def compare_states(self, a, b):
        for key in ('model', 'optimizer', 'rng', 'sampler_generator', 'loader_generator',
                    'progress', 'development_state', 'early_stopping'):
            self.f.assert_nested_equal(a[key], b[key])

    def test_old_engine_migration_exact_state_next_case_cadence_and_strict_next_resume(self):
        whole = self.trainer('whole', old=True); self.f.quiet_run(whole, stop_after=12)
        old = self.trainer('migrate', old=True); self.f.quiet_run(old, stop_after=7)
        parent = self.checkpoint(old); original_run = (old.run_dir/'run.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.trainer('migrate', resume=True)
        new = self.trainer('migrate', resume=True, migrate=True)
        self.f.assert_nested_equal(parent['model'], new.model.state_dict())
        self.f.assert_nested_equal(parent['optimizer'], new.optimizer.state_dict())
        self.f.assert_nested_equal(parent['rng'], capture_rng(new.device))
        self.f.assert_nested_equal(parent['progress'], new.state)
        self.f.assert_nested_equal(parent['sampler_generator'], new.sampler_generator.get_state())
        self.f.assert_nested_equal(parent['loader_generator'], new.loader_generator.get_state())
        self.f.assert_nested_equal(parent['early_stopping'], new.early_state)
        self.f.assert_nested_equal(parent['development_state'], new.development_state)
        saved = []; original = new.checkpoint
        def record(**kwargs):
            original(**kwargs); saved.append(new.state['global_step'])
        with patch.object(new, 'checkpoint', side_effect=record):
            self.f.quiet_run(new, stop_after=4)
        self.assertEqual(saved, [7,10,11])  # Initial migration is durable before step 8.
        ck = self.checkpoint(new); audit = ck['engineering_resume_migration']
        self.assertEqual(audit['migration_type'], MIGRATION_TYPE)
        self.assertEqual(audit['original_checkpoint_commit'], SOURCE_COMMIT)
        self.assertEqual(audit['current_code_commit'], TEST_CURRENT_COMMIT)
        self.assertEqual((audit['global_step'],audit['epoch']), (7,0))
        self.assertEqual(ck['identity'], new.identity)
        self.assertEqual(ck['origin_identity'], old.identity)
        self.assertEqual((old.run_dir/'run.json').read_bytes(), original_run)
        self.assertFalse((old.run_dir/'best-dev.ckpt').exists())
        with self.assertRaisesRegex(ValueError, 'already migrated'):
            self.trainer('migrate', resume=True, migrate=True)
        strict = self.trainer('migrate', resume=True); self.f.quiet_run(strict, stop_after=1)
        self.compare_states(self.checkpoint(whole), self.checkpoint(strict))
        for x,y in zip(self.f.rows(whole.run_dir), self.f.rows(strict.run_dir)):
            for key in ('case_id','epoch','global_step','total_loss','final','coarse','relation_scale'):
                self.assertEqual(x[key],y[key])
        self.assertEqual(self.f.rows(strict.run_dir)[7]['case_id'], old.case_ids[parent['progress']['order'][7]])

    def test_each_scientific_identity_leaf_and_unknown_fields_rejected(self):
        old, new = self.identity(True), self.identity()
        check_migration_identity(old,new)
        def leaves(value, path=()):
            if isinstance(value,dict) and value:
                for k,v in value.items(): yield from leaves(v,path+(k,))
            elif isinstance(value,list) and value:
                for i,v in enumerate(value): yield from leaves(v,path+(i,))
            else: yield path,value
        for path,value in leaves(new):
            if path[0]=='provenance' or path in (('training','checkpoint_every_steps'),('training','console_every_steps')):
                continue
            with self.subTest(path=path):
                bad=copy.deepcopy(new); target=bad
                for key in path[:-1]: target=target[key]
                target[path[-1]]=not value if isinstance(value,bool) else value+1 if isinstance(value,(int,float)) else 'changed'
                with self.assertRaises(ValueError): check_migration_identity(old,bad)
        with self.assertRaises(ValueError): check_migration_identity(old,dict(new,unrecognized_science='changed'))

    def test_epoch_boundary_validation_history_and_best_checkpoint_preserved(self):
        # A two-case synthetic epoch makes full 40-case validation affordable on CPU.
        whole=self.trainer('history_whole',old=True,count=2);self.f.quiet_run(whole,stop_after=12)
        old=self.trainer('history_migrate',old=True,count=2);self.f.quiet_run(old,stop_after=10)
        parent=self.checkpoint(old);best_path=old.run_dir/'best-dev.ckpt';best_bytes=best_path.read_bytes()
        self.assertEqual(parent['progress']['epoch'],5)
        self.assertEqual(len(parent['development_state']['validation_history']),1)
        new=self.trainer('history_migrate',resume=True,migrate=True,count=2)
        batch=new._batch
        def after_transaction(*args,**kwargs):
            if new.state['global_step']==10:
                self.assertEqual(best_path.read_bytes(),best_bytes)
                ck=self.checkpoint(new)
                self.assertEqual(ck['engineering_resume_migration']['epoch'],5)
                self.compare_states(parent,ck)
            return batch(*args,**kwargs)
        with patch.object(new,'_batch',side_effect=after_transaction):
            self.f.quiet_run(new,stop_after=2)
        self.compare_states(self.checkpoint(whole),self.checkpoint(new))
        self.assertEqual(best_path.read_bytes(),best_bytes)

    def test_provenance_whitelist_dirty_wrong_source_and_cadence_rejected(self):
        old,new=self.identity(True),self.identity()
        for version in ('old','new'):
            a,b=copy.deepcopy(old),copy.deepcopy(new)
            (a if version=='old' else b)['provenance']['git']['dirty']=True
            with self.assertRaises(ValueError):check_migration_identity(a,b)
        bad=copy.deepcopy(old);bad['provenance']['git']['commit']='a'*40
        with self.assertRaises(ValueError):check_migration_identity(bad,new)
        bad=copy.deepcopy(old);bad['provenance']['source_hashes']['src/organ_relation/training/development.py']='changed'
        with self.assertRaises(ValueError):check_migration_identity(bad,new)
        bad=copy.deepcopy(new);bad['provenance']['source_hashes']['src/organ_relation/models/monai_relation.py']='changed'
        with self.assertRaises(ValueError):check_migration_identity(old,bad)
        bad=copy.deepcopy(new);bad['provenance']['source_hashes']['src/organ_relation/training/engine.py']='future unaudited edit'
        with self.assertRaises(ValueError):check_migration_identity(old,bad)
        bad=copy.deepcopy(new);bad['training']['checkpoint_every_steps']=10
        with self.assertRaises(ValueError):check_migration_identity(old,bad)

    def test_atomic_migration_failure_keeps_old_checkpoint_and_run_identity(self):
        old=self.trainer('atomic',old=True);self.f.quiet_run(old,stop_after=2)
        path=old.run_dir/'last.ckpt'; original=path.read_bytes(); run=(old.run_dir/'run.json').read_bytes()
        new=self.trainer('atomic',resume=True,migrate=True)
        with patch('organ_relation.training.state.os.replace',side_effect=OSError('synthetic interrupted atomic write')):
            with self.assertRaises(OSError):self.f.quiet_run(new,stop_after=1)
        self.assertEqual(path.read_bytes(),original)
        self.assertEqual((old.run_dir/'run.json').read_bytes(),run)
        self.checkpoint(old)  # Original strict identity still loads.

    def test_cli_requires_explicit_resume_and_forbids_extension_preflight(self):
        spec=importlib.util.spec_from_file_location('migration_cli_test',ROOT/'scripts/train.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        common=['--config','missing.json','--data-root','missing-data']
        self.assertFalse(module.parser().parse_args(common).allow_engineering_resume_migration)
        for extra in ([],['--resume','last.ckpt','--extend-to','999'],
                      ['--resume','last.ckpt','--preflight-backward']):
            args=module.parser().parse_args(common+['--allow-engineering-resume-migration']+extra)
            with self.assertRaisesRegex(ValueError,'engineering migration requires'):
                module.execute(args)


if __name__=='__main__':unittest.main()

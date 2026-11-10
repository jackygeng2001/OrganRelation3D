"""Relation probe: exact knock-outs, restored model state, read-only CLI with resume and summary."""
import contextlib
import copy
import gzip
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from organ_relation.data.brats import index_cases
from organ_relation.losses import region_targets
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.provenance import code_hashes, sha256
from organ_relation.training.brain_tumor import region_dice
from organ_relation.training.state import digest, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('brats_relation_probe', ROOT / 'scripts/brats_relation_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
CFG = json.loads((ROOT / 'configs/development/implementation_check.json').read_text())


def build(variant=None, seed=0):
    torch.manual_seed(seed)
    model = BrainTumorSegmentor(BrainTumorConfig(**CFG['model'], **({} if variant is None else dict(variant=variant))))
    if hasattr(model.decoder, 'eta'):  # 普通解码器的消融变体没有区域回写
        with torch.no_grad():
            model.decoder.eta.copy_(torch.tensor([.5, -.4, .3]))  # 让解码器回写明显非零，敲除才有可见效果
    return model.eval()


def case_inputs(seed=1, shape=(16, 16, 16)):
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn((1, 4, *shape), generator=generator)
    label = torch.randint(0, 4, (1, *shape), generator=generator)
    return image, label


class KnockoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_baseline_and_knockouts_equal_direct_model_edits(self):
        model = build()
        image, label = case_inputs()
        tumor = region_targets(label)[0][0]
        with torch.no_grad():
            features = model.encoder(image)
            reference = model(image).final_logits
        before = {k: v.clone() for k, v in model.state_dict().items()}
        record = {}
        self.assertTrue(torch.equal(probe.probe_tail(model, features), reference))
        self.assertTrue(torch.equal(probe.probe_tail(model, features, record=record, tumor=tumor), reference))

        def edited(change):
            other = copy.deepcopy(model)
            with torch.no_grad():
                change(other)
                return other(image).final_logits
        self.assertTrue(torch.equal(probe.probe_tail(model, features, gamma=0.),
                                    edited(lambda m: m.fusion.relation_scale.fill_(0))))
        self.assertTrue(torch.equal(probe.probe_tail(model, features, eta=0.), edited(lambda m: m.decoder.eta.zero_())))
        self.assertTrue(torch.equal(probe.probe_tail(model, features, gamma=0., eta=0.),
                                    edited(lambda m: (m.fusion.relation_scale.fill_(0), m.decoder.eta.zero_()))))

        silent = copy.deepcopy(model)
        original = silent.relation._edge_weights
        with torch.no_grad(), patch.object(silent.relation, '_edge_weights', side_effect=lambda *a: torch.zeros_like(original(*a))):
            no_edges = silent(image).final_logits
        self.assertTrue(torch.equal(probe.probe_tail(model, features, relation='edges_zero'), no_edges))

        plain = BrainTumorSegmentor(BrainTumorConfig(**CFG['model'], variant='no_relation')).eval()
        plain.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith('relation.')}, strict=True)
        with torch.no_grad():
            self.assertTrue(torch.equal(probe.probe_tail(model, features, relation='skipped'), plain(image).final_logits))
        self.assertFalse(torch.equal(probe.probe_tail(model, features, eta=0.), reference))

        # 模型状态完全复原：参数不变，解码器上不残留替身方法
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertFalse(any('forward_projected' in q.__dict__ for q in model.decoder.queries))

        with torch.no_grad():
            F = features[-1]
            nodes = model.space_to_node(F, model.coarse_head(F).probabilities)
            state = model.relation.forward_with_last_edges(nodes.z0, nodes.centroid, nodes.size, nodes.confidence)
            writeback = model.fusion.phi(model.node_to_space(F, state.zK))
        expected = torch.linalg.vector_norm((model.fusion.relation_scale * writeback).double()) / torch.linalg.vector_norm(F.double())
        self.assertTrue(math.isclose(record['deep']['scaled_ratio'], expected.item(), rel_tol=1e-9))
        self.assertEqual(record['deep']['gamma'], model.fusion.relation_scale.item())
        self.assertEqual(len(record['relation']['rounds']), model.relation.rounds)
        self.assertEqual(record['relation']['last_alpha'], state.last_alpha[0].tolist())
        for round_ in record['relation']['rounds']:
            self.assertEqual([round_['alpha'][i][i] for i in range(3)], [0., 0., 0.])
            self.assertGreaterEqual(round_['message_share_of_update'], 0.)
        self.assertEqual([d['eta'] for d in record['decoder']], model.decoder.eta.tolist())
        json.dumps(record, allow_nan=False)

    def test_nested_chain_records_only_its_four_edges(self):
        torch.manual_seed(0)
        model = BrainTumorSegmentor(BrainTumorConfig(**CFG['model'], graph='nested_chain')).eval()
        image, label = case_inputs(seed=5)
        with torch.no_grad():
            features = model.encoder(image)
            reference = model(image).final_logits
        record = {}
        self.assertTrue(torch.equal(probe.probe_tail(model, features, record=record, tumor=region_targets(label)[0][0]), reference))
        self.assertEqual(record['relation']['edges'], [[0, 1], [1, 0], [1, 2], [2, 1]])
        silent = copy.deepcopy(model)
        original = silent.relation._edge_weights
        with torch.no_grad(), patch.object(silent.relation, '_edge_weights', side_effect=lambda *a: torch.zeros_like(original(*a))):
            self.assertTrue(torch.equal(probe.probe_tail(model, features, relation='edges_zero'), silent(image).final_logits))
        row = {'E1': dict(probe.probe_case(model, image, label, .5), ledger_dice=None), 'case_id': 'x'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'chain.jsonl'
            path.write_text(json.dumps(dict(kind='header', schema=probe.SCHEMA, runs={'E1': {}})) + '\n' + json.dumps(row) + '\n')
            text = probe.summarize([path])
        alphas = [row['E1']['relation']['rounds'][0]['alpha'][i][j] for i, j in record['relation']['edges']]
        self.assertIn(f'round 1 edge weights: median {np.median(alphas):.4f}', text)

    def test_probe_case_variants_and_no_relation_refusals(self):
        image, label = case_inputs(seed=3)
        for variant, expected in ((None, {'edges_zero', 'relation_skipped'}), ('no_relation', set())):
            model = build(variant)
            result = probe.probe_case(model, image, label, .5)
            knockouts = {k for k in result if k not in ('deep', 'relation', 'decoder', 'base')}
            self.assertEqual(knockouts, {'gamma_zero', 'eta_zero', 'gamma_eta_zero'} | expected)
            with torch.no_grad():
                hard = region_dice(model(image).final_logits, label, .5)
            self.assertEqual(result['base']['dice'], [r['dice'] for r in hard['organs']])
            self.assertEqual((result['base']['flipped_voxels'], result['base']['logit_relative_change']), (0, 0.))
            json.dumps(result, allow_nan=False)
        self.assertEqual(result['relation']['rounds'], [])
        self.assertEqual(result['relation']['z_change_per_node'], [0., 0., 0.])
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'no relation rounds'):
            probe.probe_tail(model, model.encoder(image), relation='edges_zero')
        with self.assertRaisesRegex(ValueError, 'full method or the no_relation'):
            probe.probe_case(build('drf_backbone'), image, label, .5)



class CommandLineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'data'
        self.cases = [f'BraTS-GLI-0000{i}-000' for i in range(3)]
        for n, case in enumerate(self.cases):
            folder = self.source / case
            folder.mkdir(parents=True)
            shape = (9, 7, 5)
            for role in ('t1n', 't1c', 't2w', 't2f', 'seg'):
                grid = np.arange(np.prod(shape)).reshape(shape)
                array = ((grid + n) % 4 if role == 'seg' else grid + 1 + n).astype(np.float32)
                (folder / f'{case}-{role}.nii.gz').write_bytes(gzip.compress(nib.Nifti1Image(array, np.eye(4)).to_bytes()))
        self.manifest = index_cases(self.source, cohort='brats2023_gli')

    def make_run(self, name, variant=None, split=None, source_hashes=None, ledger=None):
        split = split or dict(cohort='brats2023_gli', train=self.cases[:1], dev=self.cases[1:])
        model_cfg = dict(CFG['model'], **({} if variant is None else dict(variant=variant)))
        identity = dict(mode='brats_regions', model=model_cfg, loss=CFG['loss'], threshold=.5, preprocessing=CFG['preprocessing'],
                        training=CFG['training'], data=dict(manifest=self.manifest, manifest_hash=digest(self.manifest),
                                                            split=split, split_hash=digest(split)),
                        runtime=dict(convolution_backend='depth_sliced', convolution_depth_chunk=4, dtype='float32', cpu_threads=2),
                        provenance=dict(source_hashes=source_hashes or code_hashes(), git=dict(commit='cpu-fixture', dirty=False)))
        run = self.root / 'runs' / name
        run.mkdir(parents=True)
        (run / 'run.json').write_text(json.dumps(dict(identity=identity, run_id=name)))
        state = dict(schema_version=1, identity=identity, run_id=name, model=build(variant).state_dict(), optimizer={},
                     scheduler=None, progress=dict(global_step=100, epoch=4), rng={}, sampler_generator=None,
                     loader_generator=None, log={})
        save_checkpoint(run / 'best-dev.ckpt', state)
        if ledger:
            folder = run / 'validation' / 'step_00000100'
            folder.mkdir(parents=True)
            (folder / 'marker.json').write_text(json.dumps(dict(completed=True, identity={}, schema_version=1)))
            for case, dice in ledger.items():
                organs = [dict(dice=d) for d in dice]
                (folder / f'{case}.json').write_text(json.dumps(dict(case_id=case, result=dict(organs=organs))))
        return run

    def call(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = probe.main(list(map(str, args)))
        return status, out.getvalue(), err.getvalue()

    def test_read_only_probe_resume_and_summary(self):
        full = self.make_run('full', ledger={self.cases[1]: [.5, None, .25]})
        plain = self.make_run('plain', variant='no_relation')
        output = self.root / 'probe' / 'fold.jsonl'
        tracked = [p for parent in (self.root / 'runs', self.source) for p in parent.rglob('*') if p.is_file()]
        before = {str(p): sha256(p) for p in tracked}
        command = ('probe', '--run', f'E1={full}', '--run', f'A3={plain}', '--output', output, '--threads', 2)
        with patch.object(torch.Tensor, 'backward', side_effect=AssertionError('backward forbidden')):
            status, out, err = self.call(*command)
        self.assertEqual(status, 0, err)
        self.assertEqual(before, {str(p): sha256(p) for p in tracked})
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        header, cases = rows[0], rows[1:]
        self.assertEqual((header['schema'], header['cases']), (probe.SCHEMA, self.cases[1:]))
        self.assertEqual(header['runs']['E1']['checkpoint_sha256'], sha256(full / 'best-dev.ckpt'))
        self.assertTrue(header['runs']['E1']['source_matches_training'])
        self.assertEqual([r['case_id'] for r in cases], self.cases[1:])
        self.assertIn('edges_zero', cases[0]['E1'])
        self.assertNotIn('edges_zero', cases[0]['A3'])
        self.assertEqual(cases[0]['E1']['ledger_dice'], [.5, None, .25])
        self.assertIsNone(cases[1]['E1']['ledger_dice'])
        self.assertIsNone(cases[0]['A3']['ledger_dice'])

        # 续跑：表头一致时跳过已完成病例，不再推理
        with patch.object(probe, 'probe_case', side_effect=AssertionError('finished cases must be skipped')):
            status, _, err = self.call(*command)
        self.assertEqual(status, 0, err)
        self.assertEqual(len(output.read_text().splitlines()), 3)
        status, _, err = self.call('probe', '--run', f'E1={full}', '--output', output)
        self.assertEqual(status, 2)
        self.assertIn('existing output', err)

        status, text, err = self.call('summarize', output)
        self.assertEqual(status, 0, err)
        for expected in ('[E1]', '[A3]', 'knockout edges_zero', 'knockout eta_zero', 'validation ledger', 'round 2 edge'):
            self.assertIn(expected, text)

    def test_refusals_before_inference(self):
        full = self.make_run('full')
        changed = dict(code_hashes(), **{'src/organ_relation/losses.py': 'older'})
        older = self.make_run('older', source_hashes=changed)
        other_split = self.make_run('other', split=dict(cohort='brats2023_gli', train=self.cases[1:], dev=self.cases[:1]))
        plain_decoder = self.make_run('drf', variant='drf_backbone')
        output = self.root / 'probe.jsonl'
        with patch.object(probe, 'probe_case', side_effect=AssertionError('must refuse before inference')):
            for args, message in (((f'E1={older}',), 'package source differs'),
                                  ((f'E1={full}', f'B={other_split}'), 'split_hash differs'),
                                  ((f'D={plain_decoder}',), 'full method or the no_relation'),
                                  ((f'E1={full}', f'E1={full}'), 'unique'),
                                  ((f'kind={full}',), 'reserved')):
                runs = [x for a in args for x in ('--run', a)]
                status, _, err = self.call('probe', *runs, '--output', output)
                self.assertEqual(status, 2, args)
                self.assertIn(message, err)
            status, _, err = self.call('probe', '--run', f'E1={full}', '--output', full / 'inside.jsonl')
            self.assertEqual(status, 2)
            self.assertIn('outside the runs', err)
        self.assertFalse(output.exists())
        status, _, err = self.call('probe', '--run', f'E1={older}', '--output', output, '--allow-source-change', '--cases', 1)
        self.assertEqual(status, 0, err)
        header = json.loads(output.read_text().splitlines()[0])
        self.assertFalse(header['runs']['E1']['source_matches_training'])
        self.assertEqual(header['runs']['E1']['differing_sources'], ['src/organ_relation/losses.py'])
        self.assertEqual(len(header['cases']), 1)


if __name__ == '__main__':
    unittest.main()

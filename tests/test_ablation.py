"""Ablation variants, the nested-chain graph and the experiment configs, CPU only."""
import copy
import json
import unittest

import torch
from torch.nn import functional as F

import support as fixtures
from test_brain_tumor import Cases, config
from organ_relation.losses import RegionJointLoss
from organ_relation.models.brain_tumor import (BrainTumorConfig, BrainTumorSegmentor, DualReceptiveField,
                                             PlainDecoder, VARIANTS)
from organ_relation.models.depth_sliced_conv import DepthSlicedConv3d
from organ_relation.training.engine import Trainer
from organ_relation.training.state import load_checkpoint, seed_all

FULL_CHILDREN = ['encoder', 'coarse_head', 'space_to_node', 'relation', 'node_to_space', 'fusion', 'decoder']
CHILDREN = {None: FULL_CHILDREN,
            'no_relation': ['encoder', 'coarse_head', 'space_to_node', 'node_to_space', 'fusion', 'decoder'],
            'drf_backbone': ['encoder', 'coarse_head', 'decoder'],
            'unet': ['encoder', 'coarse_head', 'decoder']}


def model_config(variant, base=None):
    cfg = dict((base or config())['model'])
    if variant is not None:
        cfg['variant'] = variant
    return cfg


def build(variant, **kwargs):
    return BrainTumorSegmentor(BrainTumorConfig(**model_config(variant)), **kwargs)


def volume(shape=(9, 10, 11), seed=4):
    image = torch.randn(1, 4, *shape, generator=torch.Generator().manual_seed(seed))
    label = (torch.arange(int(torch.tensor(shape).prod())).reshape(1, *shape) % 4).long()
    return image, label


class ConfigTests(unittest.TestCase):
    def test_full_method_identity_is_unchanged_and_variants_round_trip(self):
        cfg = config()['model']
        self.assertNotIn('variant', BrainTumorConfig(**cfg).to_dict())
        self.assertEqual(BrainTumorConfig(**cfg).to_dict(), cfg)
        for variant in VARIANTS[1:]:
            self.assertEqual(BrainTumorConfig(**model_config(variant)).to_dict(), model_config(variant))
        for bad in ('full', 'no_nodes', '', 1):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'variant'):
                BrainTumorConfig(**dict(cfg, variant=bad))

    def test_each_ablation_config_differs_from_the_main_arm_in_one_item(self):
        # 旧全连接图阶段的配置（E1/A3 等），已移到 configs/development/complete_graph/
        main = json.loads((fixtures.ROOT / 'configs/development/complete_graph/main.json').read_text())
        science = lambda c: {k: v for k, v in c.items() if k != 'status' and not k.endswith('_note')}
        expected = {'no_relation': ('model', 'variant', 'no_relation'), 'drf_backbone': ('model', 'variant', 'drf_backbone'),
                    'unet': ('model', 'variant', 'unet'), 'no_hier': ('loss', 'lambda_hier', 0.0)}
        for name, (section, key, value) in expected.items():
            with self.subTest(arm=name):
                arm = json.loads((fixtures.ROOT / f'configs/development/complete_graph/{name}.json').read_text())
                wanted = copy.deepcopy(science(main))
                wanted[section][key] = value
                self.assertEqual(science(arm), wanted)
                self.assertIn('ablation_note', arm)
                BrainTumorConfig(**arm['model'])
                RegionJointLoss(**arm['loss'])


    def test_graph_round_trips_and_is_validated(self):
        cfg = dict(config()['model'], graph='nested_chain')
        self.assertEqual(BrainTumorConfig(**cfg).to_dict(), cfg)
        self.assertEqual(BrainTumorConfig(**dict(cfg, variant='no_relation')).to_dict(), dict(cfg, variant='no_relation'))
        for bad in ('complete', 'chain', '', 1):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'graph'):
                BrainTumorConfig(**dict(cfg, graph=bad))

    def test_cumulative_rows_each_add_one_item_and_full_method_only_changes_the_graph(self):
        load = lambda name: json.loads((fixtures.ROOT / f'configs/experiments/{name}.json').read_text())
        science = lambda c: {k: v for k, v in c.items() if k != 'status' and not k.endswith('_note')}
        rows = {name: load(name) for name in ('r1_drf_backbone', 'r2_nodes', 'r3_relation', 'r4_full', 'r4_k1', 'r4_k3')}
        self.assertEqual(sorted(p.stem for p in (fixtures.ROOT / 'configs/experiments').glob('*.json')), sorted(rows))

        def changed(base, section, key, value):
            result = copy.deepcopy(science(base))
            if value is None:
                del result[section][key]
            else:
                result[section][key] = value
            return result
        steps = (('r1_drf_backbone', 'r2_nodes', ('model', 'variant', 'no_relation')),
                 ('r2_nodes', 'r3_relation', ('model', 'variant', None)),
                 ('r3_relation', 'r4_full', ('loss', 'lambda_hier', 1.0)),
                 ('r4_full', 'r4_k1', ('model', 'rounds', 1)),
                 ('r4_full', 'r4_k3', ('model', 'rounds', 3)))
        for before, after, change in steps:
            with self.subTest(step=f'{before}->{after}'):
                self.assertEqual(science(rows[after]), changed(rows[before], *change))
        self.assertEqual((rows['r1_drf_backbone']['model']['variant'], rows['r1_drf_backbone']['loss']['lambda_hier']),
                         ('drf_backbone', 0.0))
        old = json.loads((fixtures.ROOT / 'configs/development/complete_graph/main.json').read_text())
        # R4 与 E1 只差关系图和轮数上限（150 对 500，二者都在上限前早停，训练过程等价）
        expected = changed(old, 'model', 'graph', 'nested_chain')
        expected['training']['max_epochs'] = 150
        self.assertEqual(science(rows['r4_full']), expected)
        for name, cfg in rows.items():
            with self.subTest(arm=name):
                self.assertEqual(cfg['model']['graph'], 'nested_chain')
                self.assertIn('ablation_note', cfg)
                BrainTumorConfig(**cfg['model'])
                RegionJointLoss(**cfg['loss'])


class ArchitectureTests(unittest.TestCase):
    def test_full_method_modules_unchanged(self):
        model = build(None)
        self.assertEqual([n for n, _ in model.named_children()], FULL_CHILDREN)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8330296)

    def test_nested_chain_parameter_counts(self):
        cfg = dict(config()['model'], graph='nested_chain')
        self.assertEqual(sum(p.numel() for p in BrainTumorSegmentor(BrainTumorConfig(**cfg)).parameters()), 8400001)
        counts = {'r1_drf_backbone': 7780444, 'r2_nodes': 7894255, 'r3_relation': 8486001, 'r4_full': 8486001}
        for name, expected in counts.items():
            with self.subTest(arm=name):
                arm = json.loads((fixtures.ROOT / f'configs/experiments/{name}.json').read_text())
                model = BrainTumorSegmentor(BrainTumorConfig(**arm['model']))
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected)

    def test_nested_chain_model_four_edges_and_all_gradients(self):
        seed_all(6)
        model = BrainTumorSegmentor(BrainTumorConfig(**dict(config()['model'], graph='nested_chain')))
        self.assertEqual([n for n, _ in model.named_children()], FULL_CHILDREN)
        image, label = volume((9, 11, 13))
        out = model.forward_with_diagnostics(image)
        nonzero = {(i, j) for i in range(3) for j in range(3) if out.last_alpha[0, i, j] != 0}
        self.assertEqual(nonzero, {(0, 1), (1, 2), (1, 0), (2, 1)})
        RegionJointLoss(**dict(config()['loss'], lambda_hier=1.0))(*out.output, label).total.backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        for name in ('inward_hidden', 'outward_hidden', 'W_inward', 'W_outward'):
            self.assertGreater(getattr(model.relation, name).weight.grad.abs().sum().item(), 0, name)


    def test_variant_modules_and_encoders(self):
        for variant, children in CHILDREN.items():
            with self.subTest(variant=variant):
                model = build(variant)
                self.assertEqual([n for n, _ in model.named_children()], children)
                drf = [m for m in model.encoder.modules() if isinstance(m, DualReceptiveField)]
                self.assertEqual(len(drf), 0 if variant == 'unet' else 3)
                dilated = [m for m in model.modules() if isinstance(m, torch.nn.Conv3d) and max(m.dilation) > 1]
                self.assertEqual(len(dilated), 0 if variant == 'unet' else 3)
                self.assertEqual(isinstance(model.decoder, PlainDecoder), variant in ('drf_backbone', 'unet'))

    def test_every_variant_trains_image_only_with_all_gradients(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                seed_all(5)
                model = build(variant)
                image, label = volume((9, 11, 13))
                out = model.forward_with_diagnostics(image)
                self.assertEqual([tuple(t.shape[1:]) for t in out.features],
                                 [(32, 9, 11, 13), (64, 5, 6, 7), (128, 3, 3, 4), (256, 2, 2, 2)])
                self.assertEqual(out.output.final_logits.shape, (1, 3, 9, 11, 13))
                self.assertEqual(out.output.coarse_logits.shape, (1, 3, 2, 2, 2))
                plain = variant in ('drf_backbone', 'unet')
                self.assertEqual(out.scale_weights is None, plain)
                RegionJointLoss(**dict(config()['loss'], lambda_hier=1.0))(*out.output, label).total.backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                self.assertGreater(model.coarse_head.parameters().__next__().grad.abs().sum(), 0)
                with self.assertRaises(TypeError):
                    model(image, label=label)

    def test_no_relation_is_the_full_pipeline_with_z0_and_zero_gates(self):
        seed_all(7)
        full = build(None).double()
        ablated = build('no_relation').double()
        missing, unexpected = ablated.load_state_dict(full.state_dict(), strict=False)
        self.assertEqual(missing, [])
        self.assertTrue(unexpected and all(k.startswith('relation.') for k in unexpected))
        image = volume()[0].double()
        out = ablated.forward_with_diagnostics(image)
        torch.testing.assert_close(out.zK, out.nodes.z0)
        self.assertTrue(torch.equal(out.last_alpha, torch.zeros(1, 3, 3, dtype=torch.double)))
        features = full.encoder(image)
        coarse = full.coarse_head(features[-1])
        nodes = full.space_to_node(features[-1], coarse.probabilities)
        zeros = torch.zeros(1, 3, 3, dtype=torch.double)
        fused = full.fusion(features[-1], full.node_to_space(features[-1], nodes.z0))
        expected = full.decoder(fused, features, nodes.z0, nodes, zeros)
        torch.testing.assert_close(out.output.final_logits, expected)
        torch.testing.assert_close(out.output.coarse_logits, coarse.logits)

    def test_plain_decoder_is_resize_concatenate_fuse(self):
        seed_all(8)
        model = build('drf_backbone').double()
        image = volume()[0].double()
        features = model.encoder(image)
        decoded = features[-1]
        for index, level in enumerate((2, 1, 0)):
            skip = features[level]
            decoded = model.decoder.up[index](decoded, size=skip.shape[2:], align_corners=False)
            decoded = model.decoder.fuse[index](torch.cat((decoded, skip), 1))
        torch.testing.assert_close(model(image).final_logits, model.decoder.head(decoded))

    def test_depth_sliced_backend_matches_native_for_plain_variants(self):
        for variant in ('drf_backbone', 'unet', 'no_relation'):
            with self.subTest(variant=variant):
                seed_all(9)
                native = build(variant).double()
                sliced = build(variant, convolution_backend='depth_sliced').double()
                sliced.load_state_dict(native.state_dict())
                self.assertTrue(any(isinstance(m, DepthSlicedConv3d) for m in sliced.decoder.modules()))
                image, label = volume()
                image = image.double()
                outputs = []
                for model in (native, sliced):
                    out = model(image)
                    RegionJointLoss(**config()['loss'])(out.coarse_logits, out.final_logits, label).total.backward()
                    outputs.append((out.final_logits.detach(), torch.cat([p.grad.flatten() for p in model.parameters()])))
                torch.testing.assert_close(outputs[1][0], outputs[0][0])
                torch.testing.assert_close(outputs[1][1], outputs[0][1])

    def test_whole_network_mixed_precision_for_each_variant(self):
        """Bounded by the approved kernel-only bf16 option on the same weights. Measured
        (seed 3) gradient relL2 amp/kernel-only: no_relation .053/.052, drf_backbone
        .056/.054, unet .118/.095 (cosine .9935/.9961); the residual-free unet encoder is
        the most rounding-sensitive on this tiny volume under BOTH options."""
        for variant in VARIANTS[1:]:
            with self.subTest(variant=variant):
                seed_all(3)
                reference = build(variant, convolution_backend='depth_sliced')
                image, label = volume()
                results = []
                for kwargs in ({}, dict(convolution_compute_dtype='bfloat16'), dict(mixed_precision='bfloat16')):
                    m = build(variant, convolution_backend='depth_sliced', **kwargs)
                    m.load_state_dict(reference.state_dict())
                    m.to(memory_format=torch.channels_last_3d)
                    out = m(image.contiguous(memory_format=torch.channels_last_3d))
                    RegionJointLoss(**dict(config()['loss'], lambda_hier=1.0))(
                        out.coarse_logits, out.final_logits, label).total.backward()
                    results.append((out, torch.cat([p.grad.flatten() for p in m.parameters()]), m))
                (expected, wanted, _), (_, kernel_grads, _), (out, grads, model) = results
                self.assertEqual((out.coarse_logits.dtype, out.final_logits.dtype), (torch.float32, torch.float32))
                self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
                rel = ((out.final_logits - expected.final_logits).norm() / expected.final_logits.norm()).item()
                self.assertLessEqual(rel, 2e-2)
                error = lambda g: ((g - wanted).norm() / wanted.norm()).item()
                self.assertLessEqual(error(grads), 1.5 * error(kernel_grads))
                self.assertGreaterEqual(F.cosine_similarity(grads, wanted, 0).item(), 0.99)


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cfg = config()

    def trainer(self, variant, path, *, resume=False, identity_variant='same', steps=3):
        seed_all(171)
        model = build(variant)
        options = dict(self.cfg['training'], max_steps=steps, validation_every=1, diagnostics_every=1,
                       checkpoint_every=1, validation_role='train_monitor')
        recorded = model_config(variant if identity_variant == 'same' else identity_variant)
        identity = dict(mode='brats_regions', model=recorded, loss=self.cfg['loss'], threshold=.5, training=options,
                        preprocessing=self.cfg['preprocessing'], provenance='test', data={'manifest_hash': 'synthetic'})
        return Trainer(model, RegionJointLoss(**self.cfg['loss']),
                       torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False, fused=False),
                       Cases(), ['one', 'two'], device='cpu', options=options, identity=identity, run_dir=path,
                       resume=path / 'last.ckpt' if resume else None, validation_dataset=Cases(),
                       validation_case_ids=['one', 'two'], tensorboard=False)

    def test_each_variant_trains_validates_and_logs_both_branches(self):
        for variant in VARIANTS[1:]:
            with self.subTest(variant=variant):
                trainer = self.trainer(variant, self.fixture.root / variant)
                self.fixture.quiet_run(trainer)
                rows = self.fixture.rows(trainer.run_dir)
                train = [r for r in rows if r['phase'] == 'train']
                self.assertEqual(len(train), 3)
                self.assertTrue(all(torch.isfinite(torch.tensor(r['total_loss'])) for r in train))
                self.assertTrue(all('coarse' in r and 'final' in r for r in train))
                groups = set(train[0]['diagnostics'])
                self.assertEqual(groups, set(CHILDREN[variant]) - {'space_to_node'})
                self.assertEqual(len([r for r in rows if r['phase'] == 'train_monitor']), 3)
                self.assertEqual(trainer.identity['model']['variant'], variant)

    def test_identity_must_name_the_built_variant(self):
        for built, recorded in (('unet', 'drf_backbone'), ('no_relation', None), (None, 'no_relation')):
            with self.subTest(built=built, recorded=recorded), self.assertRaisesRegex(ValueError, 'strict identity'):
                self.trainer(built, self.fixture.root / f'bad_{built}_{recorded}', identity_variant=recorded)

    def test_variant_resume_reproduces_uninterrupted_run(self):
        whole = self.trainer('unet', self.fixture.root / 'whole'); self.fixture.quiet_run(whole)
        part = self.trainer('unet', self.fixture.root / 'part'); self.fixture.quiet_run(part, stop_after=1)
        resumed = self.trainer('unet', part.run_dir, resume=True); self.fixture.quiet_run(resumed)
        a = load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)
        b = load_checkpoint(part.run_dir / 'last.ckpt', resumed.identity)
        for key in ('model', 'optimizer', 'rng', 'progress'):
            self.fixture.assert_nested_equal(a[key], b[key])


if __name__ == '__main__':
    unittest.main()

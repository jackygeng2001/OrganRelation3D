"""Opt-in relation scalar: exact legacy path, formula, observers and recovery."""
import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import test_training as fixtures
from test_monai_reference import Cases
import test_monai_relation as relation_tests
from organ_relation.data.full_scan import FullScanSample
from organ_relation.models.residual_fusion import ResidualFusion
from organ_relation.models.monai_reference import MonaiReferenceUNet
from organ_relation.models.monai_relation import MonaiRelationUNet
from organ_relation.training.monai_relation import MonaiRelationLoss
from organ_relation.training.monai_reference import preflight_backward
from organ_relation.training.state import load_checkpoint, seed_all, capture_rng
from organ_relation.training.tensorboard import scalar_values

ROOT = fixtures.ROOT


def config(gated=True):
    name = 'train_monai_relation_gated_A_160_40.json' if gated else 'train_monai_relation_A_160_40.json'
    return json.loads((ROOT / 'configs' / name).read_text())


class RelationScaleTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.TrainingTests(); self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.root = self.f.root

    def model(self, gated=True, enabled=True):
        cfg = config(gated)
        return MonaiRelationUNet(cfg['model'], cfg['input_processing'],
            relation_enabled=enabled, relation_config=cfg['relation'])

    def test_config_only_two_explicit_scale_fields(self):
        old, new = config(False), config()
        self.assertTrue(new['relation'].pop('learnable_relation_scale'))
        self.assertEqual(new['relation'].pop('relation_scale_init'), .1)
        # Current formal horizon/observation policy changed by explicit request;
        # retain the old non-gated config as a historical protocol fixture.
        baseline = json.loads((ROOT/'configs/train_monai_reference_A_160_40.json').read_text())
        self.assertEqual(new['training'], baseline['training'])
        for key in ('training', 'protocol', 'purpose'):
            new[key] = old[key]
        self.assertEqual(old, new)
        self.assertNotIn('learnable_relation_scale', json.loads((ROOT / 'configs/train_monai_relation_whole_volume_overfit.json').read_text())['relation'])

    def test_legacy_state_rng_and_full_output_gradients_match_previous_formula(self):
        seed_all(12); model = self.model(False)
        fusion = model.bottleneck.fusion
        self.assertIsNone(fusion.relation_scale)
        self.assertFalse(any('relation_scale' in k for k in model.state_dict()))
        image = torch.randn(1, 1, 17, 19, 21, requires_grad=True)
        output = model(image)
        params = tuple(model.parameters())
        ga = torch.autograd.grad(output.final_logits.square().mean() + output.coarse_logits.square().mean(), (image, *params))
        # Independent literal pre-change fusion, inside the complete same model.
        with patch.object(fusion, 'forward', side_effect=lambda f, g: f + fusion.phi(g)):
            reference = model(image)
            gb = torch.autograd.grad(reference.final_logits.square().mean() + reference.coarse_logits.square().mean(), (image, *params))
        for a, b in zip(output, reference): torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(ga, gb): torch.testing.assert_close(a, b, rtol=0, atol=0)
        explicit = config(False)['relation'] | dict(learnable_relation_scale=False, relation_scale_init=1.)
        seed_all(12)
        other = MonaiRelationUNet(config(False)['model'], config(False)['input_processing'], relation_enabled=True, relation_config=explicit)
        self.f.assert_nested_equal(model.state_dict(), other.state_dict())
        other.load_state_dict(model.state_dict(), strict=True)

    def test_gated_formula_scales_phi_bias_not_input(self):
        fusion = ResidualFusion(2, content_channels=1, bias=True,
            learnable_relation_scale=True, relation_scale_init=.1).double()
        with torch.no_grad():
            fusion.relation_scale.fill_(.1)
            fusion.phi.weight.fill_(2.)
            fusion.phi.bias.copy_(torch.tensor([3., -2.], dtype=torch.float64))
        f = torch.ones(1,2,2,3,4,dtype=torch.float64,requires_grad=True)
        g = torch.full((1,1,2,3,4), 4., dtype=torch.float64,requires_grad=True)
        actual, stats = fusion.forward_with_diagnostics(f, g, epsilon=1e-6)
        expected = f + .1 * fusion.phi(g)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, f + fusion.phi(.1*g)))
        actual.sum().backward()
        self.assertIsInstance(fusion.relation_scale, torch.nn.Parameter)
        self.assertTrue(fusion.relation_scale.requires_grad)
        self.assertTrue(torch.isfinite(fusion.relation_scale.grad))
        self.assertNotEqual(fusion.relation_scale.grad.item(), 0)
        denominator = f.detach().norm().item() + 1e-6
        self.assertAlmostEqual(stats['writeback_to_feature_norm'], fusion.phi(g).detach().norm().item()/denominator)
        self.assertAlmostEqual(stats['scaled_writeback_to_feature_norm'], (.1*fusion.phi(g)).detach().norm().item()/denominator)

    def test_single_parameter_same_other_parameters_and_rng(self):
        seed_all(31); legacy = self.model(False); rng_old = capture_rng(torch.device('cpu'))
        seed_all(31); gated = self.model(); rng_new = capture_rng(torch.device('cpu'))
        self.f.assert_nested_equal(rng_old, rng_new)
        old, new = legacy.state_dict(), gated.state_dict()
        added = set(new) - set(old)
        self.assertEqual(len(added), 1)
        key = added.pop(); self.assertTrue(key.endswith('fusion.relation_scale'))
        self.assertEqual(new[key].ndim, 0)
        self.assertAlmostEqual(new[key].item(), .1)
        for name in old: torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in gated.parameters()) - sum(p.numel() for p in legacy.parameters()), 1)

    def test_final_loss_reaches_gamma_relation_writeback_fusion(self):
        seed_all(17); model = self.model()
        cfg = config(); criterion = MonaiRelationLoss(cfg['loss'], **cfg['coarse_supervision'])
        sample = Cases()[0]
        output = model(sample.image.unsqueeze(0))
        criterion.branch(output.final_logits, sample.label.unsqueeze(0)).total.backward()
        gamma_grad = model.bottleneck.fusion.relation_scale.grad
        self.assertIsNotNone(gamma_grad)
        self.assertTrue(torch.isfinite(gamma_grad)); self.assertNotEqual(gamma_grad.item(), 0.)
        for group in ('relation', 'node_to_space', 'fusion'):
            parameters = model.diagnostic_parameter_groups()[group]
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in parameters))
            self.assertGreater(sum(p.grad.square().sum().item() for _,p in parameters), 0.)

    def test_observers_do_not_change_output_gradients_rng_or_cache(self):
        seed_all(91); model = self.model(); x = Cases()[0].image.unsqueeze(0)
        before = capture_rng(torch.device('cpu'))
        plain = model(x)
        a = torch.autograd.grad(plain.final_logits.square().mean(), tuple(model.parameters()))
        observed, stats = model.forward_with_relation_diagnostics(x)
        b = torch.autograd.grad(observed.final_logits.square().mean(), tuple(model.parameters()))
        for p,q in zip(plain,observed): torch.testing.assert_close(p,q,rtol=0,atol=0)
        for p,q in zip(a,b): torch.testing.assert_close(p,q,rtol=0,atol=0)
        self.f.assert_nested_equal(before, capture_rng(torch.device('cpu')))
        self.assertTrue(all(not torch.is_tensor(v) for v in stats.values()))
        self.assertIsNone(model.bottleneck._relation_diagnostics_sink)
        with patch.object(model.bottleneck.fusion, 'forward_with_diagnostics', side_effect=RuntimeError('failure')):
            with self.assertRaisesRegex(RuntimeError, 'failure'): model.forward_with_relation_diagnostics(x)
        self.assertIsNone(model.bottleneck._relation_diagnostics_sink)
        self.assertIsNone(model.bottleneck._coarse_sink)

    def test_negative_zero_gamma_and_zero_feature_norm(self):
        fusion = ResidualFusion(1, content_channels=1, bias=True, learnable_relation_scale=True, relation_scale_init=-.5)
        f = torch.zeros(1,1,2,2,2); g = torch.ones_like(f)
        for value in (-.5, 0., 2.):
            with torch.no_grad(): fusion.relation_scale.fill_(value)
            out, stats = fusion.forward_with_diagnostics(f,g,epsilon=1e-6)
            torch.testing.assert_close(out, f + value*fusion.phi(g), rtol=0,atol=0)
            self.assertGreaterEqual(stats['scaled_writeback_to_feature_norm'], 0.)
            self.assertAlmostEqual(stats['scaled_writeback_to_feature_norm'], abs(value)*stats['writeback_to_feature_norm'], places=4)

    def test_bypass_and_legacy_no_fake_observations(self):
        seed_all(8); cfg=config(); reference=MonaiReferenceUNet(cfg['model'],cfg['input_processing'])
        seed_all(8); bypass=self.model(enabled=False)
        x=Cases()[0].image.unsqueeze(0)
        out, stats=bypass.forward_with_relation_diagnostics(x)
        torch.testing.assert_close(out.final_logits,reference(x).final_logits,rtol=0,atol=0)
        self.assertFalse(bypass.relation_scale_enabled); self.assertEqual(stats,{})
        self.assertFalse(any('relation_scale' in k for k in bypass.state_dict()))
        _, stats=self.model(False).forward_with_relation_diagnostics(x)
        self.assertEqual(stats,{})

    def test_resume_scalar_optimizer_trajectory_logs_and_tb_only_two_new_tags(self):
        fixture=relation_tests.MonaiRelationTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        cfg=config();cfg['training'].update(max_epochs=None,max_steps=2,early_stopping=None)
        with patch.object(fixture,'proposed_config',return_value=cfg):
            whole=fixture.trainer(self.root/'whole');self.f.quiet_run(whole)
            first=fixture.trainer(self.root/'resume');self.f.quiet_run(first,stop_after=1)
            second=fixture.trainer(first.run_dir,True);self.f.quiet_run(second)
        a=load_checkpoint(whole.run_dir/'last.ckpt',whole.identity)
        b=load_checkpoint(second.run_dir/'last.ckpt',second.identity)
        for key in ('model','optimizer','rng','progress','sampler_generator','loader_generator','development_state'):
            self.f.assert_nested_equal(a[key],b[key])
        self.assertTrue(any(k.endswith('.relation_scale') for k in b['model']))
        gamma=second.model.bottleneck.fusion.relation_scale
        self.assertEqual(second.optimizer.state[gamma]['step'].item(),2)
        rows=self.f.rows(second.run_dir)
        other=self.f.rows(whole.run_dir)
        for r,s in zip(rows,other): self.assertEqual(r.get('relation_scale'),s.get('relation_scale'))
        train=[r for r in rows if r['phase']=='train']
        self.assertEqual(scalar_values(train[0]), {})
        epoch = next(r for r in rows if r['phase']=='train_epoch')
        self.assertTrue({'Relation/Gamma','Relation/WritebackToFeatureNorm'} <= set(scalar_values(epoch)))
        self.assertIn('writeback_to_feature_norm',train[0]['relation_scale'])
        self.assertIn('diagnostics',train[0])
        self.assertTrue(any(r.get('relation_scale_cases') for r in rows))
        events=EventAccumulator(str(second.board.directory),size_guidance={'scalars':0}).Reload()
        for tag in ('Relation/Gamma','Relation/WritebackToFeatureNorm'):
            self.assertEqual([e.step for e in events.Scalars(tag)],[2])
        self.assertFalse(any(t.startswith(('GradNorm/','System/','Optimizer/')) for t in events.Tags()['scalars']))
        bad=copy.deepcopy(second.identity);bad['relation']['relation_scale_init']=.2
        with self.assertRaises(ValueError):load_checkpoint(second.run_dir/'last.ckpt',bad)

    def test_preflight_reports_scale_geometry_gradient_without_optimizer_step(self):
        seed_all(17);model=self.model();cfg=config()
        criterion=MonaiRelationLoss(cfg['loss'],**cfg['coarse_supervision'])
        before={k:v.clone() for k,v in model.state_dict().items()}
        class Synthetic(Cases):
            def __getitem__(self,i):
                s=super().__getitem__(i)
                return FullScanSample(s.image,s.label,{'image':{'original_shape_ijk':[21,19,17]}})
        identity=dict(mode=cfg['mode'],parameter_count=sum(p.numel() for p in model.parameters()),
            model=cfg['model'],loss=cfg['loss'],preprocessing={},environment={},provenance={'git':{}},
            training=cfg['training'],relation=cfg['relation'],coarse_supervision=cfg['coarse_supervision'])
        with contextlib.redirect_stdout(io.StringIO()) as stream, patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('no step')):
            preflight_backward(model,criterion,Synthetic(),identity)
        lines=stream.getvalue().splitlines()
        result=json.loads(next(line.split(' ',1)[1] for line in lines if line.startswith('preflight_result ')))
        for key in ('gamma_grad','final_loss','coarse_loss','joint_loss'):self.assertIsNotNone(result[key])
        self.assertEqual(result['optimizer_steps'],0)
        self.assertEqual(result['missing_gradients'],[]);self.assertEqual(result['nonfinite_gradients'],[])
        self.assertNotEqual(result['gamma_grad'],0.)
        self.assertIn('original_shape_ijk',stream.getvalue());self.assertIn('bottleneck_shape',stream.getvalue())
        self.assertIn('gamma_initial',stream.getvalue());self.assertIn('scaled_writeback_to_feature_norm',stream.getvalue())
        self.f.assert_nested_equal(before,model.state_dict())

    def test_invalid_scale_configuration(self):
        for kwargs in (dict(learnable_relation_scale=1),dict(relation_scale_init=.1),
                       dict(learnable_relation_scale=True,relation_scale_init=float('nan'))):
            with self.assertRaises(ValueError):ResidualFusion(1,content_channels=1,bias=True,**kwargs)


if __name__=='__main__':unittest.main()

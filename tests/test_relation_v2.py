"""关系推理 v2：归一化 + 残差读出、包含比例读出与损失、先验平移；省略新配置项时行为不变。"""
import copy
import json
import unittest

import torch

from pathlib import Path

import support
from organ_relation.losses import RegionJointLoss, containment_loss, containment_targets, region_targets
from organ_relation.models.brain_tumor import (PRIOR_SHIFT_CLAMP, PRIOR_SHIFT_GAIN, BrainTumorConfig,
                                               BrainTumorSegmentor)
from organ_relation.models.dynamic_relation import DynamicRelation


def nodes(scale=1.0, batch=2, channels=16):
    torch.manual_seed(3)
    return (scale * torch.randn(batch, 3, channels), torch.rand(batch, 3, 3), torch.rand(batch, 3, 1) * .05,
            torch.rand(batch, 3, 1))


def relation(**kwargs):
    torch.manual_seed(5)
    return DynamicRelation(16, relation_channels=8, rounds=2, node_count=3, **kwargs)


def model(**extra):
    torch.manual_seed(7)
    return BrainTumorSegmentor(BrainTumorConfig(**{**support.config()['model'], **extra}))


V2 = dict(relation_conditioning='layernorm_residual')


class RelationV2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_omitted_options_keep_parameters_initialization_and_identity(self):
        for graph in ('complete', 'nested_chain'):
            legacy, conditioned = relation(graph=graph), relation(graph=graph, conditioning='layernorm_residual',
                                                                  containment=True)
            old = legacy.state_dict()
            new = conditioned.state_dict()
            self.assertEqual(set(new) - set(old), {'state_norm.weight', 'state_norm.bias', 'readout.weight',
                                                   'containment_hidden.weight', 'containment_hidden.bias',
                                                   'containment_output.weight', 'containment_output.bias'})
            for key, value in old.items():  # 新模块排在原有参数之后构建，原有参数的初始化不变
                self.assertTrue(torch.equal(value, new[key]), key)
        config = BrainTumorConfig(**support.config()['model'])
        self.assertNotIn('relation_conditioning', config.to_dict())
        self.assertNotIn('containment', config.to_dict())
        self.assertEqual(BrainTumorConfig(**support.config()['model'], **V2, containment='prior_shift').to_dict()['containment'],
                         'prior_shift')

    def test_invalid_combinations_rejected(self):
        base = support.config()['model']
        for extra in (dict(containment='supervised'), dict(relation_conditioning='other'),
                      dict(**V2, containment='other'), dict(**V2, variant='no_relation')):
            with self.assertRaises(ValueError):
                BrainTumorConfig(**base, **extra)
        with self.assertRaises(ValueError):
            relation(containment=True)

    def test_zero_initialized_readout_starts_as_no_relation_and_then_departs(self):
        z0, centroid, size, confidence = nodes()
        engine = relation(conditioning='layernorm_residual')
        state = engine.forward_with_last_edges(z0, centroid, size, confidence)
        self.assertTrue(torch.equal(state.zK, z0))
        self.assertIsNone(state.containment)
        with torch.no_grad():
            engine.readout.weight.normal_(std=.1)
        self.assertFalse(torch.equal(engine.forward_with_last_edges(z0, centroid, size, confidence).zK, z0))

    def test_gates_and_gru_stay_out_of_saturation_for_large_unnormalized_nodes(self):
        # 根因的回归测试：训练中的节点范数达 20–50。历史实现的门控与 GRU 随之饱和，v2 不受节点尺度影响。
        z0, centroid, size, confidence = nodes(scale=40.0)
        small = nodes(scale=1.0)[0]

        def gate_scores(engine, z):
            h = engine.state_norm(z) if engine.conditioning == 'layernorm_residual' else z
            return engine.relation_output(torch.relu(engine.relation_hidden(
                engine._relation_descriptors(h, centroid, size, confidence)))).squeeze(-1)

        legacy, conditioned = relation(), relation(conditioning='layernorm_residual')
        self.assertGreater(gate_scores(legacy, z0).abs().mean() / gate_scores(legacy, small).abs().mean(), 10)
        torch.testing.assert_close(gate_scores(conditioned, z0), gate_scores(conditioned, small), atol=1e-4, rtol=1e-4)
        result = conditioned(z0, centroid, size, confidence, return_diagnostics=True)
        off_diagonal = ~torch.eye(3, dtype=torch.bool)
        for round_ in result.rounds:
            alpha = round_.alpha[:, off_diagonal]
            self.assertTrue(((alpha > .05) & (alpha < .95)).all())
            self.assertLess(round_.z.abs().max(), 3.0)  # 归一化空间里的状态保持 O(1)

    def test_containment_logits_shape_gradients_and_edges(self):
        z0, centroid, size, confidence = nodes()
        for graph in ('complete', 'nested_chain'):
            engine = relation(graph=graph, conditioning='layernorm_residual', containment=True)
            state = engine.forward_with_last_edges(z0, centroid, size, confidence)
            self.assertEqual(state.containment.shape, (2, 2))
            state.containment.sum().backward()
            for name in ('containment_output.weight', 'containment_hidden.weight', 'gru.W_z.weight', 'state_norm.weight'):
                self.assertGreater(dict(engine.named_parameters())[name].grad.abs().sum(), 0, name)
            self.assertIsNone(engine.readout.weight.grad)  # 包含读出取自 h(K)，不经过残差读出

    def test_containment_targets_and_loss(self):
        label = torch.zeros(3, 2, 4, 4, dtype=torch.long)
        label[0, :, :2] = 2; label[0, 0, 0] = 1; label[0, 0, 1, :2] = 3   # WT 16，TC 6，ET 2
        label[1, 0, 0] = 2                                                  # WT 4，无 TC、无 ET
        target = region_targets(label)
        ratio, valid = containment_targets(target)
        torch.testing.assert_close(ratio, torch.tensor([[6 / 16, 2 / 6], [0., 0.], [0., 0.]], dtype=torch.float64))
        self.assertEqual(valid.tolist(), [[True, True], [True, False], [False, False]])
        logits = torch.tensor([[.3, -.2], [1., 5.], [2., -2.]])
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        expected = torch.stack([
            (bce(logits[0, 0], torch.tensor(6 / 16)) + bce(logits[0, 1], torch.tensor(2 / 6))) / 2,
            bce(logits[1, 0], torch.tensor(0.)), torch.tensor(0.)])
        torch.testing.assert_close(containment_loss(logits, target), expected)

    def test_loss_requires_matching_configuration_and_adds_weighted_term(self):
        cfg = support.config()['loss']
        label = (torch.arange(2 * 4 * 6 * 6).reshape(2, 4, 6, 6) % 4).long()
        coarse, final, u = torch.randn(2, 3, 2, 3, 3), torch.randn(2, 3, 4, 6, 6), torch.randn(2, 2)
        plain = RegionJointLoss(**dict(cfg, lambda_hier=0.0))
        with_term = RegionJointLoss(**dict(cfg, lambda_hier=0.0, lambda_contain=0.5))
        with self.assertRaises(ValueError):
            plain(coarse, final, label, u)
        with self.assertRaises(ValueError):
            with_term(coarse, final, label)
        with self.assertRaises(ValueError):
            RegionJointLoss(**dict(cfg, lambda_contain=0.5))  # 需要显式的 lambda_hier
        a, b = plain(coarse, final, label), with_term(coarse, final, label, u)
        torch.testing.assert_close(b.containment, containment_loss(u, region_targets(label)))
        torch.testing.assert_close(b.per_case, a.per_case + 0.5 * b.containment)
        torch.testing.assert_close(b.total, b.per_case.mean())

    def test_model_outputs_and_prior_shift(self):
        image = torch.randn(1, 4, 9, 11, 13)
        for plain in (model()(image), model(**V2)(image)):  # 未启用时仍是两个字段，历史的解包方式不变
            self.assertEqual(len(plain), 2)
        supervised, shifted = model(**V2, containment='supervised'), model(**V2, containment='prior_shift')
        self.assertEqual(set(shifted.state_dict()) - set(supervised.state_dict()), {'prior_shift_scale'})
        shifted.load_state_dict(supervised.state_dict(), strict=False)
        reference, start = supervised(image), shifted(image)
        self.assertEqual(reference.containment_logits.shape, (1, 2))
        self.assertTrue(torch.equal(start.final_logits, reference.final_logits))  # a = 0：起点没有平移
        with torch.no_grad():
            shifted.prior_shift_scale.copy_(torch.tensor([.05, -.02]))
            shifted.relation.containment_output.bias.fill_(9.0)  # 让 u 超出截断范围
        out = shifted(image)
        u = out.containment_logits[0].clamp(-PRIOR_SHIFT_CLAMP, PRIOR_SHIFT_CLAMP)
        self.assertTrue((out.containment_logits[0] > PRIOR_SHIFT_CLAMP).all())
        delta = out.final_logits - reference.final_logits
        torch.testing.assert_close(delta[0, 0], torch.zeros_like(delta[0, 0]))
        torch.testing.assert_close(delta[0, 1], torch.full_like(delta[0, 1], PRIOR_SHIFT_GAIN * .05 * u[0].item()), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(delta[0, 2], torch.full_like(delta[0, 2], PRIOR_SHIFT_GAIN * -.02 * u[1].item()), atol=1e-5, rtol=1e-5)

    def test_segmentation_loss_reaches_shift_scale_but_not_the_ratio_estimate(self):
        image = torch.randn(1, 4, 9, 11, 13)
        label = (torch.arange(9 * 11 * 13).reshape(1, 9, 11, 13) % 4).long()
        net = model(**V2, containment='prior_shift')
        with torch.no_grad():
            net.prior_shift_scale.fill_(.03)
        out = net(image)
        RegionJointLoss(**dict(support.config()['loss'], lambda_hier=0.0, lambda_contain=0.0))(
            out.coarse_logits, out.final_logits, label, out.containment_logits).total.backward()
        self.assertGreater(net.prior_shift_scale.grad.abs().sum(), 0)
        for parameter in (net.relation.containment_hidden.weight, net.relation.containment_output.weight):
            self.assertEqual(parameter.grad.abs().sum(), 0)  # lambda_contain = 0 且平移处 detach：没有梯度

    def test_existence_gate_matches_probability_product_and_is_finite(self):
        from organ_relation.models.brain_tumor import existence_gate
        logits = torch.tensor([-30., -2., 0., 3., 30.]).view(1, 1, 5, 1, 1).repeat(2, 3, 1, 1, 1)
        existence = torch.tensor([[40., 40.], [0., 0.], [-40., 3.], [5., -40.], [4., -1.]])
        for e in existence:
            gated = existence_gate(logits, e.expand(2, 2))
            self.assertTrue(torch.isfinite(gated).all())
            self.assertTrue(torch.equal(gated[:, 0], logits[:, 0]))
            pi_tc = torch.sigmoid(e[0].double()); pi_et = pi_tc * torch.sigmoid(e[1].double())
            torch.testing.assert_close(gated[:, 1].double().sigmoid(), pi_tc * logits[:, 1].double().sigmoid(), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(gated[:, 2].double().sigmoid(), pi_et * logits[:, 2].double().sigmoid(), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(existence_gate(logits, existence[:1].expand(2, 2)), logits, atol=1e-3, rtol=1e-5)  # pi = 1：不改变 logit
        self.assertTrue((existence_gate(logits, existence[2:3].expand(2, 2))[:, 1:] < 0).all())   # TC 不存在：TC、ET 全关
        with self.assertRaises(ValueError):
            existence_gate(logits, existence)

    def test_gate_model_readout_loss_and_gradients(self):
        image = torch.randn(1, 4, 9, 11, 13)
        label = (torch.arange(9 * 11 * 13).reshape(1, 9, 11, 13) % 4).long()
        net = model(**V2, containment='gate', graph='nested_chain')
        out = net(image)
        self.assertEqual(out.containment_logits.shape, (1, 2, 2))
        self.assertGreater(out.containment_logits[..., 1].min(), 2.0)  # 存在 logit 从 +4 附近起步
        target = region_targets(label)
        ratio, valid = containment_targets(target)
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        expected = ((bce(out.containment_logits[..., 0], ratio.float(), reduction='none')
                     + bce(out.containment_logits[..., 1], (ratio > 0).float(), reduction='none')) * valid).sum(1) / valid.sum(1)
        torch.testing.assert_close(containment_loss(out.containment_logits, target), expected)
        loss = RegionJointLoss(**dict(support.config()['loss'], lambda_hier=0.0, lambda_contain=0.5))(
            out.coarse_logits, out.final_logits, label, out.containment_logits)
        loss.total.backward()
        self.assertFalse([n for n, p in net.named_parameters() if p.grad is None])
        # 把 ET 的存在 logit 压到很低：ET 整块被关掉，TC 基本不变
        with torch.no_grad():
            net.relation.containment_output.bias[1] = -30.0
        closed = net(image)
        self.assertTrue((closed.final_logits[:, 1:] < 0).all())
        self.assertTrue(torch.equal(closed.final_logits[:, 0], out.final_logits[:, 0]))

    def test_p2_configs_differ_by_one_item_and_share_the_protocol(self):
        folder = support.ROOT / 'configs/relation_v2'
        rows = {p.stem.removeprefix('p2_'): json.loads(p.read_text()) for p in folder.glob('*.json')}
        self.assertEqual(sorted(rows), sorted([
            'r0_unet', 'r0_unet_seed2', 'r1_drf_backbone', 'r2_nodes', 'r2_nodes_seed2', 'r3_relation', 'r3_legacy_relation',
            'r3a_gate', 'r3ab_multiscale', 'r3ac_intensity', 'r3ad_nested', 'r4_full', 'r4_soft_hierarchy']))
        science = lambda c: {k: v for k, v in c.items() if k != 'status' and not k.endswith('_note')}

        def changed(name, *changes):
            result = copy.deepcopy(science(rows[name]))
            for section, key, value in changes:
                if value is None:
                    del result[section][key]
                else:
                    result[section][key] = value
            return result
        steps = (('r1_drf_backbone', 'r0_unet', [('model', 'variant', 'unet')]),
                 ('r0_unet', 'r0_unet_seed2', [('training', 'seed', 20260926)]),
                 ('r1_drf_backbone', 'r2_nodes', [('model', 'variant', 'no_relation')]),
                 ('r2_nodes', 'r2_nodes_seed2', [('training', 'seed', 20260926)]),
                 ('r2_nodes', 'r3_relation', [('model', 'variant', None), ('model', 'relation_conditioning', 'layernorm_residual')]),
                 ('r3_relation', 'r3_legacy_relation', [('model', 'relation_conditioning', None)]),
                 ('r3_relation', 'r3a_gate', [('model', 'containment', 'gate'), ('loss', 'lambda_contain', 0.5)]),
                 ('r3a_gate', 'r3ab_multiscale', [('model', 'node_inputs', ['multiscale'])]),
                 ('r3a_gate', 'r3ac_intensity', [('model', 'node_inputs', ['intensity'])]),
                 ('r3a_gate', 'r3ad_nested', [('model', 'output_hierarchy', 'nested')]),
                 ('r3a_gate', 'r4_full', [('model', 'node_inputs', ['intensity', 'multiscale']), ('model', 'output_hierarchy', 'nested')]),
                 ('r4_full', 'r4_soft_hierarchy', [('model', 'output_hierarchy', None), ('loss', 'lambda_hier', 1.0)]))
        for before, after, change in steps:
            with self.subTest(step=f'{before}->{after}'):
                self.assertEqual(science(rows[after]), changed(before, *change))
        # 与 dev 上已批准的 r4_full 相比：r4_soft_hierarchy 只多了关系推理 v2 的各项与协议 P2 的三项
        old = json.loads((support.ROOT / 'configs/experiments/r4_full.json').read_text())
        expected = copy.deepcopy(science(old))
        expected['model'].update(relation_conditioning='layernorm_residual', containment='gate',
                                 node_inputs=['intensity', 'multiscale'])
        expected['loss']['lambda_contain'] = 0.5
        expected['training'].update(max_epochs=80, scheduler={'kind': 'cosine', 'min_lr': 1e-06})
        del expected['training']['early_stopping']
        self.assertEqual(science(rows['r4_soft_hierarchy']), expected)
        queue = json.loads((support.ROOT / 'configs/queues/brats2018_relation_v2.json').read_text())
        jobs = [(Path(j['config']).stem.removeprefix('p2_'), j['fold']) for j in queue['jobs']]
        on_amd = [name for name in rows if not name.startswith('r0_unet')]  # U-Net 基线只在 RTX 5060 上跑
        self.assertEqual(sorted(jobs), sorted((name, fold) for name in on_amd for fold in range(5)))
        second = json.loads((support.ROOT / 'configs/queues/brats2018_relation_v2_rtx5060.json').read_text())
        self.assertEqual(second['workers'], 1)
        self.assertEqual([(Path(j['config']).stem, j['fold']) for j in second['jobs'][:5]], [('p2_r0_unet', f) for f in range(5)])
        self.assertTrue(all((support.ROOT / j['config']).is_file() for j in second['jobs']))
        self.assertEqual([fold for _, fold in jobs], sorted(fold for _, fold in jobs))  # 按折排：先所有方案的 fold0
        for name, cfg in rows.items():
            with self.subTest(arm=name):
                BrainTumorConfig(**cfg['model'])
                RegionJointLoss(**cfg['loss'])

    def test_nested_output_matches_probability_products_and_orders_regions(self):
        from organ_relation.models.brain_tumor import nested_logits
        logits = torch.tensor([-40., -3., 0., 2., 40.])
        grid = torch.stack(torch.meshgrid(logits, logits, logits, indexing='ij')).reshape(1, 3, 5, 5, 5)
        nested = nested_logits(grid)
        self.assertTrue(torch.isfinite(nested).all())
        self.assertTrue(torch.equal(nested[:, 0], grid[:, 0]))
        p = grid.double().sigmoid()
        torch.testing.assert_close(nested[:, 1].double().sigmoid(), p[:, 0] * p[:, 1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(nested[:, 2].double().sigmoid(), p[:, 0] * p[:, 1] * p[:, 2], atol=1e-6, rtol=1e-5)
        self.assertTrue((nested[:, 2] <= nested[:, 1] + 1e-4).all() and (nested[:, 1] <= nested[:, 0] + 1e-4).all())
        from organ_relation.losses import hierarchy_violation
        self.assertLess(hierarchy_violation(nested).max(), 1e-6)  # 嵌套输出没有层级违例
        with self.assertRaises(ValueError):
            nested_logits(grid[:, :2])

    def test_node_inputs_configuration_shapes_and_gradients(self):
        base = support.config()['model']
        for bad in (['multiscale', 'intensity'], [], ['other'], 'multiscale', ['intensity', 'intensity']):
            with self.assertRaises(ValueError):
                BrainTumorConfig(**base, **V2, node_inputs=bad)
        with self.assertRaises(ValueError):
            BrainTumorConfig(**base, node_inputs=['intensity'])  # 需要 layernorm_residual
        with self.assertRaises(ValueError):
            BrainTumorConfig(**base, output_hierarchy='other')
        image = torch.randn(1, 4, 9, 11, 13)
        label = (torch.arange(9 * 11 * 13).reshape(1, 9, 11, 13) % 4).long()
        plain = model(**V2, containment='gate')
        for inputs, names in ((['multiscale'], {'scale_norms', 'scale_projections'}), (['intensity'], {'intensity_projection'}),
                              (['intensity', 'multiscale'], {'scale_norms', 'scale_projections', 'intensity_projection'})):
            with self.subTest(inputs=inputs):
                net = model(**V2, containment='gate', node_inputs=inputs, output_hierarchy='nested')
                self.assertEqual({k.split('.')[0] for k in set(net.state_dict()) - set(plain.state_dict())}, names)
                for key, value in plain.state_dict().items():  # 新模块排在最后构建：其余参数的初始化不变
                    self.assertTrue(torch.equal(value, net.state_dict()[key]), key)
                features = net.encoder(image)
                extra = net.node_extra(features, net.coarse_head(features[-1]).probabilities, image)
                self.assertEqual(extra.shape, (1, 3, 256))
                out = net(image)
                RegionJointLoss(**dict(support.config()['loss'], lambda_hier=0.0, lambda_contain=0.5))(
                    out.coarse_logits, out.final_logits, label, out.containment_logits).total.backward()
                self.assertFalse([n for n, p in net.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()])
        # 强度输入：把 T1ce 整体加 1，只有强度统计变化
        net = model(**V2, containment='gate', node_inputs=['intensity'])
        with torch.no_grad():
            features = net.encoder(image)
            probabilities = net.coarse_head(features[-1]).probabilities
            shifted = image.clone(); shifted[:, 1] += 1
            self.assertFalse(torch.allclose(net.node_extra(features, probabilities, image),
                                            net.node_extra(features, probabilities, shifted)))

    def test_mixed_precision_forward_keeps_fp32_logits(self):
        net = BrainTumorSegmentor(BrainTumorConfig(**{**support.config()['model'], **V2, 'containment': 'gate',
                                                      'node_inputs': ['intensity', 'multiscale'], 'output_hierarchy': 'nested'}),
                                  convolution_backend='depth_sliced', mixed_precision='bfloat16')
        out = net(torch.randn(1, 4, 9, 11, 13).contiguous(memory_format=torch.channels_last_3d))
        self.assertEqual((out.final_logits.dtype, out.containment_logits.dtype), (torch.float32, torch.float32))


class Cases(torch.utils.data.Dataset):
    """三例小体积：第一例三个区域都有，第二例没有 ET，第三例只有水肿（没有 TC）。"""

    def __len__(self):
        return 3

    def __getitem__(self, index):
        from organ_relation.data.full_scan import FullScanSample
        generator = torch.Generator().manual_seed(40 + index)
        image = torch.randn(4, 8, 8, 8, generator=generator)
        label = torch.zeros(8, 8, 8, dtype=torch.long)
        label[1:7, 1:7, 1:7] = 2
        if index < 2:
            label[2:6, 2:6, 2:6] = 1
        if index == 0:
            label[3:5, 3:5, 3:5] = 3
        return FullScanSample(image, label, {})


class P2TrainerTests(unittest.TestCase):
    """协议 P2（按轮、余弦调度、无早停）+ 存在性门控模型：训练、验证、日志字段与中途续训。"""

    def setUp(self):
        self.fixture = support.TrainingTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        torch.set_num_threads(1)

    def trainer(self, path, *, resume=False, console=None):
        from organ_relation.training.engine import Trainer
        from organ_relation.training.state import seed_all
        formal = json.loads((support.ROOT / 'configs/relation_v2/p2_r4_full.json').read_text())
        seed_all(311)
        model_config = {**support.config()['model'], 'graph': 'nested_chain', **V2, 'containment': 'gate',
                        'node_inputs': ['intensity', 'multiscale'], 'output_hierarchy': 'nested'}
        net = BrainTumorSegmentor(BrainTumorConfig(**model_config)).to(memory_format=torch.channels_last_3d)
        options = dict(formal['training'], max_epochs=2)
        loss = dict(support.config()['loss'], lambda_hier=0.0, lambda_contain=0.5)
        identity = dict(mode='brats_regions', model=model_config, loss=loss, threshold=.5, training=options,
                        preprocessing=formal['preprocessing'], provenance='test', data={'manifest_hash': 'synthetic'})
        names = ['full', 'no_et', 'edema_only']
        return Trainer(net, RegionJointLoss(**loss), torch.optim.AdamW(net.parameters(), lr=3e-4, foreach=False, fused=False),
                       Cases(), names, device='cpu', options=options, identity=identity, run_dir=path,
                       resume=path / 'last.ckpt' if resume else None, validation_dataset=Cases(),
                       validation_case_ids=names, tensorboard=False, console=console)

    def test_trains_validates_logs_and_resumes_exactly(self):
        from organ_relation.training.console import TrainingConsole
        from organ_relation.training.engine import scheduled_lr
        from organ_relation.training.state import load_checkpoint
        root = self.fixture.root
        whole = self.trainer(root / 'whole', console=TrainingConsole())
        self.fixture.quiet_run(whole)
        rows = self.fixture.rows(whole.run_dir)
        train = [r for r in rows if r['phase'] == 'train']
        self.assertEqual(len(train), 6)
        self.assertEqual([r['lr'] for r in train],
                         [scheduled_lr({'kind': 'cosine', 'min_lr': 1e-06}, 3e-4, step, 6) for step in range(1, 7)])
        self.assertTrue(all(r['containment'] > 0 and 'hierarchy' in r for r in train))
        validations = [r for r in rows if r['phase'] == 'internal_dev']
        self.assertEqual([r['epoch'] for r in validations], [1, 2])
        self.assertTrue(all('early_stopping' not in r for r in validations))
        for case in validations[-1]['region_diagnostic_cases'].values():
            self.assertEqual((len(case['containment_logits']), len(case['containment_logits'][0])), (2, 2))
        self.assertTrue(all(r['hierarchy'] < 1e-6 for r in train))  # 嵌套输出：层级违例恒为 0
        self.assertTrue((whole.run_dir / 'best-dev.ckpt').is_file())
        part = self.trainer(root / 'part'); self.fixture.quiet_run(part, stop_after=4)
        resumed = self.trainer(part.run_dir, resume=True); self.fixture.quiet_run(resumed)
        a = load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)
        b = load_checkpoint(part.run_dir / 'last.ckpt', resumed.identity)
        for key in ('model', 'optimizer', 'rng', 'progress'):
            self.fixture.assert_nested_equal(a[key], b[key])
        self.assertEqual(a['progress']['epoch'], 2)
        again = self.trainer(part.run_dir, resume=True)  # 已跑满的 run 再次启动：不再训练，直接结束
        self.assertEqual(self.fixture.quiet_run(again)['global_step'], 6)


if __name__ == '__main__':
    unittest.main()

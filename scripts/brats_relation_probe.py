"""Read-only CPU probe: does the region-graph signal actually reach the decoder?

对 BraTS 区域模型（完整方法或 no_relation 变体）的 checkpoint，在该折全部（或抽样）验证病例上做
CPU FP32 推理（原生卷积、连续内存布局），逐例记录：

1. 深层回写：||gamma*phi(G)||/||F||（实际扰动）与 ||phi(G)||/||F||（未缩放），整幅与"肿瘤区"各一次，
   以及 F 与 F' 的余弦相似度；
2. 解码器三个尺度：||eta_l*Phi_l(G_l)||/||E_l||（整幅与肿瘤区）和尺度权重；
3. 关系推理（仅完整方法）：每轮边权 alpha、||消息||/||z||、本轮节点变化中由消息带来的比例
   ||GRU(m,z)-GRU(0,z)|| / ||GRU(m,z)-z||，以及 z(K) 相对 z(0) 的改变；
4. 推理时敲除：gamma=0、eta=0、两者都为 0；完整方法另有"边权全部置 0、保留 GRU 轮次"
   与"跳过关系轮次 z(K):=z(0)"。每项给出 Dice、预测翻转比例与 logit 相对变化。

编码器每例只算一次，各敲除只重算最深层之后的部分；基线与模型原生 forward 逐位一致（有测试）。
真值只用于评估 Dice 和划定统计用的"肿瘤区"（WT 真值在各特征网格上的最大池化），不进入模型。
不改写 runs/、源数据与 checkpoint；若给了 --data-cache，缺失的缓存条目会像训练时一样被补写。

用法：
  python scripts/brats_relation_probe.py probe --run E1=runs/<完整方法> --run A3=runs/<no_relation> \
      --output <新的 .jsonl> [--data-cache cache/npy] [--cases all|N] [--threads 8]
  python scripts/brats_relation_probe.py summarize <.jsonl> [<.jsonl> ...]
同一次 probe 的各个 run 必须使用相同的数据清单、划分与预处理。输出文件已存在时，只有表头一致才续跑。
数值来自 CPU FP32，与训练时的 GPU bf16 不逐位相同；逐例与该 run 自己的验证记录对照，汇总里给出差异。
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import torch.nn.functional as functional
from torch.func import functional_call

from organ_relation.data.brats import BraTSDataset, index_cases
from organ_relation.losses import REGION_NAMES, region_targets
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.provenance import code_hashes, git_state, sha256
from organ_relation.training.brain_tumor import region_dice
from organ_relation.training.state import digest, load_checkpoint

SCHEMA = 'brats_relation_probe_v1'
COMMON_VARIANTS = (('gamma_zero', dict(gamma=0.0)), ('eta_zero', dict(eta=0.0)),
                   ('gamma_eta_zero', dict(gamma=0.0, eta=0.0)))
RELATION_VARIANTS = (('edges_zero', dict(relation='edges_zero')), ('relation_skipped', dict(relation='skipped')))


def norm(x, mask=None):
    x = x.double()
    return torch.linalg.vector_norm(x if mask is None else x * mask).item()


def ratio(numerator, denominator, mask=None):
    return norm(numerator, mask) / (norm(denominator, mask) + 1e-12)


def tumor_grid(tumor, shape):
    """全分辨率 WT 真值 → 特征网格上的统计区域（最大池化）；只划定统计范围，不进入模型。"""
    return (functional.adaptive_max_pool3d(tumor[None, None].double(), tuple(shape)) > 0).double()


class _RecordDecoderWriteback:
    """在一次解码中记录各尺度回写占比；退出时去掉实例上的替身方法，恢复类方法。"""

    def __init__(self, decoder, record, tumor):
        self.queries, self.record, self.tumor = list(decoder.queries), record, tumor

    def __enter__(self):
        for query in self.queries:
            original = query.forward_projected

            def observed(skip, zK, weights, projection, scale, _original=original):
                content = _original(skip, zK, weights, projection, scale)
                mask = tumor_grid(self.tumor, skip.shape[2:])
                self.record.append(dict(eta=float(scale), ratio=ratio(content, skip),
                                        ratio_tumor=ratio(content, skip, mask),
                                        scale_weights=weights[0, :, 0].tolist()))
                return content
            query.forward_projected = observed
        return self

    def __exit__(self, *exc):
        for query in self.queries:
            query.__dict__.pop('forward_projected', None)
        return False


def check_probe_model(model):
    if not isinstance(model, BrainTumorSegmentor) or model.variant not in (None, 'no_relation'):
        raise ValueError('relation probe needs the full method or the no_relation variant (nodes, writeback, region decoder)')


@torch.no_grad()
def probe_tail(model, features, *, gamma=None, eta=None, relation='on', record=None, tumor=None):
    """最深层之后的前向（与 BrainTumorSegmentor._run 相同的运算次序），可敲除 gamma、eta、边或关系轮次。

    relation: 'on' 正常；'edges_zero' 保留 GRU 轮次但边权全部为 0（节点之间不交流），尺度描述中的
    入边权重也为 0；'skipped' 不做关系轮次，z(K):=z(0)，入边权重为 0（与 no_relation 变体的前向相同）。
    """
    check_probe_model(model)
    if relation not in ('on', 'edges_zero', 'skipped'):
        raise ValueError('relation must be on, edges_zero or skipped')
    if model.variant == 'no_relation' and relation != 'on':
        raise ValueError('the no_relation variant has no relation rounds to knock out')
    if record is not None and tumor is None:
        raise ValueError('recording needs the tumor region for its statistics')
    F = features[-1]
    coarse = model.coarse_head(F)
    nodes = model.space_to_node(F, coarse.probabilities)
    rounds = []
    if model.variant == 'no_relation' or relation == 'skipped':
        zK, last_alpha = nodes.z0, nodes.z0.new_zeros(nodes.z0.shape[0], 3, 3)
    else:
        engine, z = model.relation, nodes.z0
        for _ in range(engine.rounds):
            alpha = engine._edge_weights(z, nodes.centroid, nodes.size, nodes.confidence)
            if relation == 'edges_zero':
                alpha = torch.zeros_like(alpha)
            message = engine._aggregate_messages(z, alpha)
            updated = engine.gru(message, z)
            if record is not None:
                silent = engine.gru(torch.zeros_like(message), z)
                rounds.append(dict(alpha=alpha[0].tolist(), message_to_z=ratio(message, z),
                                   message_share_of_update=ratio(updated - silent, updated - z)))
            z = updated
        zK, last_alpha = z, alpha
    writeback = model.fusion.phi(model.node_to_space(F, zK))
    scale = model.fusion.relation_scale if gamma is None else writeback.new_tensor(float(gamma))
    fused = F + scale * writeback
    overrides = {} if eta is None else {'eta': torch.full_like(model.decoder.eta, float(eta))}
    arguments = (fused, features, zK, nodes, last_alpha)
    if record is None:
        return functional_call(model.decoder, overrides, arguments)
    mask = tumor_grid(tumor, F.shape[2:])
    z0 = nodes.z0.double()
    record['deep'] = dict(gamma=float(scale), unscaled_ratio=ratio(writeback, F), scaled_ratio=ratio(scale * writeback, F),
                          unscaled_ratio_tumor=ratio(writeback, F, mask), scaled_ratio_tumor=ratio(scale * writeback, F, mask),
                          tumor_fraction_of_grid=mask.mean().item(),
                          cosine_F_Fprime=functional.cosine_similarity(F.double().flatten(), fused.double().flatten(), dim=0).item())
    # 模型实际具有的有向边 [sender, receiver]；汇总边权时只统计这些边（双向链里 WT 与 ET 之间没有边）
    edges = [] if model.variant == 'no_relation' else sorted(
        [i, j] for mask in model.relation.edge_masks().values() for i, j in mask.nonzero().tolist())
    record['relation'] = dict(
        z_change_per_node=(torch.linalg.vector_norm(zK.double() - z0, dim=-1) / torch.linalg.vector_norm(z0, dim=-1))[0].tolist(),
        last_alpha=last_alpha[0].tolist(), rounds=rounds, edges=edges)
    record['decoder'] = []
    with _RecordDecoderWriteback(model.decoder, record['decoder'], tumor):
        return functional_call(model.decoder, overrides, arguments)


def compare(base, logits, label, threshold):
    hard = region_dice(logits, label, threshold)
    before, after = base.sigmoid() >= threshold, logits.sigmoid() >= threshold
    flipped, union = int((before != after).sum()), int((before | after).sum())
    return dict(dice=[r['dice'] for r in hard['organs']], mean=hard['mean_dice'],
                logit_relative_change=ratio(logits - base, base), flipped_voxels=flipped,
                flipped_fraction_of_predicted=flipped / max(union, 1))


@torch.no_grad()
def probe_case(model, image, label, threshold):
    """一例：基线记录 + 全部敲除。返回可直接写入 JSON 的字典。"""
    check_probe_model(model)
    features = model.encoder(image)
    tumor = region_targets(label)[0][0]
    stats = {}
    base = probe_tail(model, features, record=stats, tumor=tumor)
    result = dict(stats, base=compare(base, base, label, threshold))
    variants = COMMON_VARIANTS + (RELATION_VARIANTS if model.variant is None else ())
    for name, kwargs in variants:
        result[name] = compare(base, probe_tail(model, features, **kwargs), label, threshold)
    return result


def parse_run(text):
    label, sep, path = text.partition('=')
    if not sep or not label or not path:
        raise argparse.ArgumentTypeError('--run must be LABEL=RUN_DIR')
    return label, Path(path)


def load_run(label, run_dir, checkpoint_name, allow_source_change):
    run_dir = run_dir.resolve()
    run = json.loads((run_dir / 'run.json').read_text())
    identity = run['identity']
    if identity.get('mode') != 'brats_regions':
        raise ValueError(f'{label}: BraTS region run required')
    checkpoint = run_dir / checkpoint_name
    state = load_checkpoint(checkpoint, identity)  # SHA256 校验，且身份必须与 run.json 完全一致
    if state['run_id'] != run['run_id']:
        raise ValueError(f'{label}: checkpoint/run_id mismatch')
    saved = {k: v for k, v in identity['provenance']['source_hashes'].items() if k.startswith('src/')}
    current = {k: v for k, v in code_hashes().items() if k.startswith('src/')}
    differing = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
    if differing and not allow_source_change:
        raise ValueError(f'{label}: package source differs from the training commit ({len(differing)} files); '
                         'pass --allow-source-change only if those versions are documented as computing identically')
    model = BrainTumorSegmentor(BrainTumorConfig(**identity['model']))  # CPU FP32、原生卷积
    model.load_state_dict(state['model'], strict=True)
    check_probe_model(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    step = state['progress']['global_step']
    header = dict(run_dir=str(run_dir), run_id=run['run_id'], checkpoint=checkpoint_name, checkpoint_sha256=sha256(checkpoint),
                  global_step=step, epoch=state['progress'].get('epoch'), variant=model.variant,
                  training_commit=identity['provenance'].get('git'), source_matches_training=not differing,
                  differing_sources=differing)
    return dict(model=model, identity=identity, header=header, ledger=load_ledger(run_dir, step))


def load_ledger(run_dir, step):
    """该 run 在 checkpoint 那一步自己的验证逐例记录（GPU 训练时写入）；没有就返回 None。"""
    folder = run_dir / 'validation' / f'step_{step:08d}'
    if not folder.is_dir():
        return None
    ledger = {}
    for path in folder.glob('*.json'):
        row = json.loads(path.read_text())
        if 'result' in row:  # 跳过每轮的完成标记
            ledger[row['case_id']] = [r['dice'] for r in row['result']['organs']]
    return ledger


def run_probe(args):
    torch.set_num_threads(args.threads)
    labels = [label for label, _ in args.run]
    if len(set(labels)) != len(labels) or set(labels) & {'kind', 'case_id', 'shape', 'gt_voxels', 'seconds'}:
        raise ValueError('run labels must be unique and not a reserved row field')
    runs = {label: load_run(label, path, args.checkpoint_name, args.allow_source_change) for label, path in args.run}
    first = next(iter(runs.values()))['identity']
    for label, run in runs.items():
        identity = run['identity']
        for key in ('preprocessing', 'threshold'):
            if identity[key] != first[key]:
                raise ValueError(f'{label}: {key} differs between runs')
        for key in ('manifest_hash', 'split_hash'):
            if identity['data'][key] != first['data'][key]:
                raise ValueError(f'{label}: data {key} differs between runs')
    saved, split = first['data']['manifest'], first['data']['split']
    if digest(split) != first['data']['split_hash']:
        raise ValueError('saved split does not match its hash')
    output = args.output.resolve()
    source = (args.source or Path(saved['source'])).resolve()
    if output.suffix != '.jsonl' or any(output.is_relative_to(Path(r['header']['run_dir'])) for r in runs.values()) \
            or output.is_relative_to(source if source.is_dir() else source.parent):
        raise ValueError('--output must be a .jsonl file outside the runs and the source dataset')
    manifest = index_cases(source, cohort=saved['cohort'], partition=saved.get('source_partition', 'all'))
    if manifest != saved or digest(manifest) != first['data']['manifest_hash']:
        raise ValueError('source metadata differs from the runs\' manifest')
    by_id = {r['case_id']: r for r in manifest['records']}
    cases = sorted(split['dev'])
    if args.cases != 'all':
        count = int(args.cases)
        if not 0 < count <= len(cases):
            raise ValueError(f'--cases must be all or 1..{len(cases)}')
        chosen = np.random.default_rng(args.case_seed).choice(len(cases), count, replace=False)
        cases = [cases[i] for i in sorted(chosen)]
    header = dict(kind='header', schema=SCHEMA, runs={label: run['header'] for label, run in runs.items()}, cases=cases,
                  threshold=first['threshold'],
                  execution=dict(device='cpu', dtype='float32', convolution_backend='native', memory_format='contiguous',
                                 threads=args.threads, torch=str(torch.__version__)),
                  probe_commit=git_state())
    done = set()
    if output.exists():
        rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
        old = rows[0] if rows else {}
        if any(old.get(key) != header[key] for key in ('schema', 'runs', 'cases', 'threshold')):
            raise ValueError('existing output was written for other runs, checkpoints or cases; choose a new --output')
        done = {row['case_id'] for row in rows[1:]}
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            stream.write(json.dumps(header, allow_nan=False) + '\n')
    dataset = BraTSDataset(dict(manifest, records=[by_id[c] for c in cases]), **first['preprocessing'],
                           cache_dir=args.data_cache)
    for index, case in enumerate(cases):
        if case in done:
            continue
        started = time.perf_counter()
        sample = dataset[index]
        image, label = sample.image[None].float(), sample.label[None]
        row = dict(kind='case', case_id=case, shape=list(image.shape[2:]),
                   gt_voxels=[int(t.sum()) for t in region_targets(label)[0]])
        for name, run in runs.items():
            result = probe_case(run['model'], image, label, first['threshold'])
            result['ledger_dice'] = None if run['ledger'] is None else run['ledger'].get(case)
            row[name] = result
        row['seconds'] = time.perf_counter() - started
        with output.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(f'{index + 1}/{len(cases)} {case} {row["seconds"]:.0f}s', file=sys.stderr, flush=True)
    return output


def _quantiles(values):
    v = np.asarray(values, float)
    return f'median {np.median(v):.3f} [IQR {np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}]'


def _mean_delta(rows, label, variant, region=None):
    deltas = []
    for row in rows:
        base, other = row[label]['base'], row[label][variant]
        a, b = (base['mean'], other['mean']) if region is None else (base['dice'][region], other['dice'][region])
        if a is not None and b is not None:
            deltas.append(b - a)
    return np.mean(deltas) if deltas else float('nan'), deltas


def summarize(paths):
    """按 run 标签汇总（多个文件中同名标签合并，例如多个折的 E1），再逐文件拆分。返回英文文本。"""
    groups, labels = [], []
    for path in paths:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if not rows or rows[0].get('schema') != SCHEMA:
            raise ValueError(f'{path}: not a relation probe output')
        groups.append((Path(path).name, rows[1:]))
        labels += [label for label in rows[0]['runs'] if label not in labels]
    lines = []
    for title, rows in [('all files', [r for _, g in groups for r in g])] + groups:
        lines.append(f'\n#### {title} ({len(rows)} cases)')
        for label in labels:
            part = [r for r in rows if label in r]
            if not part:
                continue
            deep = [r[label]['deep'] for r in part]
            lines.append(f'[{label}] gamma {_quantiles([d["gamma"] for d in deep])}')
            lines.append(f'  deep writeback ||gamma*phi(G)||/||F||: volume {_quantiles([d["scaled_ratio"] for d in deep])}; '
                         f'tumor {_quantiles([d["scaled_ratio_tumor"] for d in deep])}')
            lines.append(f'  unscaled ||phi(G)||/||F|| {_quantiles([d["unscaled_ratio"] for d in deep])}; '
                         f'cosine(F, F\') {_quantiles([d["cosine_F_Fprime"] for d in deep])}')
            for i, scale in enumerate(('1/4', '1/2', 'full')):
                dec = [r[label]['decoder'][i] for r in part]
                lines.append(f'  decoder writeback {scale} ||eta*Phi(G)||/||E||: volume {_quantiles([d["ratio"] for d in dec])}; '
                             f'tumor {_quantiles([d["ratio_tumor"] for d in dec])}')
            rounds = [r[label]['relation']['rounds'] for r in part]
            # 旧输出没有 edges 字段，当作完全图（全部非对角边）
            edges = [[tuple(e) for e in r[label]['relation'].get('edges', [[i, j] for i in range(3) for j in range(3) if i != j])]
                     for r in part]
            for t in range(len(rounds[0])):
                alpha = [rr[t]['alpha'][i][j] for rr, ee in zip(rounds, edges) for i, j in ee]
                lines.append(f'  round {t + 1} edge weights: median {np.median(alpha):.4f}, max {max(alpha):.4f}, >0.05 {np.mean(np.asarray(alpha) > .05):.1%}; '
                             f'||message||/||z|| {_quantiles([rr[t]["message_to_z"] for rr in rounds])}; '
                             f'message share of node update {_quantiles([rr[t]["message_share_of_update"] for rr in rounds])}')
            if rounds[0]:
                lines.append(f'  ||z(K)-z(0)||/||z(0)|| {_quantiles([v for r in part for v in r[label]["relation"]["z_change_per_node"]])}')
            base = [r[label]['base'] for r in part]
            means = [b['mean'] for b in base if b['mean'] is not None]
            region_means = [np.mean([b['dice'][i] for b in base if b['dice'][i] is not None] or [np.nan]) for i in range(3)]
            lines.append(f'  baseline (CPU FP32) mean Dice {np.mean(means):.4f}; ' +
                         ', '.join(f'{n} {v:.4f}' for n, v in zip(REGION_NAMES, region_means)))
            ledger = [(b['dice'][i], r[label]['ledger_dice'][i]) for r, b in zip(part, base) if r[label]['ledger_dice']
                      for i in range(3) if b['dice'][i] is not None and r[label]['ledger_dice'][i] is not None]
            if ledger:
                diff = np.abs([a - b for a, b in ledger])
                lines.append(f'  vs training validation ledger (GPU): {len(ledger)} case-region Dice values, mean |diff| {diff.mean():.4f}, max {diff.max():.4f}')
            for variant in [v for v, _ in COMMON_VARIANTS + RELATION_VARIANTS if v in part[0][label]]:
                mean, deltas = _mean_delta(part, label, variant)
                regions = ', '.join(f'{n} {_mean_delta(part, label, variant, i)[0]:+.4f}' for i, n in enumerate(REGION_NAMES))
                flips = [r[label][variant]['flipped_fraction_of_predicted'] for r in part]
                logits = [r[label][variant]['logit_relative_change'] for r in part]
                lines.append(f'  knockout {variant:16s} mean Dice change {mean:+.4f} ({regions}); worse by >0.01 in '
                             f'{sum(d < -.01 for d in deltas)}/{len(deltas)} cases; flipped {_quantiles(flips)}; logit change {_quantiles(logits)}')
    return '\n'.join(lines).lstrip('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    probe = commands.add_parser('probe', help='run the CPU probe on validation cases')
    probe.add_argument('--run', type=parse_run, action='append', required=True, help='LABEL=RUN_DIR (repeatable)')
    probe.add_argument('--output', type=Path, required=True, help='new .jsonl, or an existing one to continue')
    probe.add_argument('--checkpoint-name', default='best-dev.ckpt')
    probe.add_argument('--source', type=Path, help='defaults to the source saved in the runs; metadata must match')
    probe.add_argument('--data-cache', type=Path, help='optional npy cache directory shared with training')
    probe.add_argument('--cases', default='all', help='all, or a number of validation cases drawn with --case-seed')
    probe.add_argument('--case-seed', type=int, default=0)
    probe.add_argument('--threads', type=int, default=4)
    probe.add_argument('--allow-source-change', action='store_true',
                       help='accept a training commit whose src/ differs (recorded in the output header)')
    report = commands.add_parser('summarize', help='summarize one or more probe outputs')
    report.add_argument('outputs', type=Path, nargs='+')
    args = parser.parse_args(argv)
    try:
        if args.command == 'probe':
            print(run_probe(args))
        else:
            print(summarize(args.outputs))
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print('relation probe stopped: ' + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

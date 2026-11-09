"""Run every fold of one experiment config; rerun the same command to resume after any interruption."""
# 用法：python scripts/run_experiment.py configs/experiments/no_relation.json
#
# 本机相关的设置（数据、划分、缓存、输出目录、设备、启动环境变量）写在 configs/machine.local.json，
# 不进 git；模板见 configs/machine.example.json。每一折用独立子进程调用 scripts/brats.py train --auto-resume：
# 已早停结束的折直接跳过，未完成的折从 last.ckpt 严格续训，第一次存档前中断的残留目录归档后重开。
# 任何一折失败即停止，不跳到下一折。
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MACHINE_KEYS = {'device', 'convolution_backend', 'cohort', 'source_partition', 'source', 'data_cache',
                'split_template', 'runs_dir', 'environment'}


def load_machine(path):
    """读取并严格校验本机配置；相对路径相对仓库根目录解析。"""
    try:
        machine = json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise ValueError(f'machine config {path} not found; copy configs/machine.example.json and edit it') from None
    if not isinstance(machine, dict) or set(machine) != MACHINE_KEYS:
        raise ValueError(f'machine config keys must be exactly {sorted(MACHINE_KEYS)}')
    for key in MACHINE_KEYS - {'environment'}:
        if not isinstance(machine[key], str) or not machine[key]:
            raise ValueError(f'machine config {key} must be a non-empty string')
    environment = machine['environment']
    if not isinstance(environment, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in environment.items()):
        raise ValueError('machine config environment must map variable names to string values')
    if '{fold}' not in machine['split_template']:
        raise ValueError('split_template must contain {fold}')
    resolve = lambda value: Path(value) if Path(value).is_absolute() else ROOT / value
    return dict(machine, source=resolve(machine['source']), data_cache=resolve(machine['data_cache']),
                runs_dir=resolve(machine['runs_dir']), split_template=str(resolve(machine['split_template'])))


def fold_command(config, machine, fold):
    """一折的完整训练命令与运行目录；运行目录名为 <cohort>_<配置文件名>_fold<k>。"""
    run_dir = machine['runs_dir'] / f"{machine['cohort']}_{config.stem}_fold{fold}"
    split = Path(machine['split_template'].format(fold=fold))
    command = [sys.executable, str(ROOT / 'scripts' / 'brats.py'), 'train', '--config', str(config),
               '--device', machine['device'], '--convolution-backend', machine['convolution_backend'],
               '--cohort', machine['cohort'], '--source-partition', machine['source_partition'],
               '--source', str(machine['source']), '--split', str(split), '--data-cache', str(machine['data_cache']),
               '--run-dir', str(run_dir), '--auto-resume']
    return command, run_dir, split


def main(argv=None, *, runner=subprocess.run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, help='experiment config, e.g. configs/experiments/main.json')
    parser.add_argument('--machine', type=Path, default=ROOT / 'configs' / 'machine.local.json')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--dry-run', action='store_true', help='print the commands without running them')
    args = parser.parse_args(argv)
    try:
        config = args.config if args.config.is_absolute() else Path.cwd() / args.config
        if not config.is_file():
            raise ValueError(f'experiment config {args.config} not found')
        if len(set(args.folds)) != len(args.folds) or any(f < 0 for f in args.folds):
            raise ValueError('--folds must be distinct non-negative integers')
        machine = load_machine(args.machine)
        plan = [fold_command(config, machine, fold) for fold in args.folds]
        missing = [str(p) for p in (machine['source'], *(split for _, _, split in plan)) if not p.exists()]
        if missing:
            raise ValueError('missing input files: ' + ', '.join(missing))
    except ValueError as exc:
        print(f'run_experiment stopped: {exc}', file=sys.stderr)
        return 2
    environment = dict(os.environ, **machine['environment'])  # 显式写在本机配置里的变量，会记录进 run identity
    for fold, (command, run_dir, _) in zip(args.folds, plan):
        print(f'[fold {fold}] {run_dir}', flush=True)
        if machine['environment']:
            print('  environment: ' + ' '.join(f'{k}={v}' for k, v in machine['environment'].items()), flush=True)
        if args.dry_run:
            print('  ' + ' '.join(command), flush=True)
            continue
        code = runner(command, cwd=ROOT, env=environment).returncode
        if code != 0:
            print(f'run_experiment stopped: fold {fold} exited with code {code}; rerun the same command to resume',
                  file=sys.stderr)
            return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

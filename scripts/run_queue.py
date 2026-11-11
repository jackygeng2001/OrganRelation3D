"""Run a queue of (config, fold) training jobs, several at a time on one GPU; rerun the same command to resume."""
# 用法：python scripts/run_queue.py configs/queues/<队列>.json
#
# 队列文件：{"workers": 2, "jobs": [{"config": "configs/experiments/x.json", "fold": 0}, ...]}，按顺序执行。
# 每个作业就是 scripts/run_experiment.py 对单折发出的同一条命令（scripts/brats.py train --auto-resume），
# 因此运行目录、严格续训与身份校验都不变；作业各自的输出写到 <runs_dir>/queue_logs/<运行目录名>.log。
# 同时运行 workers 个作业：本项目每步由大量小卷积核组成，单个训练占不满显卡，实测两个并行的总吞吐
# 约为单个的 1.4–1.5 倍（见 EXPERIMENTS.md）。作业之间互不影响：各自的随机数、checkpoint 与日志。
# 状态记在 <runs_dir>/queue_status/<队列名>.json：已完成的作业不再启动；中断（断电、Ctrl-C、重启）后
# 重新执行同一条命令，未完成的作业从各自的 last.ckpt 续训。失败的作业立即重试一次；仍失败则排到队尾，
# 等其余作业跑完再试最后一次；还失败才记为 failed（长时间无人值守时不让一个作业挡住整个队列）。
# 结束时汇总，有失败则以非零码退出；之后再执行同一条命令会重跑失败的作业。
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from run_experiment import fold_command, load_machine  # noqa: E402


def load_queue(path):
    queue = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(queue, dict) or set(queue) - {'workers', 'jobs', 'note'} or 'jobs' not in queue:
        raise ValueError('queue file must be {"workers": n, "jobs": [...]} (optional "note")')
    workers = queue.get('workers', 1)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('workers must be an integer from 1 to 4')
    jobs, seen = [], set()
    for job in queue['jobs']:
        if not isinstance(job, dict) or set(job) != {'config', 'fold'} or type(job['fold']) is not int or job['fold'] < 0:
            raise ValueError('each job must be {"config": path, "fold": non-negative integer}')
        config = Path(job['config'])
        config = config if config.is_absolute() else ROOT / config
        if not config.is_file():
            raise ValueError(f'experiment config {job["config"]} not found')
        key = (config.resolve(), job['fold'])
        if key in seen:
            raise ValueError(f'duplicate job {job}')
        seen.add(key)
        jobs.append((config, job['fold']))
    if not jobs:
        raise ValueError('queue has no jobs')
    return workers, jobs


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def say(text):
    print(f'[{now()}] {text}', flush=True)


def last_validation(run_dir):
    """只用于显示：读运行日志末尾，取最近一次验证的轮次与平均 Dice。"""
    try:
        with (run_dir / 'metrics.jsonl').open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 4_000_000))
            lines = stream.read().decode('utf-8', 'ignore').splitlines()
        for line in reversed(lines):
            if '"phase":"internal_dev"' in line.replace(' ', ''):
                row = json.loads(line)
                return f"epoch {row['epoch']} dice {row['metrics']['mean_case_dice']:.4f}"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return 'no validation yet'


def main(argv=None, *, popen=subprocess.Popen, sleep=time.sleep):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queue', type=Path, help='queue file, e.g. configs/queues/brats2018_relation_v2.json')
    parser.add_argument('--machine', type=Path, default=ROOT / 'configs' / 'machine.local.json')
    parser.add_argument('--workers', type=int, help='override the number of simultaneous jobs in the queue file')
    parser.add_argument('--heartbeat', type=int, default=1800, help='seconds between status lines')
    parser.add_argument('--dry-run', action='store_true', help='print the plan without running anything')
    args = parser.parse_args(argv)
    try:
        workers, jobs = load_queue(args.queue)
        if args.workers is not None:
            if not 1 <= args.workers <= 4:
                raise ValueError('--workers must be from 1 to 4')
            workers = args.workers
        machine = load_machine(args.machine)
        plan = []
        for config, fold in jobs:
            command, run_dir, split = fold_command(config, machine, fold)
            plan.append(dict(name=run_dir.name, command=command, run_dir=run_dir, split=split))
        if len({job['name'] for job in plan}) != len(plan):
            raise ValueError('two jobs map to the same run directory (configs with the same file name?)')
        missing = [str(p) for p in {machine['source'], *(job['split'] for job in plan)} if not p.exists()]
        if missing:
            raise ValueError('missing input files: ' + ', '.join(sorted(missing)))
    except ValueError as exc:
        print(f'run_queue stopped: {exc}', file=sys.stderr)
        return 2
    status_path = machine['runs_dir'] / 'queue_status' / f'{args.queue.stem}.json'
    log_dir = machine['runs_dir'] / 'queue_logs'
    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    if args.dry_run:
        for index, job in enumerate(plan):
            print(f"{index + 1:3d}  {status.get(job['name'], {}).get('state', 'pending'):8s} {job['name']}")
        print(f'{len(plan)} jobs, {workers} at a time; status file {status_path}')
        return 0
    status_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=1), encoding='utf-8')
        os.replace(temporary, status_path)

    environment = dict(os.environ, **machine['environment'])
    pending = [job for job in plan if status.get(job['name'], {}).get('state') != 'done']
    say(f'{len(plan)} jobs in queue, {len(plan) - len(pending)} already done, {workers} at a time; logs in {log_dir}')
    running, stopping = [], []

    def stop(signum, frame):
        stopping.append(signum)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def launch(job, attempt):
        log = (log_dir / f"{job['name']}.log").open('ab')
        log.write(f'\n===== {now()} run_queue attempt {attempt} =====\n'.encode())
        log.flush()
        process = popen(job['command'], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=os.name == 'posix')
        entry = status.setdefault(job['name'], {})
        entry.update(state='running', attempts=entry.get('attempts', 0) + 1, started=now())
        save()
        say(f"start  {job['name']} (attempt {attempt})")
        running.append(dict(job=job, process=process, log=log, attempt=attempt))

    last_beat = time.monotonic()
    while (pending or running) and not stopping:
        while pending and len(running) < workers:
            job = pending.pop(0)
            launch(job, 3 if job.get('last_chance') else 1)
        sleep(5)
        for item in list(running):
            code = item['process'].poll()
            if code is None:
                continue
            running.remove(item)
            item['log'].close()
            job, entry = item['job'], status[item['job']['name']]
            if code == 0:
                entry.update(state='done', finished=now(), result=last_validation(job['run_dir']))
                say(f"done   {job['name']} ({entry['result']})")
            elif item['attempt'] == 1 and not stopping:
                say(f"failed {job['name']} with code {code}; retrying once")
                launch(job, 2)
                continue
            elif item['attempt'] == 2 and not stopping:
                # 两次都失败：先让出位置，排到队尾等其余作业跑完再试最后一次（例如两个大作业同时占显存导致的失败）
                entry.update(state='failed', finished=now(), returncode=code)
                job['last_chance'] = True
                pending.append(job)
                say(f"failed {job['name']} twice (code {code}); moved to the end of the queue for a last attempt")
            else:
                entry.update(state='failed', finished=now(), returncode=code)
                say(f"FAILED {job['name']} with code {code}; see {log_dir / (job['name'] + '.log')}; continuing")
            save()
        if time.monotonic() - last_beat >= args.heartbeat:
            last_beat = time.monotonic()
            done = sum(1 for job in plan if status.get(job['name'], {}).get('state') == 'done')
            say(f'{done}/{len(plan)} done; running: '
                + '; '.join(f"{item['job']['name']} ({last_validation(item['job']['run_dir'])})" for item in running))
    if stopping:
        say('stop requested: interrupting running jobs (they resume from their last checkpoint next time)')
        for item in running:
            try:
                if hasattr(os, 'killpg'):
                    os.killpg(item['process'].pid, signal.SIGINT)
                else:  # Windows：没有进程组信号，直接结束子进程（训练每 25 步存档，续跑时从存档接着跑）
                    item['process'].terminate()
            except (ProcessLookupError, PermissionError):
                pass
        for item in running:
            try:
                item['process'].wait(timeout=120)
            except subprocess.TimeoutExpired:
                item['process'].kill()
            item['log'].close()
            status[item['job']['name']].update(state='interrupted', finished=now())
        save()
        return 130
    failed = [job['name'] for job in plan if status.get(job['name'], {}).get('state') == 'failed']
    say(f'queue finished: {len(plan) - len(failed)} done, {len(failed)} failed' + (': ' + ', '.join(failed) if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())

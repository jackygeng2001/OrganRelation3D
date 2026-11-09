"""Best-effort scalar observer. JSONL/checkpoints remain the source of truth."""
import json
from pathlib import Path
import sys
import time


def scalar_values(row):
    """Translate scalar log rows only; never touch live model tensors."""
    from .brain_tumor import region_scalars  # 所有 BraTS 日志行都由三区域映射处理
    return region_scalars(row)


class TensorBoardObserver:
    def __init__(self, run_dir, *, enabled=True):
        self.directory = Path(run_dir) / 'tensorboard'
        self.enabled = enabled
        self.writer = None
        self.failed = False

    def _failure(self, exc):
        self.failed = True
        self.enabled = False
        writer, self.writer = self.writer, None
        try:
            print(f'[TensorBoard disabled] {type(exc).__name__}: {exc}. '
                  'Training continues; metrics.jsonl/checkpoint remain authoritative. '
                  'TensorBoard curves may be incomplete.', file=sys.stderr, flush=True)
        except Exception:
            pass
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

    def start(self, committed_step, log_path):
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            # EventAccumulator reads filenames lexicographically. Default names
            # use SECOND-resolution time followed by host/PID/unpadded counter;
            # .10 can sort before .9 on a quick restart. Advance at most one
            # second before creating a new file, so purge is always read last.
            existing = list(self.directory.glob('events.out.tfevents.*'))
            if existing:
                latest = max(int(path.name.split('.')[3]) for path in existing)
                delay = latest + 1 - time.time()
                if delay > 1.1:
                    raise OSError('event timestamp is in the future; refusing out-of-order restart')
                if delay > 0:
                    time.sleep(delay + .01)
                if int(time.time()) <= latest:
                    raise OSError('clock moved backwards; refusing out-of-order restart')
            # Purge K itself and replay only committed rows at K, as validation
            # at that SAME step may have crashed before its checkpoint commit.
            self.writer = SummaryWriter(log_dir=str(self.directory),
                                        purge_step=committed_step if committed_step else None)
            if committed_step:
                with Path(log_path).open(encoding='utf-8') as stream:
                    for line in stream:
                        row = json.loads(line)
                        if row['global_step'] == committed_step:
                            self._record(row)
                self.writer.flush()
        except Exception as exc:
            self._failure(exc)

    def _record(self, row):
        for tag, value in scalar_values(row).items():
            self.writer.add_scalar(tag, value, global_step=row['global_step'])

    def record(self, row):
        if self.writer is not None:
            try:
                self._record(row)
            except Exception as exc:
                self._failure(exc)

    def flush(self):
        if self.writer is not None:
            try:
                self.writer.flush()
            except Exception as exc:
                self._failure(exc)

    def close(self):
        self.flush()
        if self.writer is not None:
            writer, self.writer = self.writer, None
            try:
                writer.close()
            except Exception as exc:
                self._failure(exc)

"""Atomic state and scalar logs. Load only checkpoints produced by trusted runs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, canonical(value) + b'\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),
                accelerator=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if device.type == 'cuda':
        if state['accelerator'] is None:
            raise ValueError('checkpoint has no accelerator RNG state')
        torch.cuda.set_rng_state_all(state['accelerator'])
    elif state['accelerator'] is not None:
        raise ValueError('cannot resume accelerator run on CPU')


MAGIC = b'OR3DCK1\n'


def save_checkpoint(path, state):
    stream = io.BytesIO()
    torch.save(state, stream)
    payload = stream.getvalue()
    atomic_bytes(path, MAGIC + hashlib.sha256(payload).digest() + payload)


def load_checkpoint(path, identity, *, extension=False):
    raw = Path(path).read_bytes()
    start = len(MAGIC)
    if not raw.startswith(MAGIC) or len(raw) <= start + 32:
        raise ValueError('incomplete or unsupported checkpoint')
    payload = raw[start + 32:]
    if hashlib.sha256(payload).digest() != raw[start:start + 32]:
        raise ValueError('corrupt checkpoint checksum')
    # This contains Python/NumPy RNG state; never load untrusted checkpoints.
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    required = {'schema_version', 'identity', 'run_id', 'model', 'optimizer', 'scheduler',
                'progress', 'rng', 'sampler_generator', 'loader_generator', 'log'}
    if not isinstance(state, dict) or not required <= state.keys() or state['schema_version'] != 1:
        raise ValueError('incomplete checkpoint state')
    if state['identity'] != identity:
        if extension:
            from .extension import check_extension_identity
            check_extension_identity(state['identity'], identity)
        else:
            keys = sorted(k for k in set(state['identity']) | set(identity)
                          if state['identity'].get(k) != identity.get(k))
            raise ValueError('resume identity mismatch: ' + ', '.join(keys))
    if state['scheduler'] is not None:
        raise ValueError('scheduler is not implemented in training v1')
    return state


# The parts of a run identity that define the experiment (not code version or machine).
SCIENTIFIC_IDENTITY_KEYS = ('mode', 'model', 'loss', 'threshold', 'preprocessing', 'training', 'optimizer', 'data')


def prepare_auto_resume(run_dir, identity, *, now=None):
    """One command that starts, resumes, or restarts an experiment run.

    Returns (resume_path, archived_path). last.ckpt present: strict resume from
    it (all usual checks still apply). Missing or empty directory: new run.
    Otherwise the directory must be the residue of an interruption BEFORE the
    first checkpoint: no checkpoint file anywhere and a run.json whose
    scientific identity equals this one. Nothing in it can be resumed, so it is
    renamed (never deleted) to <name>.aborted-<UTC time> and the run starts
    again from step 0 with the same seed. Anything else is refused untouched.
    """
    run_dir = Path(run_dir)
    last = run_dir / 'last.ckpt'
    if last.is_file():
        return last, None
    if not run_dir.exists() or (run_dir.is_dir() and not any(run_dir.iterdir())):
        return None, None
    if not run_dir.is_dir():
        raise ValueError(f'{run_dir} is not a directory')
    if any(run_dir.rglob('*.ckpt')):
        raise ValueError(f'{run_dir} has checkpoints but no last.ckpt; not touched, inspect it by hand')
    try:
        recorded = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))['identity']
        if not isinstance(recorded, dict):
            raise TypeError('identity is not a mapping')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'{run_dir} is not an interrupted run (no readable run.json); not touched') from exc
    differing = [key for key in SCIENTIFIC_IDENTITY_KEYS if recorded.get(key) != identity.get(key)]
    if differing:
        raise ValueError(f'{run_dir} belongs to a different experiment ({", ".join(differing)}); not touched')
    stamp = (now or datetime.now(timezone.utc)).strftime('%Y%m%dT%H%M%SZ')
    archived = run_dir.with_name(f'{run_dir.name}.aborted-{stamp}')
    if archived.exists():
        raise ValueError(f'{archived} already exists; not touched')
    os.rename(run_dir, archived)
    return None, archived


class ScalarLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, row):
        data = canonical(row) + b'\n'  # Tensor / NaN rejected, never retained.
        with self.path.open('ab') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    def position(self, step):
        raw = self.path.read_bytes()
        return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), global_step=step)

    def recover(self, committed):
        raw = self.path.read_bytes()
        end = committed['bytes']
        if type(end) is not int or end < 0 or len(raw) < end:
            raise ValueError('log is shorter than committed checkpoint position')
        if hashlib.sha256(raw[:end]).hexdigest() != committed['sha256']:
            raise ValueError('committed log checksum mismatch')
        if len(raw) > end:
            atomic_bytes(self.path, raw[:end])
            print(f'Resume: discarded {len(raw)-end} uncommitted log bytes.', flush=True)

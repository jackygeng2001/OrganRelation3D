"""Atomic per-case results and ledger, independent of training checkpoint state."""
import hashlib
import json
from pathlib import Path

from ..training.state import atomic_json, digest


class CaseLedger:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'ledger.json'
        self.identity = identity
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding='utf-8'))
            if self.state.get('schema_version') != 1 or self.state.get('identity') != identity:
                raise ValueError('evaluation ledger identity mismatch')
        else:
            self.state = dict(schema_version=1, identity=identity, completed={})

    def _path(self, case_id):
        # Case IDs need not be filesystem-safe; never interpolate them as paths.
        return self.directory / (digest(case_id) + '.json')

    def read(self, case_id, data_identity):
        entry = self.state['completed'].get(case_id)
        path = self._path(case_id)
        if (not entry or entry.get('status') != 'completed' or entry.get('output_path') != path.name
                or entry.get('data_identity') != data_identity or not path.exists()):
            return None
        raw = path.read_bytes()
        if entry.get('sha256') != hashlib.sha256(raw).hexdigest():
            return None
        try:
            value = json.loads(raw)
            if (value['identity'] != self.identity or value['case_id'] != case_id
                    or value['data_identity'] != data_identity):
                return None
            return value['result']
        except (ValueError, KeyError, TypeError):
            return None

    def commit(self, case_id, data_identity, result):
        path = self._path(case_id)
        record = dict(identity=self.identity, case_id=case_id, data_identity=data_identity, result=result)
        atomic_json(path, record)
        raw = path.read_bytes()
        if json.loads(raw) != record:
            raise ValueError('result integrity check failed')
        completed = dict(self.state['completed'])
        completed[case_id] = dict(status='completed', data_identity=data_identity,
                                 output_path=path.name, sha256=hashlib.sha256(raw).hexdigest())
        committed = dict(self.state, completed=completed)
        atomic_json(self.path, committed)
        self.state = committed
        # A crash between result and ledger writes leaves an uncommitted case;
        # it is recomputed, never blindly skipped based on file existence.

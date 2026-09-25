"""Checkout provenance for repository scripts; no tensor or imaging dependencies.

These helpers describe the source checkout (including editable installs).
Report/config discovery is a repository workflow, not a wheel resource API.
"""
from pathlib import Path
import hashlib
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict:
    def run(*args):
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *args],
                                       stderr=subprocess.DEVNULL, text=True,
                                       encoding="utf-8").strip()
    try:
        if Path(run("rev-parse", "--show-toplevel")).resolve() != PROJECT_ROOT:
            return {"commit": None, "reason": "not an independent project repository"}
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "reason": "no available project commit"}


def code_hashes() -> dict:
    files = sorted([*PROJECT_ROOT.glob("src/**/*.py"), *PROJECT_ROOT.glob("scripts/*.py")])
    return {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p) for p in files}

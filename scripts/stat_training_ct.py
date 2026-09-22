"""Run from a checkout without installation; all data paths are CLI inputs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organ_relation.ct_stats import main

if __name__ == "__main__":
    raise SystemExit(main())

"""Estimate candidate full-FOV grids from cached metadata; no installation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from organ_relation.candidate_estimates import main

if __name__ == "__main__":
    raise SystemExit(main())

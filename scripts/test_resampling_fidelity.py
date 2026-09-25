"""Sequential CPU pilot, isolated environment; no training or network imports."""
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from organ_relation.data.fidelity_pilot import main
if __name__=='__main__':raise SystemExit(main())

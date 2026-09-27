"""Run a declared CPU test suite; missing dependencies or skipped tests fail."""
import argparse
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    'tensor': ('backbone', 'coarse_nodes', 'dynamic_relation', 'node_to_space',
               'segmentor', 'joint_loss'),
    'data': ('metadata', 'candidate_estimates', 'fidelity'),
    'fullscan': ('full_scan', 'full_scan_probe'),
    'monai': ('monai_reference',),
    'training': ('training', 'monitoring', 'backbone_only', 'balanced_ce', 'foreground_macro_ce'),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=(*SUITES, 'metadata', 'all'), required=True)
    args = parser.parse_args(argv)
    # Do not let a new test file silently disappear from the declared suites.
    declared = {f'test_{name}.py' for names in SUITES.values() for name in names}
    discovered = {p.name for p in (ROOT / 'tests').glob('test_*.py')}
    if declared != discovered:
        parser.error(f'update SUITES to match test files: {sorted(declared ^ discovered)}')
    names = (sum(SUITES.values(), ()) if args.suite == 'all' else
             SUITES['data'][:2] if args.suite == 'metadata' else SUITES[args.suite])
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for name in names:
        suite.addTests(loader.discover(str(ROOT / 'tests'), pattern=f'test_{name}.py'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print('Incomplete validation: skipped tests are not a pass. Use the matching CPU environment.',
              file=sys.stderr)
    return 0 if result.wasSuccessful() and not result.skipped and result.testsRun else 1


if __name__ == '__main__':
    raise SystemExit(main())

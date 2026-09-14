import flopscope
import sys
from unittest import mock
sys.path.append('local_sim')
import p2_compact_fullrank_precision_e14 as probe
original = probe.independent_wick
def inject(*args, **kwargs):
    result = original(*args, **kwargs)
    result[(4, 1)] = result[(4, 1)].copy()
    result[(4, 1)][0] = float('nan')
    return result
with mock.patch.object(probe, 'independent_wick', inject), mock.patch('sys.argv', ['e14', '--out', 'experiments/2026-09-14_e14/b4_f64_nan_mutation.json']):
    try:
        probe.main()
    except SystemExit as exc:
        assert exc.code == 1

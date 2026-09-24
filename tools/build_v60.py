"""Generate `estimator_k3_v60_adage.py`: v55 with PER-LINEAGE ADAPTIVE AGE.

    python tools/build_v60.py [--verify]

HANDOFF 2026-09-24 Step 3 / response §4. v55 folds the oldest exact lineage into
the shared basis `Q` at a FIXED age (`_SHARED_AGE = 5` exact lineages). This
replaces that with a tolerance: fold when the oldest exact lineage's carried Gram
energy has fallen below `tau` times the YOUNGEST exact lineage's, with a floor of
3 exact lineages and a ceiling of 6. Sweeping `tau` moves the mean age, and the
cost becomes PER-NET variable -- which is the property the leaders' telemetry
shows (+/-6% per-MLP FLOPs) and ours has never had.

Why energy is the right statistic here: the N1 probe (RESULTS.md, 2026-09-23)
measured Spearman(carried energy, downstream sensitivity ||d mu_L||^2) = 0.994.
Energy ranks a lineage's contribution almost perfectly. N1's negative conclusion
was about choosing WHICH directions to keep at a fixed rank; this chooses WHEN to
fold, which that result does not bound.

THE COST OF THIS, AND IT IS NOT FLOPS
-------------------------------------
The decision compares two flopscope scalars, so it **materialises a scalar** --
one `__bool__` per layer where the gate is live. Every other line in this file
deliberately avoids that. `local_sim/p2_weff.py` records why: a forum-reported
deployment invariant says a scalar materialisation inside `predict` can force a
synchronisation and eat the 0.4 s residual allowance even when the counted
mathematics is unchanged, and "the requirement is zero, and zero is counted".
Every graded row we hold reports 0 materialisations.

Measured locally: 16 comparisons cost 32 FLOPs, 32 op calls (+0.14% on 22,978)
and 16 `__bool__` calls. The op cost is irrelevant. The SYNC cost is the risk,
and it cannot be measured locally, because the grader's arithmetic runs server
side -- tr64's report carries `flopscope_server_comms_wall_s` (median 0.78 s)
alongside its kernel and overhead, so a forced round trip is a real thing there
and an invisible one here.

**Therefore: this file is a PROBE first.** Accuracy and per-net FLOPs decide
whether the lever clears its 2% kill. Only if it does does the materialisation
question become a gate, and then it needs a server reading, not an argument --
and the failure mode is the residual cliff, which FAILS the MLP (~0.9 against
5e-9), so it would be bought with a canary and not assumed.

`_ADAPT_TAU = None` restores fixed-age behaviour exactly; `--verify` asserts the
generated file is then bit-identical to v55.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def _rel(p) -> str:
    """`p` relative to the repo root for LOGGING, never raising.

    `Path.relative_to` raises when the path is not under `ROOT`, and under
    worker-q `ROOT` is the job's snapshot directory while an output path is
    deliberately absolute and outside it (`.gpuq.toml`: "Always give heavy jobs
    an ABSOLUTE output path"). That combination failed jobs 2090 and 2091 on a
    print statement AFTER all their work was written -- a cosmetic line that
    cost two exit-code-1 job reports.
    """
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)

BASE = ROOT / "estimator_k3_v55_p2r4.py"
OUT = ROOT / "estimator_k3_v60_adage.py"

# 1. the tower needs one more slot to carry the per-layer decision
_A_SLOTS = '''                 # STAGE 3: the consumed-query basis's two stashes
                 "_qU", "_qold", "F", "R3", "R4", "w3b", "_p3")
'''
_I_SLOTS = '''                 # STAGE 3: the consumed-query basis's two stashes
                 "_qU", "_qold", "F", "R3", "R4", "w3b", "_p3",
                 # STEP 3 (v60): this layer's fold decision, set by
                 # `_shared_contract_lean` and read by `_shared_join_lean`, so
                 # the two cannot disagree and the scalar is materialised ONCE.
                 "_dojoin")
'''

_A_INIT = "        self._p3 = None\n"
_I_INIT = "        self._p3 = None\n        self._dojoin = None\n"

# 1b. `_wick_tower` builds a NEW tower between the contract (which sets the
#     decision) and the join (which reads it), and carries `_wts` across by
#     hand. The decision has to travel the same way or the join never sees it --
#     which is exactly how the first build of this file silently stopped folding
#     at all, and why `--verify` exists.
_A_CARRY = "    out._wts = tw._wts\n"
_I_CARRY = "    out._wts = tw._wts\n    out._dojoin = tw._dojoin\n"

# 2. the knobs and the gate, inserted before the lean contract
_A_GATE = "def _shared_contract_lean(tw, LA, LP, A_L, M):\n"

_I_GATE = '''#: STEP 3 (v60): per-lineage adaptive age. `None` = v55's fixed `_SHARED_AGE`
#: and this file is then bit-identical to v55. A float is the tolerance `tau`:
#: fold the oldest exact lineage once its carried Gram energy falls below
#: `tau` x the youngest's. Larger tau folds sooner (cheaper, less accurate).
_ADAPT_TAU = None
#: Never fold below this many exact lineages, whatever the energies say.
_ADAPT_FLOOR = 3
#: Always fold at or above this many, whatever the energies say. This is what
#: keeps the cost BOUNDED -- without it a net whose energies never decay would
#: carry every lineage exactly and blow the budget.
_ADAPT_CEIL = 6


def _adapt_gate(tw, LA, LP, n, c):
    """Fold the oldest exact lineage at this layer? `True`/`False`.

    With `_ADAPT_TAU is None` this is exactly v55's fixed-age test and costs
    nothing. Otherwise it compares the oldest exact lineage's carried Gram
    energy to the youngest's, which the range finder's own column sums already
    express -- two extra reductions, `2 n^2` FLOPs.

    THE ONE DATA-DEPENDENT BRANCH IN THIS FILE. It materialises a scalar, which
    every other line here avoids on purpose; see this file's generator
    (`tools/build_v60.py`) for why that is a risk that must be bought with a
    server reading rather than argued. The floor and ceiling are pure Python on
    `tw.k`/`tw.ko`, so the branch is only reached in the band between them.
    """
    ex = tw.k - tw.ko
    if _ADAPT_TAU is None:
        return ex >= _SHARED_AGE
    if ex < int(_ADAPT_FLOOR):
        return False
    if ex >= int(_ADAPT_CEIL):
        return True
    e_old = ((LA[:, c:c + n] * LA[:, c:c + n]).sum()
             + (LP[:, c:c + n] * LP[:, c:c + n]).sum())
    e_new = ((LA[:, -n:] * LA[:, -n:]).sum()
             + (LP[:, -n:] * LP[:, -n:]).sum())
    return bool(e_old < float(_ADAPT_TAU) * e_new)


def _shared_contract_lean(tw, LA, LP, A_L, M):
'''

# 3. the LEAN contract's stash gate becomes the decision. The same three lines
#    also appear in the non-lean `_shared_contract`, so the anchor carries the
#    lean function's trailing `_qU` comment to stay unique. Only the lean path is
#    live in v55 (`_P2_SEPARABLE = True`); the non-lean path keeps fixed age and
#    is left alone deliberately, so a fallback cannot silently go adaptive.
_A_STASH = '''    if _SHARED_ON and _shared_rank(n) and (tw.k - tw.ko) >= _SHARED_AGE:
        tw._wts = ((LA[:, c:c + n] * LA[:, c:c + n]).sum(axis=0),
                   (LP[:, c:c + n] * LP[:, c:c + n]).sum(axis=0))
        # QUERY BASIS: the joiner's own frozen left factors, kept as VIEWS of
        # arrays this layer already holds. No product, no copy.
        tw._qU = (LA[:, c:c + n], LP[:, c:c + n])
'''
_I_STASH = '''    if _SHARED_ON and _shared_rank(n):
        tw._dojoin = _adapt_gate(tw, LA, LP, n, c)
    if _SHARED_ON and _shared_rank(n) and tw._dojoin:
        tw._wts = ((LA[:, c:c + n] * LA[:, c:c + n]).sum(axis=0),
                   (LP[:, c:c + n] * LP[:, c:c + n]).sum(axis=0))
        # QUERY BASIS: the joiner's own frozen left factors, kept as VIEWS of
        # arrays this layer already holds. No product, no copy.
        tw._qU = (LA[:, c:c + n], LP[:, c:c + n])
'''

# 4. the join reads the decision instead of re-testing the age
_A_JOIN = '''    if not r or (tw.k - tw.ko) < _SHARED_AGE or tw._wts is None:
        return tw
    if _QUERY_BASIS and tw._qU is None:
        return tw
'''
_I_JOIN = '''    # `_dojoin is None` means `_shared_contract_lean` never ran for this layer,
    # so the conservative answer is "do not fold" -- same as v55's behaviour
    # when the age test failed.
    if not r or not tw._dojoin or tw._wts is None:
        return tw
    if _QUERY_BASIS and tw._qU is None:
        return tw
'''


def build() -> str:
    src = BASE.read_text(encoding="utf-8")
    pairs = [("__slots__", _A_SLOTS, _I_SLOTS), ("__init__", _A_INIT, _I_INIT),
             ("wick_tower carry", _A_CARRY, _I_CARRY),
             ("gate block", _A_GATE, _I_GATE), ("stash", _A_STASH, _I_STASH),
             ("join", _A_JOIN, _I_JOIN)]
    for name, a, _b in pairs:
        if src.count(a) != 1:
            raise SystemExit(f"build_v60: anchor {name!r} occurs {src.count(a)} "
                             "times, expected 1")
    for _name, a, b in pairs:
        src = src.replace(a, b, 1)
    return src


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(build(), encoding="utf-8", newline="\n")
    print(f"[build_v60] -> {_rel(out)}")
    if a.verify:
        r = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "_parity.py"),
             str(BASE), str(out), str(a.n)],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        print(r.stdout.strip())
        if r.returncode != 0:
            print(r.stderr[-2000:])
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

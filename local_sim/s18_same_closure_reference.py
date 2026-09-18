"""Reproducible same-closure C10 storage references; label-free tiny checks.

The generated files are OFFLINE DIAGNOSTICS. Their normal ``predict`` budget
guard remains intact. ``offline_predict`` invokes that same instrumented path
with an explicitly large diagnostic budget and refuses a capped/fallback run.
No grader wall-time or budget loophole is used.

    uv run python local_sim/s18_same_closure_reference.py --self-test
    uv run python local_sim/s18_same_closure_reference.py --emit-dir DIR
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# This lab file lives beside the local_sim/flopscope shim. The reference must
# meter with the installed official package, not accidentally import that shim.
sys.path[:] = [str(ROOT)] + [p for p in sys.path if Path(p or ".").resolve() != ROOT / "local_sim"]
CONTROL = ROOT / "estimator_k3_v43_complete_c10.py"
CONTROL_SHA = "fd8d497db6dfbc3affcf579119c7115c3aee4d190a3e92fec4d85c366152d1bf"
SWITCHES = {
    "unshared": (False, 32),
    "full_p2": (True, 0),
    "unshared_full_p2": (False, 0),
}


def build(name: str) -> str:
    if name not in SWITCHES:
        raise ValueError(f"unknown reference {name!r}")
    raw = CONTROL.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONTROL_SHA:
        raise RuntimeError("frozen C10 source hash changed")
    shared, p2_rank = SWITCHES[name]
    src = raw.decode("utf-8")
    edits = {
        "_SHARED_ON = True\n": f"_SHARED_ON = {shared}\n",
        "_P2_RANK =32\n": f"_P2_RANK ={p2_rank}\n",
    }
    for old, new in edits.items():
        if src.count(old) != 1:
            raise RuntimeError(f"expected one exact switch {old!r}")
        src = src.replace(old, new, 1)
    # No source-line preface: every other byte must match the control.
    return src


def load(name: str):
    """Execute exact generated source in a private module, without a temp file."""
    import types

    module = types.ModuleType("s18_reference_" + name)
    module.__file__ = str(CONTROL)
    exec(compile(build(name), module.__file__, "exec"), module.__dict__)
    return module


def offline_predict(mod, mlp, *, diagnostic_budget: int = 2**55, estimator=None):
    """Count *all* math under flopscope; refuse silent generation caps.

    Caller owns a ``flopscope.BudgetContext`` with this same high budget and
    records its FLOPs/calls. This function does not create uninstrumented arrays.
    The large budget is for a nondeployable reference only.
    """
    if diagnostic_budget <= 2**41:
        raise ValueError("diagnostic budget must exceed official limit")
    est = estimator if estimator is not None else mod.Estimator()
    out = est.predict(mlp, diagnostic_budget)
    rec = est.last_run
    if rec["mode"] != "full" or rec["generations"] is not None:
        raise RuntimeError(f"reference was capped/fell back: {rec}")
    return out


def self_test() -> None:
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench import MLP

    raw = CONTROL.read_text(encoding="utf-8")
    try:
        build("not_a_variant")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted unknown variant")
    global CONTROL_SHA
    saved_sha = CONTROL_SHA
    CONTROL_SHA = "0" * 64
    try:
        try:
            build("unshared")
        except RuntimeError as exc:
            assert "hash changed" in str(exc)
        else:
            raise AssertionError("accepted changed frozen-source identity")
    finally:
        CONTROL_SHA = saved_sha
    for name, (shared, rank) in SWITCHES.items():
        generated = build(name)
        back = generated.replace(f"_SHARED_ON = {shared}\n", "_SHARED_ON = True\n", 1)
        back = back.replace(f"_P2_RANK ={rank}\n", "_P2_RANK =32\n", 1)
        assert back == raw, name
        mod = load(name)
        assert mod._SHARED_ON is shared and mod._P2_RANK == rank
        assert mod._CLOSURE_SCOPE == "all" and mod._P3_ON is False
        assert mod._HUB_ROUTES == ("i", "ii") and mod._P4_FINAL_VAR
        assert (mod._OWN_THETA_FAM, mod._OWN_S_LAM, mod._OWN_T_LAM,
                mod._OWN_S_FEED) == (0.625, 1.225, -0.1, 1.0)

    # Source-specific full P2 orientation and both D21 directions at birth.
    mod = load("full_p2")
    rng = np.random.default_rng(78125)  # synthetic, not a decision-block seed
    n = 7
    R = rng.normal(size=(n, n)).astype("float32")
    np.fill_diagonal(R, 0)
    with flops.BudgetContext(flop_budget=2**40, quiet=True):
        Z, L = mod._p2_compress(fnp.asarray(R))
        got = np.asarray(Z @ L.T)
    np.testing.assert_allclose(got, 3 * R.T, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.diag(got), np.zeros(n), atol=2e-5)
    for i in range(n):
        for j in range(n):
            np.testing.assert_allclose(got[i, j], 3 * R[j, i], atol=2e-5)
            np.testing.assert_allclose(got[j, i], 3 * R[i, j], atol=2e-5)

    # A short no-truth recurrence proves the unshared policy keeps ko=0 and
    # that the high-budget route does not enter the normal fallback.
    n, depth = 8, 6
    weights = [fnp.asarray(rng.normal(0, n**-0.5, (n, n)), dtype=fnp.float64)
               for _ in range(depth)]
    mlp = MLP(weights=weights, width=n, depth=depth, seed=0)
    unshared = load("unshared")
    try:
        offline_predict(unshared, mlp, diagnostic_budget=2**41)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted official budget for diagnostic")
    original_contract = unshared._shared_contract_lean
    old_tiers = []

    def checked_contract(tw, *args):
        old_tiers.append(tw.ko)
        return original_contract(tw, *args)

    unshared._shared_contract_lean = checked_contract
    with flops.BudgetContext(flop_budget=2**55, quiet=True) as ctx:
        out = offline_predict(unshared, mlp)
    assert float(ctx.flops_used) > 0
    assert np.asarray(out).shape == (depth, n)
    assert np.isfinite(np.asarray(out)).all()
    assert old_tiers and all(k == 0 for k in old_tiers)
    print("PASS: exact source switches, full-P2 transpose/orientations, no-share continuation, uncapped instrumented path")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--emit-dir", type=Path)
    args = ap.parse_args()
    if not args.self_test and args.emit_dir is None:
        ap.error("choose --self-test or --emit-dir")
    if args.self_test:
        self_test()
    if args.emit_dir is not None:
        args.emit_dir.mkdir(parents=True, exist_ok=True)
        for name in SWITCHES:
            path = args.emit_dir / f"estimator_k3_s18_reference_{name}.py"
            path.write_text(build(name), encoding="utf-8", newline="\n")
            print(path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()

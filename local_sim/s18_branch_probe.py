"""Label-free, same-incoming-state C10 storage-policy continuation probe.

The tiny scaled rank exercises the join (the production rank 256 cannot join
at width 8). This is a policy/instrument check, not an accuracy comparison.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:] = [str(ROOT)] + [p for p in sys.path if Path(p or ".").resolve() != HERE]

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

sys.path.append(str(HERE))
from mint_provenance import canonical_mint
from lifecycle_eval import build_weights
from s18_reference_meter import _assert_ledger_conserved
from s18_same_closure_reference import CONTROL, CONTROL_SHA, SWITCHES, load

GENERATOR_SHA = "874e284155ed79a4be274f4f3dc2c6ddf6a7c2da5d3f12c210913a9bfdf91f54"
BUDGET = 2**55
SEED = 19_990_919  # synthetic label-free probe; no decision block


def _identity():
    if hashlib.sha256(CONTROL.read_bytes()).hexdigest() != CONTROL_SHA:
        raise RuntimeError("frozen C10 source changed")
    if hashlib.sha256((HERE / "s18_same_closure_reference.py").read_bytes()).hexdigest() != GENERATOR_SHA:
        raise RuntimeError("frozen reference generator changed")


def _control():
    spec = importlib.util.spec_from_file_location("s18_branch_c10", CONTROL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arrays(state):
    for value in state:
        if value is None or isinstance(value, (float, int)):
            continue
        if hasattr(value, "__slots__"):
            for name in value.__slots__:
                item = getattr(value, name)
                if item is not None and hasattr(item, "shape"):
                    yield name, item
        elif hasattr(value, "shape"):
            yield "state", value


def state_hash(state):
    h = hashlib.sha256()
    for index, value in enumerate(state):
        if value is None or isinstance(value, (float, int)):
            h.update(repr((index, value)).encode())
        elif hasattr(value, "__slots__"):
            for name in value.__slots__:
                item = getattr(value, name)
                if item is None or isinstance(item, (float, int)):
                    h.update(repr((index, name, item)).encode())
    for name, arr in _arrays(state):
        a = np.asarray(arr)
        if not np.isfinite(a).all():
            raise RuntimeError(f"nonfinite incoming state {name}")
        h.update(name.encode())
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _ledger(ctx):
    buckets = ctx.summary_dict(by_namespace=True).get("by_namespace") or {}
    operations = {}
    for bucket in buckets.values():
        for op, rec in (bucket.get("operations") or {}).items():
            item = operations.setdefault(op, {"flops": 0, "calls": 0})
            item["flops"] += int(rec.get("flop_cost", 0))
            item["calls"] += int(rec.get("calls", 0))
    calls = _assert_ledger_conserved(int(ctx.flops_used), operations)
    return {"flops": int(ctx.flops_used), "calls": calls, "operations": operations}


def run(*, width=33, depth=8, branch_layer=4, shared_rank=4):
    _identity()
    if width < 5 or depth < 6 or not 0 < shared_rank < width:
        raise ValueError("invalid tiny join shape")
    if branch_layer not in (0, 4) or branch_layer >= depth - 1:
        raise ValueError("join branches at layer 4; P2 branches before first birth")
    flops.configure(symmetry_warnings=False)
    weights_np, _, _ = canonical_mint(width, depth, SEED)
    weights = build_weights(weights_np, "f64_readonly", fnp)
    mods = {"control": _control(), **{name: load(name) for name in SWITCHES}}
    for mod in mods.values():
        mod._SHARED_RANK = shared_rank
        mod._SHARED_FORCE = False
    ctl = mods["control"]
    est = ctl.Estimator()
    table = ctl._tables()
    dtype = fnp.float32
    state = (fnp.zeros(width, dtype=dtype), fnp.eye(width, dtype=dtype),
             None, 0.0, fnp.zeros(width, dtype=dtype))
    with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as prefix_ctx:
        const = est._constants(width, dtype)
        cast = [x.astype(dtype) for x in weights]
        for layer in range(branch_layer):
            state = ctl._layer(cast[layer], state, table, dtype, const, 2.0,
                               last_layer=False, gens=None, layer_index=layer,
                               depth=depth)
            state_hash(state)
            mu = fnp.where(fnp.isfinite(state[0]), state[0], dtype(0.0))
            state = (mu, state[1], state[2], state[3], state[4])
    incoming = state_hash(state)
    prefix = _ledger(prefix_ctx)
    if branch_layer == 4 and (state[2] is None or state[2].k != 4 or state[2].ko != 0):
        raise RuntimeError("prefix did not reach expected four unjoined lineages")

    arms = {}
    names = ("control", "unshared") if branch_layer == 4 else (
        "control", "full_p2", "unshared_full_p2")
    for name in names:
        mod = mods[name]
        if state_hash(state) != incoming:
            raise RuntimeError("incoming state changed before branch")
        joins, ranks = [], []
        original_join, original_p2 = mod._shared_join_lean, mod._p2_compress

        def traced_join(tw):
            before = tw.ko
            out = original_join(tw)
            joins.append({"layer": current_layer, "incoming_k": tw.k,
                          "before_ko": before, "after_ko": out.ko})
            return out

        def traced_p2(resid):
            z, l = original_p2(resid)
            ranks.append({"layer": current_layer, "rank": int(l.shape[1])})
            return z, l

        mod._shared_join_lean = traced_join
        mod._p2_compress = traced_p2
        try:
            own_state = state
            with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
                for current_layer in range(branch_layer, depth):
                    own_state = mod._layer(cast[current_layer], own_state, table,
                                           dtype, const, 2.0,
                                           last_layer=current_layer == depth - 1,
                                           gens=None, layer_index=current_layer,
                                           depth=depth)
                    state_hash(own_state)
                    mu = fnp.where(fnp.isfinite(own_state[0]), own_state[0], dtype(0.0))
                    own_state = (mu, own_state[1], own_state[2], own_state[3], own_state[4])
                output = np.asarray(own_state[0])
            ledger = _ledger(ctx)
        finally:
            mod._shared_join_lean, mod._p2_compress = original_join, original_p2
        if output.shape != (width,) or not np.isfinite(output).all():
            raise RuntimeError(f"{name}: nonfinite or incomplete output")
        if len(ranks) != depth - branch_layer - 1:
            raise RuntimeError(f"{name}: incomplete births")
        expected_rank = width if name in ("full_p2", "unshared_full_p2") else min(32, width)
        if any(r["rank"] != expected_rank for r in ranks):
            raise RuntimeError(f"{name}: capped or wrong P2 rank")
        if name in ("unshared", "unshared_full_p2"):
            if joins or (own_state[2] is not None and own_state[2].ko):
                raise RuntimeError(f"{name}: unexpected shared join")
        elif (not [j for j in joins if j["after_ko"] > j["before_ko"]] or
              [j for j in joins if j["after_ko"] > j["before_ko"]][0] !=
              {"layer": 4, "incoming_k": 4, "before_ko": 0, "after_ko": 1}):
            raise RuntimeError(f"{name}: first effective join boundary differs: {joins}")
        arms[name] = {"incoming_sha256": incoming, "joins": joins,
                      "birth_ranks": ranks, "ledger": ledger,
                      "final_sha256": hashlib.sha256(output.tobytes()).hexdigest()}
    return {"status": "complete", "width": width, "depth": depth,
            "branch_layer": branch_layer, "shared_rank": shared_rank,
            "incoming_sha256": incoming, "prefix_ledger": prefix, "arms": arms,
            "control_sha256": CONTROL_SHA, "generator_sha256": GENERATOR_SHA,
            "accuracy_claim": False}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))

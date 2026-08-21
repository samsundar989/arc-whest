"""Re-score every recorded verdict under BOTH estimands. CP0.3.

Contract and predictions A1-A5: `experiments/LOG.md`, 2026-08-20, committed as
`0eb1279` BEFORE this file existed.

THE QUESTION
------------
`docs/VALIDATION.md` 2b binds the geometric mean of per-MLP ratios. The
competition scores the ARITHMETIC mean of per-MLP scores, and on our own graded
data the two select opposite winners (1.0334x vs 0.9666x, same 50 rows). Samuel
decided on 2026-08-20 to adopt arithmetic.

Adopting it is not in question here. What IS in question: **does the change
resurrect anything we killed, or kill anything we believe?** That is answerable
from artifacts already on disk, without minting a network or drawing a seed.

WHAT THIS DOES NOT DO
---------------------
It does not re-run any experiment, does not mint, does not draw a seed block, and
does not touch the vault. It reads stored per-MLP arrays and re-aggregates them.

**A re-aggregation of old data is NOT new external evidence.** `AGENTS.md` requires
new external evidence to reopen a closed lane. If a verdict moves across its gate
here, the lane does not reopen -- the WORDING of the kill changes. That rule was
fixed in the contract before these numbers existed.

Run:  uv run python local_sim/rescore_verdicts.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

from competition_score import (  # noqa: E402
    paired_compare,
    rows_from_graded_report,
)

OUT = os.path.join(_ROOT, "experiments", "2026-08-20_rescore", "rescore_verdicts.json")

# --------------------------------------------------------------------------- #
# Constants transcribed from the code that produced the original verdicts, so a
# reproduction failure is visible rather than absorbed.
#   local_sim/packed_sparse_probe.py:1382, 2069-2078
# --------------------------------------------------------------------------- #
LAMBDA = 1e11
GRADED_DENSE_RESIDUAL_S = 9.84e-3
COMPETITOR_RESIDUAL_S = 185.8e-3
COMPETITOR_N = 84_992
COMPETITOR_FIXED_SHARE = 0.039
B = 2.72e11

_PASS = _FAIL = 0
RESULTS: List[Dict[str, Any]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print(f"  {name:<64} ok   {detail}")
    else:
        _FAIL += 1
        print(f"  {name:<64} FAIL {detail}")
    return ok


def head(t: str) -> None:
    print(f"\n{t}")
    print("-" * 118)


def geo(ctrl: np.ndarray, cand: np.ndarray) -> Tuple[float, float, float]:
    """`exp(mean(log(ctrl/cand)))` -- packed_sparse_probe.py:_paired, verbatim."""
    lr = np.log(ctrl) - np.log(cand)
    g = float(np.exp(lr.mean()))
    se = float(lr.std(ddof=1) / math.sqrt(len(lr))) if len(lr) > 1 else float("nan")
    return g, float(np.exp(lr.mean() - 1.96 * se)), float(np.exp(lr.mean() + 1.96 * se))


def biggest_mover_share(ctrl: np.ndarray, cand: np.ndarray) -> float:
    """Fraction of the total arithmetic delta carried by the single biggest mover.

    The screen from `docs/GOTCHAS.md`: >= 100% means treat as a draw until proven
    otherwise. It fires on ~10% of draws from a TRUE 5% improvement, so it is a
    screen, not a proof.
    """
    d = ctrl - cand
    tot = float(d.sum())
    return float(d[np.argmax(np.abs(d))] / tot) if tot != 0 else float("nan")


def compare(label: str, ctrl: np.ndarray, cand: np.ndarray, *,
            lane: str, published: Optional[float] = None,
            published_kind: str = "geometric", gate: Optional[float] = None,
            note: str = "") -> Dict[str, Any]:
    """One lane, both estimands, side by side."""
    ctrl = np.asarray(ctrl, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    res = paired_compare(ctrl, cand)
    g, glo, ghi = geo(ctrl, cand)
    arith = res.arithmetic_gain
    row = {
        "lane": lane, "label": label, "n": int(ctrl.size),
        "arithmetic_gain": arith,
        "arith_t_ci": list(res.t_ci_difference),
        "arith_mean_diff": res.mean_difference,
        "geometric_gain": g, "geometric_ci": [glo, ghi],
        "wins": res.n_candidate_better, "sign_p": res.sign_test_p,
        "biggest_mover_share": biggest_mover_share(ctrl, cand),
        "gate": gate, "published": published, "published_kind": published_kind,
        "note": note,
    }
    # t interval on the paired DIFFERENCE, expressed as a gain bracket
    lo_d, hi_d = res.t_ci_difference
    mc = float(ctrl.mean())
    row["arith_gain_ci_via_t"] = [mc / (mc - lo_d) if mc != lo_d else float("inf"),
                                  mc / (mc - hi_d) if mc != hi_d else float("inf")]
    RESULTS.append(row)

    print(f"  {label:<44} n={row['n']:>3}  "
          f"arith {arith:>7.4f}x   geo {g:>7.4f}x   "
          f"arith/geo {arith / g:>6.4f}   wins {res.n_candidate_better}/{row['n']}"
          f"   top-mover {row['biggest_mover_share'] * 100:>6.1f}%")
    if gate is not None:
        a_verdict = "PASS" if arith >= gate else "FAIL"
        g_verdict = "PASS" if g >= gate else "FAIL"
        flip = " <<< FLIPS ACROSS THE GATE" if a_verdict != g_verdict else ""
        print(f"  {'':<44} gate {gate:.2f}x -> geometric {g_verdict}, "
              f"arithmetic {a_verdict}{flip}")
        row["gate_geometric"] = g_verdict
        row["gate_arithmetic"] = a_verdict
        row["flips"] = a_verdict != g_verdict
    return row


# --------------------------------------------------------------------------- #
# lane loaders
# --------------------------------------------------------------------------- #


def _vc(mse: np.ndarray, flops: np.ndarray, resid_s: float) -> np.ndarray:
    """`V*C` per MLP under a modelled per-arm residual -- packed_sparse_probe:2211."""
    return np.asarray(mse) * (np.asarray(flops) + LAMBDA * resid_s)


def lane_d14() -> None:
    """Sprint 5, block D14, SPENT single-use. Killed at 1.1216x vs a 1.15x gate."""
    f = os.path.join(_ROOT, "experiments", "packed",
                     "2026-08-06_s5E_D14_decision.json")
    d = json.load(open(f, encoding="utf-8"))
    per = d["per_mlp"]
    ctrl_k = "v2.4.1 dense control"
    cand_k = "SPRINT-5 CANDIDATE (frozen)"
    bm = np.array(per[ctrl_k]["mse"]); bf = np.array(per[ctrl_k]["flops"])
    am = np.array(per[cand_k]["mse"]); af = np.array(per[cand_k]["flops"])
    r_ctrl = float(np.median(per[ctrl_k]["resid"]))
    r_cand = float(np.median(per[cand_k]["resid"]))

    m2 = GRADED_DENSE_RESIDUAL_S * (r_cand / max(r_ctrl, 1e-12))
    m3 = max(m2, COMPETITOR_FIXED_SHARE * COMPETITOR_RESIDUAL_S
             + ((1.0 - COMPETITOR_FIXED_SHARE) * COMPETITOR_RESIDUAL_S
                / COMPETITOR_N) * d["n_candidate"])
    ctrl_vc = _vc(bm, bf, GRADED_DENSE_RESIDUAL_S)
    cand_vc = _vc(am, af, m3)

    pub = d["verdicts"][d["deciding_model"]][0]
    g, _lo, _hi = geo(ctrl_vc, cand_vc)
    check("A1: D14 geometric headline reproduces from the stored arrays",
          abs(g - pub) / pub < 1e-9, f"{g:.10f} vs published {pub:.10f}")
    compare("D14 packed (SPENT, gate 1.15x)", ctrl_vc, cand_vc,
            lane="Sprint 4/5 packed activation sparsity",
            published=pub, gate=d["gate"],
            note=f"deciding model {d['deciding_model']}; r_ctrl {r_ctrl*1e3:.1f} ms "
                 f"-> {GRADED_DENSE_RESIDUAL_S*1e3:.2f} ms anchored, "
                 f"r_cand {r_cand*1e3:.1f} ms -> {m3*1e3:.2f} ms")


def lane_d12() -> None:
    """Sprint 4, block D12, SPENT. Its 1.2079x was later retired."""
    f = os.path.join(_ROOT, "experiments", "packed", "D12_decision.json")
    d = json.load(open(f, encoding="utf-8"))
    arms = d["arms"]
    ks = list(arms.keys())
    ctrl_k = next(k for k in ks if "dense" in k.lower())
    cand_k = next(k for k in ks if k != ctrl_k)
    c, a = arms[ctrl_k]["per_mlp"], arms[cand_k]["per_mlp"]
    if not c.get("mse") or float(np.mean(c["mse"])) == 0.0:
        print(f"  D12: cost-only artifact (no MSE) -- skipped")
        return
    r_c = float(np.median(c["resid"])); r_a = float(np.median(a["resid"]))
    m2 = GRADED_DENSE_RESIDUAL_S * (r_a / max(r_c, 1e-12))
    compare("D12 packed (SPENT)",
            _vc(c["mse"], c["flops"], GRADED_DENSE_RESIDUAL_S),
            _vc(a["mse"], a["flops"], m2),
            lane="Sprint 4 packed (superseded by D14)",
            note="model 2 (local ratio on the graded anchor)")


def lane_oa() -> None:
    """The degree-4 design carrier, D18, 48 nets. Published 1.4775x ARITHMETIC."""
    f = os.path.join(_ROOT, "experiments", "2026-08-19_carriers", "oa_arms.json")
    d = json.load(open(f, encoding="utf-8"))
    per = d["per_mlp"]
    bk = "B sobol N=131072"
    ck = "C bchOA N=131072"
    bm = np.array(per[bk]["mse"]); bf = np.array(per[bk]["flops"])
    cm = np.array(per[ck]["mse"]); cf = np.array(per[ck]["flops"])
    # adjusted score = mse * max(0.1, C/B); counted only, as the arms table records
    bs = bm * np.maximum(0.1, bf / B)
    cs = cm * np.maximum(0.1, cf / B)
    row = compare("D18 design carrier B->C (matched N and cost)", bs, cs,
                  lane="the degree-4 design carrier",
                  published=1.4775, published_kind="arithmetic",
                  note="counted FLOPs only; the arms differ by 0.008% in cost")
    check("A1: the D18 carrier reproduces its published 1.4775x ARITHMETIC gain",
          abs(row["arithmetic_gain"] - 1.4775) < 5e-4,
          f"{row['arithmetic_gain']:.6f} vs published 1.4775")
    # and the N-lever leg, A->B
    ak = "A sobol N=16384"
    am_ = np.array(per[ak]["mse"]); af_ = np.array(per[ak]["flops"])
    compare("D18 A->B, the N lever on our own carrier",
            am_ * np.maximum(0.1, af_ / B), bs,
            lane="N as an operating-point lever",
            note="8x the points; the flat law predicts 1.000x")


def lane_graded() -> None:
    """The two graded submissions, 50 public MLPs each, scored on the same networks.

    Through `competition_score.rows_from_graded_report`, NOT a local parser. The
    first draft of this function hand-rolled one, looked for `final_layer_mse` in
    `per_mlp` where it actually lives in `results.public_scores[].scores`, and got
    `mse = 0` -> `log(0)` -> a silent `inf` gain. That is the same
    two-implementations-of-one-identity error this project has now made three
    times in two days; the canonical adapter existed the whole time.
    """
    import glob
    rep = os.path.join(_ROOT, "experiments", "run_reports")

    def scores(pat):
        f = glob.glob(os.path.join(rep, pat))
        if not f:
            return None
        d = json.load(open(f[0], encoding="utf-8"))
        rws = rows_from_graded_report(d)
        return np.array([r.score() for r in rws]), rws

    a = scores("*324420*")   # v2.4
    b = scores("*327743*")   # v2.4.1
    if a is None or b is None:
        print("  graded pair: reports not both present -- skipped")
        return
    (sa, ra), (sb, rb) = a, b
    if sa.size != sb.size:
        print(f"  graded pair: {sa.size} vs {sb.size} rows -- skipped")
        return
    check("A1: each graded report reproduces its published aggregate bitwise",
          abs(float(sa.mean()) - 2.719427529595e-07) < 1e-18
          and abs(float(sb.mean()) - 2.631432177242e-07) < 1e-18,
          f"{float(sa.mean()):.12e} / {float(sb.mean()):.12e}")
    row = compare("GRADED v2.4 -> v2.4.1 (50 public MLPs)", sa, sb,
                  lane="the graded pair",
                  published=1.0334, published_kind="arithmetic",
                  note="the fixture that motivated the whole change")
    check("A1: the graded pair reproduces the documented estimand disagreement",
          abs(row["arithmetic_gain"] - 1.0334) < 2e-3
          and abs(row["geometric_gain"] - 0.9666) < 2e-3,
          f"arith {row['arithmetic_gain']:.4f} (1.0334), "
          f"geo {row['geometric_gain']:.4f} (0.9666)")


def main() -> int:
    print("=" * 118)
    print("CP0.3 — EVERY RECORDED VERDICT, RE-SCORED UNDER BOTH ESTIMANDS")
    print("=" * 118)
    print("Contract: experiments/LOG.md 2026-08-20, committed 0eb1279 before this file.")
    print("No network is minted, no seed block drawn, no vault touch.\n")

    head("1. the graded pair — the fixture that motivated the change")
    lane_graded()
    head("2. the SPENT single-use decision blocks")
    lane_d14()
    lane_d12()
    head("3. the design-carrier lane")
    lane_oa()

    head("4. A2 — is arithmetic systematically >= geometric?")
    ratios = [r["arithmetic_gain"] / r["geometric_gain"] for r in RESULTS]
    n_ge = sum(1 for x in ratios if x >= 1.0)
    check("A2: arithmetic >= geometric in at least 3 of the comparisons",
          n_ge >= 3, f"{n_ge}/{len(ratios)}; arith/geo = "
                     + ", ".join(f"{x:.4f}" for x in ratios))

    head("5. did any verdict FLIP across its gate?")
    gated = [r for r in RESULTS if r.get("gate") is not None]
    flipped = [r for r in gated if r.get("flips")]
    for r in gated:
        print(f"  {r['label']:<44} gate {r['gate']:.2f}x   "
              f"geometric {r['gate_geometric']}   arithmetic {r['gate_arithmetic']}")
    if flipped:
        print()
        for r in flipped:
            print(f"  *** {r['lane']}: the verdict is STATISTIC-DEPENDENT.")
            print(f"      geometric {r['geometric_gain']:.4f}x -> {r['gate_geometric']}, "
                  f"arithmetic {r['arithmetic_gain']:.4f}x -> {r['gate_arithmetic']}, "
                  f"gate {r['gate']:.2f}x")
            print(f"      Per the contract this does NOT reopen the lane -- a")
            print(f"      re-aggregation of old data is not new EXTERNAL evidence.")
            print(f"      The WORDING of the kill must weaken. See CURRENT_STATE.")
    else:
        print("\n  No gated verdict flips. The change is verdict-neutral.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"contract_commit": "0eb1279",
                   "lambda": LAMBDA, "B": B,
                   "graded_dense_residual_s": GRADED_DENSE_RESIDUAL_S,
                   "n_flipped": len(flipped),
                   "comparisons": RESULTS}, fh, indent=1)
    print(f"\nsaved {os.path.relpath(OUT, _ROOT)}")

    print("\n" + "=" * 118)
    print(f"{_PASS}/{_PASS + _FAIL} reproduction checks passed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

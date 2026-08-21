"""Phase-2 mint provenance, checked against the ORGANISERS' OWN BYTES.

P2-CP0 deliverable. `local_sim/mint_provenance.py` proves that
`canonical_mint` equals `whestbench.sample_mlp` **as installed here**, which is
whestbench 0.14.0. The Phase-2 data was baked with whestbench **0.16.0**, which
is not installable from any public source (see the Phase-2 manifest's
`package_parity`). So the usual proof has a hole at exactly the release we need.

This module closes that hole from the other side. Instead of comparing our mint
to a package we cannot install, it compares our mint to the **weights the
organisers actually published at `v2-phase2`**:

    canonical_mint(1024, 16, row["mlp_seed"])  ==  row["weights"]     bit for bit

Under seed protocol 3.0 (`whestbench_explicit_per_mlp_seeds`) the parquet
`mlp_seed` column is the **input** seed, so this is a complete end-to-end test of
the seed tree, the draw order, the float64-then-cast precision, and the He scale
-- against 0.16.0's real output, without 0.16.0.

Equality is on the raw uint32 bit patterns, never a tolerance.

It also checks three things the same rows can settle for free:

  * **the truth-noise floor**, exactly: the card states `gt_mse = avg_variance / N`
    and `avg_variance` is a published column, so the Phase-2 public split's noise
    floor is a computed number rather than an estimate;
  * **the no-bias / no-output-layer contract**: `weights` has exactly `depth`
    matrices and `all_layer_means` exactly `depth` rows, so every matrix is
    followed by a ReLU and none is a bare linear head;
  * **the layer-1 closed form**, which is exact and shape-free:
    `E[ReLU(h_1)_j] = ||W_1[:, j]|| / sqrt(2 pi)`. Checking it against
    `all_layer_means[0]` at N = 1e9 simultaneously confirms the row-vector
    convention and that the published means are means of what we think they are.

Run:
    uv run python local_sim/phase2_mint_parity.py
    uv run python local_sim/phase2_mint_parity.py --shard 1 --max-rows 4

One artifact per (shard, rows); a run REFUSES to overwrite a differing file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

from mint_provenance import canonical_mint  # noqa: E402
from phase2_manifest import load as load_manifest  # noqa: E402

REPO = "aicrowd/arc-whestbench-public-2026"
OUT_DIR = os.path.join(_ROOT, "experiments", "2026-08-21_phase2_manifest")
def _out_path(shard: int, n_rows: int) -> str:
    """One file per (shard, rows) -- a 4-row run can no longer clobber a 16-row one.

    The flat `phase2_mint_parity.json` this module first wrote was overwritten by
    running the command printed in its own docstring, destroying 12 rows of
    committed evidence. An evidence file that a documented invocation can shrink
    is not evidence.
    """
    return os.path.join(OUT_DIR, f"phase2_mint_parity_shard{shard}_rows{n_rows}.json")


def _max_abs_z_pvalue(z: float, m: int) -> float:
    """P(max of m iid |N(0,1)| >= z). EXACT, so it needs no asymptotic bar.

    An earlier version of this file quoted an "expected max" from the tail
    approximation sqrt(2 log(2m/sqrt(2 pi))), which drops the 1/t factor and
    OVERSTATES E[max] -- 4.354 against a simulated 4.136 (sd 0.286) at
    m = 16,384. An independent Skeptic caught it. A p-value is exact, needs no
    simulation, and is the quantity a bar should be set on anyway.
    """
    per = math.erf(z / math.sqrt(2.0))        # P(|N(0,1)| < z)
    if per >= 1.0:
        return 0.0
    return -math.expm1(m * math.log(per))     # 1 - per**m, stable near 1


def _shard(idx: int, revision: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        REPO,
        f"data/mini-{idx:05d}-of-00007.parquet",
        repo_type="dataset",
        revision=revision,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = every row")
    a = ap.parse_args()

    man = load_manifest()
    width = man["shape"]["width"]
    depth = man["shape"]["depth"]
    revision = man["dataset"]["commit"]
    n_truth = man["truth"]["n_samples_per_mlp"]
    parity = man["package_parity"]

    import pyarrow.parquet as pq

    path = _shard(a.shard, revision)
    tbl = pq.read_table(
        path,
        columns=["mlp_id", "mlp_name", "mlp_seed", "weights",
                 "all_layer_means", "avg_variance"],
    )
    n_rows = tbl.num_rows if a.max_rows == 0 else min(a.max_rows, tbl.num_rows)

    print("=" * 78)
    print("PHASE-2 MINT PARITY  --  canonical_mint  vs  the PUBLISHED v2-phase2 weights")
    print("=" * 78)
    print(f"  dataset      {REPO}@{revision[:12]}  shard {a.shard}")
    print(f"  shape        {width} x {depth}     rows checked: {n_rows} of {tbl.num_rows}")
    print(f"  bake stack   whestbench {parity['bake_stack']['whestbench']}"
          f" / flopscope {parity['bake_stack']['flopscope']}"
          f"   (installed {parity['installed_here']['whestbench']}"
          f" / {parity['installed_here']['flopscope']})")
    print(f"  parity       {parity['status']} -- this check does NOT need the")
    print("               packages, only the organisers' published bytes")
    print()

    rows = []
    seeds_ok = w_ok = shape_ok = 0
    max_l1_rel = 0.0
    max_l1_z = 0.0
    z_rms_all = []
    gt_noise = []

    seed_col = tbl.column("mlp_seed").to_pylist()
    name_col = tbl.column("mlp_name").to_pylist()
    id_col = tbl.column("mlp_id").to_pylist()
    var_col = tbl.column("avg_variance").to_pylist()

    print(f"  {'id':>4} {'mlp_name':<22} {'weights bitwise':>15} "
          f"{'L1 max|z|':>10} {'L1 rms z':>9}")
    print("  " + "-" * 76)

    for i in range(n_rows):
        pub_w = np.asarray(
            tbl.column("weights")[i].as_py(), dtype=np.float32
        ).reshape(depth, width, width)
        pub_m = np.asarray(
            tbl.column("all_layer_means")[i].as_py(), dtype=np.float32
        ).reshape(depth, width)

        mine_w, est_seed, _ = canonical_mint(width, depth, seed_col[i])
        mine = np.stack(mine_w)

        bit = (
            mine.dtype == pub_w.dtype
            and mine.shape == pub_w.shape
            and np.array_equal(mine.view(np.uint32), pub_w.view(np.uint32))
        )
        w_ok += int(bit)

        # No biases and no separate output layer: `depth` matrices, `depth` mean
        # rows. A bare linear head would give depth+1 matrices or depth-1 rows.
        sh = pub_w.shape == (depth, width, width) and pub_m.shape == (depth, width)
        shape_ok += int(sh)

        # Layer-1 closed form, exact for x ~ N(0, I) under the ROW-vector
        # convention h @ W:  E[ReLU] = ||W[:, j]|| / sqrt(2 pi).
        #
        # A raw relative error is NOT a test here -- the published value is a
        # Monte-Carlo mean over N = 1e9 draws and carries its own noise. So the
        # deviation is scored in units of that noise, which is also known in
        # closed form: for z ~ N(0, sigma^2), Var[ReLU(z)] = sigma^2 (1/2 - 1/2pi).
        # The right statement is "the residual is consistent with N(0,1)", not
        # "the residual is small".
        sig = np.linalg.norm(pub_w[0].astype(np.float64), axis=0)
        l1 = sig / math.sqrt(2 * math.pi)
        se = sig * math.sqrt(0.5 - 1.0 / (2 * math.pi)) / math.sqrt(n_truth)
        resid = (pub_m[0].astype(np.float64) - l1) / se
        rel = float(np.abs(l1 - pub_m[0].astype(np.float64)).max() / np.abs(l1).max())
        max_l1_rel = max(max_l1_rel, rel)
        z_abs_max = float(np.abs(resid).max())
        z_rms = float(np.sqrt((resid ** 2).mean()))
        max_l1_z = max(max_l1_z, z_abs_max)
        z_rms_all.append(z_rms)

        gt_noise.append(var_col[i] / n_truth)
        seeds_ok += int(isinstance(seed_col[i], int))

        rows.append({
            "mlp_id": id_col[i],
            "mlp_name": name_col[i],
            "mlp_seed": int(seed_col[i]),
            "mlp_seed_bits": int(seed_col[i]).bit_length(),
            "derived_estimator_seed": int(est_seed),
            "weights_bitwise_identical": bool(bit),
            "shape_contract_ok": bool(sh),
            "layer1_closed_form_max_rel": rel,
            "avg_variance": var_col[i],
            "gt_mse": var_col[i] / n_truth,
        })
        rows[-1]["layer1_z_abs_max"] = z_abs_max
        rows[-1]["layer1_z_rms"] = z_rms
        print(f"  {id_col[i]:>4} {name_col[i]:<22} "
              f"{('YES' if bit else 'NO'):>15} {z_abs_max:>10.2f} {z_rms:>9.3f}")

    # ---- the layer-1 statistic DECIDES, it does not merely print ----------
    # Bars stated before the numbers were looked at, and both are wide enough
    # that only a structural error trips them:
    #   rms z has sd ~ 1/sqrt(2 * width * n) under the null, i.e. ~0.0055 here,
    #   so [0.90, 1.10] is an ~18-sigma bar -- it cannot fire on noise, and the
    #   wrong (column-norm) convention reads rms z ~ 675.
    #   the max |z| p-value guards a single structural outlier that an rms
    #   average would dilute.
    Z_RMS_LO, Z_RMS_HI, Z_MAX_P_MIN = 0.90, 1.10, 1e-6
    z_rms_mean = float(np.mean(z_rms_all))
    z_p = _max_abs_z_pvalue(max_l1_z, n_rows * width)
    z_ok = (Z_RMS_LO <= z_rms_mean <= Z_RMS_HI) and (z_p >= Z_MAX_P_MIN)

    verdict = (w_ok == n_rows == shape_ok) and z_ok

    gt = float(np.mean(gt_noise))
    seed_bits = max(int(r["mlp_seed"]).bit_length() for r in rows)

    print("  " + "-" * 76)
    print(f"  weights bit-identical to the published Phase-2 bytes : {w_ok}/{n_rows}")
    print(f"  shape contract (no biases, no output layer)          : {shape_ok}/{n_rows}")
    print(f"  layer-1 closed form, worst max relative error        : {max_l1_rel:.3e}")
    print(f"  layer-1 residual, rms z   [bar {Z_RMS_LO}, {Z_RMS_HI}]"
          f"{'':>9}: {z_rms_mean:.4f}   {'PASS' if Z_RMS_LO <= z_rms_mean <= Z_RMS_HI else 'FAIL'}")
    print(f"  layer-1 worst |z| over {n_rows * width:,} neurons"
          f"{'':>8}: {max_l1_z:.2f}"
          f"   exact P(max >= z) = {z_p:.3f}   "
          f"{'PASS' if z_p >= Z_MAX_P_MIN else 'FAIL'}")
    print(f"  widest published mlp_seed                            : {seed_bits} bits")
    print(f"  ground-truth noise floor, EXACT (avg_variance / N)   : {gt:.6e}")
    print()

    if verdict:
        print("  MINT PARITY WITH THE PHASE-2 STACK IS PROVEN.")
        print("  It did not need whestbench 0.16.0: the published weights ARE")
        print("  0.16.0's output, and our reimplementation reproduces them exactly.")
        print("  This closes the MINT half of the parity blocker. The COST half")
        print("  (flopscope 0.12 pricing) stays open -- it is not testable from data.")
    else:
        print("  *** MINT PARITY FAILED. Every fresh-mint number at 1024 x 16 would")
        print("  *** be measured on networks the grader does not have. STOP. ***")

    art = {
        "artifact": "phase2_mint_parity",
        "artifact_version": 1,
        "dataset": {"repo": REPO, "revision_commit": revision,
                    "shard": f"data/mini-{a.shard:05d}-of-00007.parquet"},
        "shape": {"width": width, "depth": depth},
        "truth_n_samples": n_truth,
        "package_parity_at_check_time": {
            "status": parity["status"],
            "blockers": parity["blockers"],
            "limitation_stamp": parity["limitation_stamp"],
            "note": "This check is INDEPENDENT of the blocker: it compares our "
                    "mint to the organisers' published bytes, not to a package.",
        },
        "n_rows_checked": n_rows,
        "weights_bit_identical": w_ok,
        "shape_contract_ok": shape_ok,
        "layer1_closed_form_max_rel": max_l1_rel,
        "layer1_z_abs_max": max_l1_z,
        "layer1_z_rms_mean": float(np.mean(z_rms_all)),
        "layer1_z_max_pvalue_exact": z_p,
        "layer1_z_bars": {"rms_lo": Z_RMS_LO, "rms_hi": Z_RMS_HI,
                          "max_p_min": Z_MAX_P_MIN},
        "layer1_z_gate_passed": bool(z_ok),
        "layer1_z_note": "residual of the published layer-1 mean against the "
                         "EXACT closed form, in units of its own N = 1e9 Monte "
                         "Carlo standard error. rms z ~ 1 and max |z| near the "
                         "Gaussian order statistic is the calibrated statement; "
                         "a small relative error is not.",
        "max_published_seed_bits": seed_bits,
        "gt_noise_floor_exact": gt,
        "gt_noise_floor_formula": "mean_i(avg_variance_i) / N",
        "verdict": "PROVEN" if verdict else "FAILED",
        "verdict_components": {
            "weights_bitwise": w_ok == n_rows,
            "shape_contract": shape_ok == n_rows,
            "layer1_z_calibrated": bool(z_ok),
        },
        "rows": rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out = _out_path(a.shard, n_rows)
    stable = {k: v for k, v in art.items() if k != "generated_at_utc"}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            old = json.load(fh)
        same = json.dumps({k: old.get(k) for k in stable}, sort_keys=True,
                          default=str) == json.dumps(stable, sort_keys=True,
                                                     default=str)
        if not same:
            raise SystemExit(
                f"REFUSED: {os.path.relpath(out, _ROOT)} exists with different "
                "content. Evidence rows are immutable. Delete it deliberately, "
                "or run a different shard/row count."
            )
        print(f"  artifact unchanged -> {os.path.relpath(out, _ROOT)}")
    else:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(art, indent=2) + "\n")
        print(f"  artifact -> {os.path.relpath(out, _ROOT)}")
    print("=" * 78)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

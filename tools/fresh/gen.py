"""Fresh-network generator (Lane V, V1) -- whestbench's OWN mint, reserved seeds.

PLAN rev 18 §2: acceptance and designation are decided on networks nothing has
touched. Those networks must come from the grader's code path, not from a
look-alike, so this module mints through the exact functions the organizers'
bake calls (`whestbench/dataset.py` create_dataset, seed protocol 3.0):

    weight_ss, _sample_ss, est_seed = whestbench.seeds.derive_seed_streams(input_seed)
    mlp = whestbench.generation.sample_mlp(width, depth, fnp.random.default_rng(weight_ss),
                                           seed=est_seed)

`mint_numpy` is a dependency-free mirror of the same draw (SeedSequence spawn,
PCG64 standard_normal in float64, scale, cast to float32). It exists so the GPU
labeller (`.venv-gpu`, no whestbench) and the shimmed evaluator workers can
rebuild a net from its seed. `--parity` proves both paths equal the organizers'
published `mini` bytes BIT FOR BIT (uint32 patterns, no tolerance), and every
panel net's sha256 is recorded so any later rebuild is checked, not trusted.

Seed reservation (never reuse a number from these ranges for anything else):

    FRESH-A      9_100_000 + i,  i in [0, 100)     acceptance panel
    FRESH-LOCK   9_100_100 + i,  i in [0, 200)     designation lockbox
    LANE-M       9_200_000 + 1000*w_tag + i        width-override nets (n = 256/512/1024)

Organizer seeds are random int63 (~1e18); a collision with this 7-digit range is
impossible in practice and `tools/panels.py --check` asserts it anyway.

    python tools/fresh/gen.py --parity [--n 5]
    python tools/fresh/gen.py --build          write experiments/2026-10-06_laneV/seeds.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEEDS_JSON = ROOT / "experiments" / "2026-10-06_laneV" / "seeds.json"

WIDTH, DEPTH = 1024, 16
FRESH_A_BASE, FRESH_A_N = 9_100_000, 100
FRESH_LOCK_BASE, FRESH_LOCK_N = 9_100_100, 200
FRESH_LOCK_N1E9 = 20          # the first 20 FRESH-LOCK nets also get N = 1e9
LANE_M_BASE = 9_200_000
RESERVED = (9_100_000, 9_300_000)


def main_checkout() -> Path:
    """The primary checkout, so worktrees share one label cache.

    In a git worktree `.git` is a file; the common dir's parent is the main repo.
    """
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=str(ROOT),
                                capture_output=True, text=True, check=True).stdout.strip()
        p = Path(common)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return p.parent
    except Exception:
        return ROOT


def cache_dir() -> Path:
    return Path(os.environ.get("ARC_FRESH_CACHE", main_checkout() / "cache" / "fresh"))


def lane_m_seed(width: int, i: int) -> int:
    tag = {256: 0, 512: 1, 1024: 2}[width]
    return LANE_M_BASE + 1000 * tag + i


def mint_numpy(input_seed: int, width: int = WIDTH, depth: int = DEPTH) -> np.ndarray:
    """(depth, width, width) float32, stored (input, output): h <- ReLU(h @ W[l])."""
    weight_ss = np.random.SeedSequence(int(input_seed)).spawn(3)[0]
    rng = np.random.default_rng(weight_ss)
    scale = float(np.sqrt(2.0 / width))
    return np.stack([(rng.standard_normal((width, width)) * scale).astype(np.float32)
                     for _ in range(depth)])


def estimator_seed(input_seed: int) -> int:
    """Protocol 3.0 MLP.seed (the third SeedSequence child)."""
    return int(np.random.SeedSequence(int(input_seed)).spawn(3)[2].generate_state(1)[0])


def mint_whestbench(input_seed: int, width: int = WIDTH, depth: int = DEPTH):
    """The grader's path verbatim. Requires the real whestbench + flopscope."""
    import flopscope.numpy as fnp  # noqa: PLC0415
    from whestbench.generation import sample_mlp  # noqa: PLC0415
    from whestbench.seeds import derive_seed_streams  # noqa: PLC0415
    weight_ss, _sample_ss, est = derive_seed_streams(int(input_seed))
    mlp = sample_mlp(width, depth, fnp.random.default_rng(weight_ss), seed=est)
    w = np.stack([np.asarray(x, dtype=np.float32) for x in mlp.weights])
    return w, int(mlp.seed)


def sha256_weights(w: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(w, dtype=np.float32).tobytes()).hexdigest()


def panel_seeds(panel: str) -> list:
    if panel == "fresh_a":
        return [FRESH_A_BASE + i for i in range(FRESH_A_N)]
    if panel == "fresh_lock":
        return [FRESH_LOCK_BASE + i for i in range(FRESH_LOCK_N)]
    raise KeyError(panel)


def load_seeds() -> dict:
    return json.loads(SEEDS_JSON.read_text(encoding="utf-8"))


FRESH_D_JSON = ROOT / "experiments" / "2026-10-06_laneV" / "fresh_d.json"


def panel_record(panel: str) -> dict:
    """seeds + weight hashes (+ label_n) of a pinned fresh panel."""
    if panel == "fresh_d":
        return json.loads(FRESH_D_JSON.read_text(encoding="utf-8"))
    return load_seeds()["panels"][panel]


def label_path(tag: str, seed: int, n: int) -> Path:
    """Where `mc.py` writes the label of one net (tag = panel, 'mini', 'lane_m_w512')."""
    return cache_dir() / "labels" / tag / f"{seed}_N{n:.0e}.npz".replace("+", "")


def load_label(panel: str, i: int, n: int = 10**8) -> dict:
    """{'means', 'var', 'n', 'floor'} for net i of a fresh panel.

    floor[l] = mean_j Var[h_lj] / N: the expected MSE contribution of the label's
    own Monte Carlo noise at layer l. Subtract floor[-1] from a final-layer MSE to
    get a floor-free absolute number; paired differences need no correction.
    """
    rec = panel_record(panel)
    n = rec.get("label_n", n)
    p = label_path(panel, rec["seeds"][i], n)
    if not p.exists():
        raise SystemExit(f"{panel}[{i}] has no N={n:.0e} label at {p}; run tools/fresh/mc.py")
    with np.load(p) as d:
        if str(d["weight_sha256"]) != rec["weight_sha256"][i]:
            raise SystemExit(f"{p}: label was drawn for different weights")
        means, var, nn = np.array(d["means"]), np.array(d["var"]), int(d["n"])
    return {"means": means, "var": var, "n": nn, "floor": var.mean(axis=1) / nn}


def mint_checked(panel: str, i: int) -> np.ndarray:
    """Rebuild net i of a fresh panel and verify it against the recorded hash."""
    rec = panel_record(panel)
    if panel == "fresh_d":   # torch-generated d8b nets: staged float32 files
        w = np.fromfile(cache_dir() / "fresh_d_weights" / f"{rec['seeds'][i]}.f32",
                        dtype=np.float32).reshape(DEPTH, WIDTH, WIDTH)
    else:
        w = mint_numpy(rec["seeds"][i])
    h = sha256_weights(w)
    if h != rec["weight_sha256"][i]:
        raise SystemExit(f"{panel}[{i}] seed {rec['seeds'][i]}: weight hash {h[:12]} != "
                         f"recorded {rec['weight_sha256'][i][:12]} -- generator drift")
    return w


# --------------------------------------------------------------------------- #

def parity(n: int) -> int:
    sys.path.append(str(ROOT / "local_sim"))  # APPEND: real packages must win
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    import real_data as rd  # noqa: PLC0415
    bad = 0
    rows = []
    for i in range(n):
        w_row, _truth, meta = rd.get(i, "mini", dtype=np.float32)
        s = int(meta["mlp_seed"])
        w_wb, est_wb = mint_whestbench(s)
        w_np = mint_numpy(s)
        eq_wb = np.array_equal(w_wb.view(np.uint32), w_row.view(np.uint32))
        eq_np = np.array_equal(w_np.view(np.uint32), w_row.view(np.uint32))
        # INFORMATIONAL ONLY. MLP.seed is the estimator's private RNG stream, not
        # part of the network: the offline loader resolves the mini metadata as
        # protocol 2.0 (hands back the input seed) and a private 4.0 suite hands a
        # keyed digest nobody can reproduce. v68/v73 never read it. We check only
        # that our 3.0 mirror equals whestbench's own derivation.
        eq_est = est_wb == estimator_seed(s)
        ok = eq_wb and eq_np and eq_est
        bad += not ok
        rows.append(dict(i=i, name=meta["mlp_name"], mlp_seed=s, whestbench_bitexact=eq_wb,
                         numpy_bitexact=eq_np, estimator_seed_match=eq_est,
                         sha256=sha256_weights(w_row)))
        print(f"  mini[{i}] {meta['mlp_name']:<24} seed={s}  whestbench={eq_wb}  "
              f"numpy={eq_np}  est-seed-3.0={eq_est}", flush=True)
    import whestbench  # noqa: PLC0415
    out = SEEDS_JSON.parent / "gen_parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"whestbench": getattr(whestbench, "__version__", "?"),
                               "numpy": np.__version__, "rows": rows,
                               "verdict": "BIT-IDENTICAL" if not bad else f"{bad} MISMATCH"},
                              indent=1) + "\n", encoding="utf-8")
    print("PARITY: BIT-IDENTICAL" if not bad else f"PARITY: {bad} MISMATCHES")
    return bad


def build() -> None:
    if SEEDS_JSON.exists():
        raise SystemExit(f"{SEEDS_JSON} exists; panels are pinned and never rebuilt")
    out = {"generator": "whestbench.generation.sample_mlp via seeds.derive_seed_streams "
                        "(protocol 3.0); hashes from mint_numpy, cross-checked against "
                        "mint_whestbench on every net",
           "width": WIDTH, "depth": DEPTH, "reserved_range": list(RESERVED), "panels": {}}
    for panel in ("fresh_a", "fresh_lock"):
        seeds = panel_seeds(panel)
        hashes, ests = [], []
        for k, s in enumerate(seeds):
            w = mint_numpy(s)
            w_wb, est = mint_whestbench(s)
            if not np.array_equal(w.view(np.uint32), w_wb.view(np.uint32)):
                raise SystemExit(f"{panel} seed {s}: numpy mirror != whestbench mint")
            hashes.append(sha256_weights(w))
            ests.append(est)
            if (k + 1) % 25 == 0:
                print(f"  {panel}: {k + 1}/{len(seeds)}", flush=True)
        out["panels"][panel] = {"seeds": seeds, "estimator_seeds": ests,
                                "weight_sha256": hashes}
    out["panels"]["fresh_lock"]["n1e9_subset"] = list(range(FRESH_LOCK_N1E9))
    SEEDS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SEEDS_JSON.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {SEEDS_JSON}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    if a.parity:
        return 1 if parity(a.n) else 0
    if a.build:
        build()
    return 0


if __name__ == "__main__":
    sys.exit(main())

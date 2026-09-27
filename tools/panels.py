"""The three evaluation panels, and the index files that pin them.

PLAN §4:

    DEV     the 100 organizer `mini` nets — every candidate is scored here;
            never fitted on
    FIT     full rows 0-99, DISJOINT from mini. Since 2026-09-25 (HANDOFF
            09-25 Step 2, authorised) it is part of TRAIN.
    VAL     full rows 900-999 (`twin_train_split.json` `holdout_100`): the
            twin's checkpoint-selection panel. Repeatedly read, never fitted —
            a VALIDATION set, not an untouched test.

PANEL SEMANTICS, repaired 2026-09-27 (PLAN rev 15 §3). The Sep 21 `holdout.json`
(full rows 100-999) is HISTORICAL: rows 100-899 are now TRAIN, so `--panel holdout`
would silently score trained nets under a holdout label. It is refused; use `val`,
or `holdout_legacy` to reproduce a historical row (labelled as containing TRAIN).
`python tools/panels.py --check` asserts membership and disjointness.

Identity is `mlp_seed`. The seed is what the organizers' generator consumes, so
two rows with the same seed are the same network and two rows with different
seeds are not; disjointness is therefore exact rather than probabilistic. A
sha256 of the weight tensor is recorded for DEV and FIT as an independent check
that the seed identity is not lying (`--verify-hashes`), because the whole point
of FIT is that nothing fitted on it has seen a DEV net.

The index files (`data/panels/{dev,fit,holdout}.json`) pin the membership so a
later session evaluates the same nets without re-deriving them, and so a diff
shows if a panel ever moves.

    python tools/panels.py --build            write the three index files
    python tools/panels.py --verify-hashes    hash DEV+FIT weights, ~5 min
    python tools/panels.py --show             summarize what is pinned
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
LOCAL_SIM = ROOT / "local_sim"

# APPEND, never insert: inserting puts the numpy shims named `flopscope` and
# `whestbench` ahead of the real packages and every counted FLOP becomes fiction.
if str(LOCAL_SIM) not in sys.path:
    sys.path.append(str(LOCAL_SIM))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

PANEL_DIR = ROOT / "data" / "panels"
PANELS = ("dev", "fit", "val", "holdout_legacy", "holdout")
#: Files `--build` writes. `val` / `holdout_legacy` are views over them plus the
#: pinned twin split; `holdout` is refused (see the module docstring).
BUILT = ("dev", "fit", "holdout")
TWIN_SPLIT = PANEL_DIR / "twin_train_split.json"

#: FIT is the first 100 full-split nets that are not in mini, in split order.
#: Deterministic, reproducible, and independent of any result — a panel chosen
#: after seeing a score is not a panel.
FIT_SIZE = 100


def _real_data():
    import real_data  # noqa: PLC0415
    return real_data


def _meta(split: str) -> Dict[str, list]:
    ds = _real_data()._load(split).select_columns(["mlp_id", "mlp_name", "mlp_seed"])
    return ds.to_dict()


def build() -> Dict[str, dict]:
    rd = _real_data()
    mini = _meta("mini")
    full = _meta("full")

    mini_seeds = set(mini["mlp_seed"])
    if len(mini_seeds) != len(mini["mlp_seed"]):
        raise SystemExit("mini split has duplicate mlp_seeds; identity is not a seed")

    disjoint = [i for i, s in enumerate(full["mlp_seed"]) if s not in mini_seeds]
    overlap = len(full["mlp_seed"]) - len(disjoint)

    fit_idx = disjoint[:FIT_SIZE]
    holdout_idx = disjoint[FIT_SIZE:]

    common = {
        "dataset": rd.DATASET,
        "revision": rd.REVISION,
        "built_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "identity": "mlp_seed",
        "mini_full_overlap": overlap,
    }

    out = {
        "dev": dict(common, panel="dev", split="mini",
                    purpose="score every candidate here; never fit on it",
                    indices=list(range(len(mini["mlp_seed"]))),
                    seeds=list(mini["mlp_seed"]),
                    names=list(mini["mlp_name"])),
        "fit": dict(common, panel="fit", split="full",
                    purpose="the ONLY panel a coefficient may be fitted on",
                    indices=fit_idx,
                    seeds=[full["mlp_seed"][i] for i in fit_idx],
                    names=[full["mlp_name"][i] for i in fit_idx]),
        "holdout": dict(common, panel="holdout", split="full",
                        purpose="read at most twice a week; never fitted",
                        indices=holdout_idx,
                        seeds=[full["mlp_seed"][i] for i in holdout_idx],
                        names=[full["mlp_name"][i] for i in holdout_idx]),
    }

    # The property that makes FIT worth having, asserted rather than assumed.
    for a, b in (("dev", "fit"), ("dev", "holdout"), ("fit", "holdout")):
        clash = set(out[a]["seeds"]) & set(out[b]["seeds"])
        if clash:
            raise SystemExit(f"{a} and {b} share {len(clash)} nets — panels are not disjoint")

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in out.items():  # holdout.json: the historical Sep 21 definition
        (PANEL_DIR / f"{name}.json").write_text(
            json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return out


def _load_file(name: str) -> dict:
    p = PANEL_DIR / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"panel {name!r} is not built; run: python tools/panels.py --build")
    return json.loads(p.read_text(encoding="utf-8"))


def twin_split() -> dict:
    return json.loads(TWIN_SPLIT.read_text(encoding="utf-8"))


def train_indices() -> set:
    """full-split rows the twin trains on (FIT + former holdout 100-899)."""
    return set(twin_split()["train"]["indices"])


def load(panel: str) -> dict:
    if panel == "holdout":
        raise SystemExit(
            "--panel holdout is RETIRED (2026-09-27): data/panels/holdout.json is the "
            "Sep 21 definition (full rows 100-999) and rows 100-899 are now TRAIN "
            "(twin_train_split.json). Use --panel val (full 900-999, the checkpoint-"
            "selection set) or --panel holdout_legacy to reproduce a historical row.")
    if panel in ("dev", "fit"):
        return _load_file(panel)
    legacy = _load_file("holdout")
    if panel == "holdout_legacy":
        return dict(legacy, panel="holdout_legacy",
                    purpose="HISTORICAL Sep 21 holdout (full 100-999); rows 100-899 are TRAIN")
    if panel == "val":
        pos = {i: k for k, i in enumerate(legacy["indices"])}
        idx = list(twin_split()["holdout_100"]["indices"])
        missing = [i for i in idx if i not in pos]
        if missing:
            raise SystemExit(f"val rows {missing[:5]} are not in holdout.json")
        return dict(legacy, panel="val", indices=idx,
                    seeds=[legacy["seeds"][pos[i]] for i in idx],
                    names=[legacy["names"][pos[i]] for i in idx],
                    purpose="VALIDATION (twin checkpoint selection); never fitted, "
                            "repeatedly read -- not an untouched test")
    raise SystemExit(f"unknown panel {panel!r}; choose from {PANELS[:-1]}")


def trained_fraction(panel: str, n: int | None = None) -> float:
    """Share of the first `n` rows of `panel` that are in the twin's TRAIN set."""
    d = load(panel)
    if d["split"] != "full":
        return 0.0
    idx = d["indices"][:n] if n else d["indices"]
    tr = train_indices()
    return sum(i in tr for i in idx) / max(1, len(idx))


def label(panel: str, n: int | None = None) -> str:
    """The RESULTS.md panel tag, which says when trained nets are being scored."""
    f = trained_fraction(panel, n)
    tag = panel.upper().replace("_", "-")
    return f"{tag} (TRAIN {f:.0%})" if f > 0 else tag


def check() -> int:
    """Membership and disjointness of every panel. Returns the number of failures."""
    bad = 0

    def ok(cond, msg):
        nonlocal bad
        print(("  ok    " if cond else "  FAIL  ") + msg)
        bad += not cond

    dev, fit, legacy = _load_file("dev"), _load_file("fit"), _load_file("holdout")
    ts = twin_split()
    train = list(ts["train"]["indices"])
    val = list(ts["holdout_100"]["indices"])
    tr, va = set(train), set(val)
    full_seeds = set(fit["seeds"]) | set(legacy["seeds"])
    ok(dev["split"] == "mini" and len(dev["indices"]) == 100, "DEV = mini rows 0-99")
    ok(not set(dev["seeds"]) & full_seeds, "DEV seeds disjoint from every full row")
    ok(dev.get("weight_sha256_covers", 0) == 100 and fit.get("weight_sha256_covers", 0) == 100
       and not set(dev["weight_sha256"]) & set(fit["weight_sha256"]),
       "DEV/FIT weight hashes recorded (100 + 100) and disjoint")
    ok(len(train) == len(tr) == ts["train"]["n"] == 900, "TRAIN has 900 unique rows")
    ok(len(val) == len(va) == 100, "VAL has 100 unique rows")
    ok(not tr & va, "TRAIN and VAL disjoint")
    ok(tr | va == set(range(1000)), "TRAIN + VAL = full rows 0-999")
    ok(set(fit["indices"]) <= tr, "FIT (full 0-99) is inside TRAIN")
    ok(va <= set(legacy["indices"]), "VAL is inside the historical holdout")
    ok(len(set(legacy["indices"]) & tr) == 800,
       "HOLDOUT-LEGACY holds 800 TRAIN rows (why --panel holdout is refused)")
    ok(not set(fit["seeds"]) & set(legacy["seeds"]), "FIT and HOLDOUT-LEGACY seeds disjoint")
    print("PANELS OK" if not bad else f"{bad} PANEL CHECK FAILURES")
    return bad


def nets(panel: str, n: int | None = None) -> List[tuple]:
    """[(split, index, seed, name), ...] for `panel`, truncated to `n`."""
    d = load(panel)
    rows = list(zip(d["indices"], d["seeds"], d["names"]))
    if n is not None:
        rows = rows[:n]
    return [(d["split"], i, s, nm) for i, s, nm in rows]


def verify_hashes(limit: int | None = None) -> int:
    """Hash the weight tensors of DEV and FIT and confirm no hash is shared.

    Seed identity says these are different networks. This says the *weights* are
    different, which is the claim that actually matters for FIT.
    """
    import numpy as np  # noqa: PLC0415
    rd = _real_data()
    seen: Dict[str, str] = {}
    bad = 0
    for panel in ("dev", "fit"):
        d = load(panel)
        idx = d["indices"][:limit] if limit else d["indices"]
        digests = []
        for k, i in enumerate(idx):
            w, _truth, meta = rd.get(i, d["split"])
            h = hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest()
            digests.append(h)
            tag = f"{panel}[{i}] {meta['mlp_name']}"
            if h in seen:
                print(f"COLLISION: {tag} has the same weights as {seen[h]}")
                bad += 1
            seen[h] = tag
            if (k + 1) % 10 == 0:
                print(f"  {panel}: {k + 1}/{len(idx)} hashed", flush=True)
        d["weight_sha256"] = digests
        d["weight_sha256_covers"] = len(digests)
        (PANEL_DIR / f"{panel}.json").write_text(
            json.dumps(d, indent=1) + "\n", encoding="utf-8")
        print(f"{panel}: {len(digests)} weight hashes recorded")
    print("DISJOINT by weight hash" if not bad else f"{bad} COLLISIONS")
    return bad


def show() -> None:
    for name in BUILT:
        p = PANEL_DIR / f"{name}.json"
        if not p.exists():
            print(f"{name:8s} not built")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        hashed = d.get("weight_sha256_covers", 0)
        print(f"{name:8s} split={d['split']:5s} n={len(d['indices']):4d} "
              f"weight-hashed={hashed:4d}  {d['purpose']}")
        print(f"         first={d['names'][0]}  last={d['names'][-1]}")
    dev = PANEL_DIR / "dev.json"
    if dev.exists():
        print(f"\nmini/full overlap: "
              f"{json.loads(dev.read_text(encoding='utf-8'))['mini_full_overlap']} nets")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify-hashes", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="hash only the first N of each panel (a smoke check)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="assert panel membership and disjointness (DEV/TRAIN/VAL)")
    a = ap.parse_args(argv)
    if a.check:
        return 1 if check() else 0
    if a.build:
        out = build()
        for k, v in out.items():
            print(f"built {k}: {len(v['indices'])} nets from {v['split']}")
    if a.verify_hashes:
        if verify_hashes(a.limit):
            return 1
    if a.show or not (a.build or a.verify_hashes):
        show()
    return 0


if __name__ == "__main__":
    sys.exit(main())

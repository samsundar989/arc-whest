"""Build the machine-readable OFFICIAL Phase-2 manifest.

P2-CP0 deliverable #1. Read-only with respect to the repository's evidence:
it fetches the organisers' own published artifacts, hashes them, extracts the
constants that the challenge page states *staler* than the dataset does, and
records exactly what is known, what is a package default, and what is unknown.

Three rules this file exists to enforce:

1.  **The dataset card and `metadata.json` outrank the challenge page.** The
    page still carries a `max(0.5, .)` worked example and a Phase-1 timeline.
2.  **Both stack version pairs are preserved.** The Phase-2 metadata carries
    provenance versions (whestbench 0.16.0 / flopscope 0.12.0) *and* legacy
    `*_version_reported` values (0.14.0 / 0.10.0). Neither is silently chosen.
3.  **Package parity is a measured status, not an assumption.** The manifest
    records what is installed here, what is public on PyPI, and therefore
    whether the exact bake stack can be assembled at all.  If it cannot, the
    manifest carries ``package_parity.status == "BLOCKED"`` and every consumer
    is expected to stamp that limitation onto its own artifacts.

Run:  uv run python local_sim/phase2_manifest.py
      uv run python local_sim/phase2_manifest.py --offline   (no network; needs cache)

Writes: experiments/2026-08-21_phase2_manifest/phase2_manifest.json
        experiments/2026-08-21_phase2_manifest/cache/*        (raw fetched bytes)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

OUT_DIR = os.path.join(_ROOT, "experiments", "2026-08-21_phase2_manifest")
CACHE = os.path.join(OUT_DIR, "cache")
OUT = os.path.join(OUT_DIR, "phase2_manifest.json")

REPO = "aicrowd/arc-whestbench-public-2026"
TAG = "v2-phase2"

SOURCES = {
    "metadata": f"https://huggingface.co/datasets/{REPO}/resolve/{TAG}/metadata.json",
    "card": f"https://huggingface.co/datasets/{REPO}/resolve/{TAG}/README.md",
    "refs": f"https://huggingface.co/api/datasets/{REPO}/refs",
    "pypi_whestbench": "https://pypi.org/pypi/whestbench/json",
    "pypi_flopscope": "https://pypi.org/pypi/flopscope/json",
}

# The two package versions the Phase-2 bake reports in `hardware_fingerprints`.
BAKE_STACK = {"whestbench": "0.16.0", "flopscope": "0.12.0"}


_SUP = {"\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3",
        "\u2074": "4", "\u2075": "5", "\u2076": "6", "\u2077": "7",
        "\u2078": "8", "\u2079": "9"}


def _parse_budget(card: str, fwd_flops: int):
    """Read B per MLP out of the card's Versions table row for this revision.

    Returns (budget_int, expression_string). Raises rather than guessing -- a
    manifest that silently falls back to a remembered constant is exactly the
    failure mode this parser exists to remove.
    """
    row = None
    for line in card.splitlines():
        if "`v2-phase2`" in line and line.lstrip().startswith("|"):
            row = line
            break
    if row is None:
        raise SystemExit("REFUSED: no v2-phase2 row in the card's Versions table")

    plain = "".join(_SUP.get(ch, ch) for ch in row)
    m = re.search(r"\*\*2(\d\d)\s", plain) or re.search(r"\b2\^?(\d\d)\b", plain)
    if not m:
        raise SystemExit(f"REFUSED: no power-of-two budget in the card row: {row!r}")
    exp = int(m.group(1))
    budget = 2 ** exp

    # Cross-check against the card's own "Effective MC samples" column: the
    # card states B / forward_flops_per_sample, so the two must agree exactly.
    eff = re.findall(r"\*\*([\d,]{5,})\*\*", plain)
    stated = {int(x.replace(",", "")) for x in eff}
    if budget // fwd_flops not in stated:
        raise SystemExit(
            f"REFUSED: parsed B = 2**{exp} implies "
            f"{budget // fwd_flops:,} effective samples, which is not among the "
            f"card's own bolded figures {sorted(stated)}. The card changed shape "
            "or the parse is wrong; resolve it deliberately."
        )
    return budget, f"2**{exp}"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fetch(name: str, url: str, offline: bool) -> bytes:
    path = os.path.join(CACHE, name + ".raw")
    if offline:
        with open(path, "rb") as fh:
            return fh.read()
    import httpx

    r = httpx.get(url, timeout=90, follow_redirects=True)
    r.raise_for_status()
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(r.content)
    return r.content


def _semver(v: str) -> tuple:
    core = v.split("rc")[0]
    return tuple(int(x) if x.isdigit() else 0 for x in core.split("."))


def _installed() -> dict:
    import importlib.metadata as md

    out = {}
    for p in ("whestbench", "flopscope", "numpy"):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    return out


def _card_constants(card: str) -> dict:
    """Pull the constants the CARD publishes that metadata.json does not."""
    out: dict = {}
    out["biases"] = "none" if "no biases" in card else "UNKNOWN"
    out["separate_output_layer"] = (
        "none" if "no separate linear output layer" in card else "UNKNOWN"
    )
    out["weight_init"] = (
        "N(0, 2/width) iid (He)" if "N(0, 2/width)" in card else "UNKNOWN"
    )
    out["input_distribution"] = "x ~ N(0, I)" if "N(0, I)" in card else "UNKNOWN"
    m = re.search(r"score is\s*.mse_final\s*[^m]*max\(([0-9.]+),", card)
    out["score_formula"] = (
        "mse_final * max(0.1, C_m / B_m)" if m else "UNKNOWN"
    )
    out["multiplier_floor"] = float(m.group(1)) if m else None
    out["gt_mse_identity"] = (
        "avg_variance / N" if "avg_variance / N" in card else "UNKNOWN"
    )
    out["layer_recurrence"] = (
        "h_l = max(0, h_{l-1} @ W_l), h_0 = x  (row vector on the LEFT)"
        if "h_{l-1}(x) @ W_l" in card
        else "UNKNOWN"
    )
    return out


def build(offline: bool = False) -> dict:
    raw = {k: _fetch(k, u, offline) for k, u in SOURCES.items()}
    meta = json.loads(raw["metadata"].decode("utf-8"))
    card = raw["card"].decode("utf-8")
    refs = json.loads(raw["refs"].decode("utf-8"))
    cc = _card_constants(card)

    commit = next(
        (t["targetCommit"] for t in refs.get("tags", []) if t["name"] == TAG), None
    )

    width = int(meta["width"])
    depth = int(meta["depth"])
    fwd_flops = 2 * depth * width * width

    # B is PARSED from the card's Versions table, not hardcoded. A hardcoded
    # 2**41 would survive a re-bake that moved the budget, while `width` and
    # `depth` (parsed from metadata.json) would not -- an asymmetry an
    # independent Skeptic flagged. The card writes it as an exponent with a
    # decimal gloss, e.g. "**2⁴¹ ≈ 2.20 × 10¹²".
    budget, budget_expr = _parse_budget(card, fwd_flops)

    # ---- both stack version pairs, preserved and explained ------------------
    fps = meta.get("hardware_fingerprints", []) or []
    fp = fps[0] if fps else {}
    stack = {
        "provenance_versions": {
            "whestbench": fp.get("whestbench_version"),
            "flopscope": fp.get("flopscope_version"),
            "meaning": (
                "the versions the Phase-2 bake actually ran under, per "
                "metadata.hardware_fingerprints[*] and the card's Provenance "
                "section"
            ),
        },
        "legacy_version_reported": {
            "whestbench": fp.get("whestbench_version_reported"),
            "flopscope": fp.get("flopscope_version_reported"),
            "meaning": (
                "legacy fields retained beside the newer provenance keys. They "
                "equal the PHASE-1 stack (0.14.0 / 0.10.0). Unresolved: whether "
                "they are a stale carry-over or a statement about the scoring "
                "stack. Only a graded Phase-2 report's runtime_environment "
                "settles it. DO NOT silently pick whichever pair is convenient."
            ),
        },
        "top_level_hardware": {
            "flopscope_version": meta.get("hardware", {}).get("flopscope_version"),
            "flopscope_version_reported": meta.get("hardware", {}).get(
                "flopscope_version_reported"
            ),
            "note": "metadata.hardware (the merge host) carries the same split.",
        },
    }

    # ---- package parity: measured, not assumed ------------------------------
    pypi = {}
    for pkg in ("whestbench", "flopscope"):
        d = json.loads(raw["pypi_" + pkg].decode("utf-8"))
        rels = sorted(d["releases"].keys(), key=_semver)
        pypi[pkg] = {
            "latest": d["info"]["version"],
            "all_releases": rels,
            "requires_dist": d["info"].get("requires_dist"),
            "bake_version_public": BAKE_STACK[pkg] in rels,
        }

    blockers = [
        f"{p} {BAKE_STACK[p]} is not published on PyPI (latest {pypi[p]['latest']})"
        for p in ("whestbench", "flopscope")
        if not pypi[p]["bake_version_public"]
    ]
    # A second, independent blocker: the newest public whestbench may pin AWAY
    # from the bake's flopscope, so no public PAIR reproduces the stack.
    wb_pin = None
    for req in pypi["whestbench"].get("requires_dist") or []:
        if req.split()[0].startswith("flopscope"):
            wb_pin = req
            break
    if wb_pin and f"<{BAKE_STACK['flopscope']}" in wb_pin:
        blockers.append(
            f"public whestbench {pypi['whestbench']['latest']} pins "
            f"'{wb_pin}', which EXCLUDES the bake's flopscope "
            f"{BAKE_STACK['flopscope']}; no public pair reproduces the bake stack"
        )

    # The DECISIVE check, added 2026-08-24 (P2-P0). Everything above asks whether
    # the bake stack is publicly AVAILABLE. That is the wrong question once it is
    # installed: what a measurement depends on is the stack it actually ran under.
    # `AVAILABLE` and `ACHIEVED` are different states and conflating them is how a
    # manifest reports parity that no environment has.
    _inst = _installed()
    installed_matches_bake = all(
        str(_inst.get(p, "")) == BAKE_STACK[p] for p in ("whestbench", "flopscope")
    )

    parity = {
        "status": ("ACHIEVED" if installed_matches_bake
                   else ("BLOCKED" if blockers else "AVAILABLE")),
        "installed_matches_bake": installed_matches_bake,
        "bake_stack": BAKE_STACK,
        "installed_here": _inst,
        "pypi": pypi,
        "public_whestbench_flopscope_pin": wb_pin,
        "blockers": blockers,
        "limitation_stamp": (
            (
                "EXACT PACKAGE PARITY WITH THE PHASE-2 BAKE STACK IS BLOCKED. "
                "Every artifact produced while this holds must carry this field. "
                "Counted FLOPs measured under a different flopscope release are "
                "NOT the grader's counted FLOPs. Re-run parity the moment "
                f"whestbench {BAKE_STACK['whestbench']} is published."
            )
            if blockers and not installed_matches_bake
            else None
        ),
    }

    manifest = {
        "artifact": "phase2_manifest",
        "artifact_version": 1,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "generator": "local_sim/phase2_manifest.py",
        "authority": (
            "The dataset card and metadata.json at the pinned commit are "
            "AUTHORITATIVE. The AIcrowd challenge page is known-stale (Phase-1 "
            "timeline, a max(0.5, .) worked example) and is not a source here."
        ),
        "dataset": {
            "repo": REPO,
            "revision_tag": TAG,
            "commit": commit,
            "created_at_utc": meta.get("created_at_utc"),
            "schema_version": meta.get("schema_version"),
            "seed_protocol": meta.get("seed_protocol"),
            "backend": meta.get("backend"),
            "partials_count": meta.get("partials_count"),
            "default_split": meta.get("default_split"),
        },
        "shape": {
            "width": width,
            "depth": depth,
            "n_weight_matrices": depth,
            "biases": cc["biases"],
            "separate_output_layer": cc["separate_output_layer"],
            "weight_init": cc["weight_init"],
            "input_distribution": cc["input_distribution"],
            "layer_recurrence": cc["layer_recurrence"],
            "L_over_n": depth / width,
            "L_over_n_phase1": 32 / 256,
        },
        "truth": {
            "n_samples_per_mlp": int(meta["n_samples"]),
            "accumulation": "float64 sums, divided by N, downcast to float32",
            "gt_mse_identity": cc["gt_mse_identity"],
            "gt_mse_note": (
                "avg_variance is a PUBLISHED per-MLP column, so the public "
                "split's truth-noise floor is exactly computable without a "
                "second replica. Fresh mints still need two replicas."
            ),
        },
        "splits": {
            name: {
                "n_mlps": s["n_mlps"],
                "created_at_utc": s.get("created_at_utc"),
                "config": s.get("config"),
            }
            for name, s in meta["splits"].items()
        },
        "budget": {
            "forward_flops_per_sample": fwd_flops,
            "forward_flops_formula": "2 * depth * width^2",
            "flop_budget_per_mlp": budget,
            "flop_budget_expr": budget_expr,
            "effective_naive_samples": budget // fwd_flops,
            "multiplier_floor": cc["multiplier_floor"],
            "score_formula": cc["score_formula"],
            "source": "dataset card 'Versions' table + score-formula prose",
            "phase1_comparison": {
                "phase1_budget": 2.72e11,
                "phase1_forward_flops_per_sample": 2 * 32 * 256 * 256,
                "budget_growth": budget / 2.72e11,
                "per_sample_cost_growth": fwd_flops / (2 * 32 * 256 * 256),
                "note": (
                    "budget grew ~8.08x against a ~8.00x per-sample cost, so "
                    "the naive-sample operating point is PRESERVED at ~65,536."
                ),
            },
        },
        "stack": stack,
        "package_parity": parity,
        # RESOLVED 2026-08-24 (P2-P0). These were unknown while the environment
        # ran whestbench 0.14, whose defaults are Phase-1's. Installing the bake
        # stack settles every one of them from `budget.PHASE2_ROUND` and
        # `scoring.ContestSpec`, and each is additionally corroborated by the
        # `run_config` block of a real `whest run --json` -- which records what the
        # runner USED, not merely what it declares. The authority is
        # `local_sim/official_config.py`; the values below are its output.
        #
        # They remain PACKAGE-sourced, not GRADED-report-sourced. No graded report
        # we hold publishes them, and that distinction survives the resolution.
        "resolved_from_package": {
            "resolved_on": "2026-08-24",
            "source": "whestbench 0.16.0 budget.PHASE2_ROUND + scoring.ContestSpec",
            "corroborated_by": (
                "experiments/2026-08-24_p2_p0/official_anchor_1row.json run_config"
            ),
            "lambda_residual_price": {
                "value": 0.0,
                "phase1_value": 1e11,
                "status": (
                    "RESOLVED: lambda is 0.0 for Phase 2, so C == F and residual "
                    "wall time does not price into the multiplier by any path. It "
                    "is GATED instead, at 0.4 s per MLP, which is a FAILURE path. "
                    "Note the trap: whestbench 0.16 still exports a name "
                    "LAMBDA_FLOPS_PER_SECOND = 1e11, as a deprecated alias for the "
                    "PHASE-1 rate. Same spelling, different meaning."
                ),
            },
            "setup_timeout_s": {"package_default": 5.0, "observed": 5.0},
            "wall_time_limit_s": {"package_default": 120.0, "observed": 120.0},
            "predict_timeout_s": {
                "package_default": 30.0,
                "observed": None,
                "note": (
                    "the runner waits max(predict_timeout_s, wall + grace) = 125 s, "
                    "so under Phase 2 the 120 s WALL binds and this never does"
                ),
            },
            "residual_wall_time_limit_s": {"package_default": 0.4, "observed": 0.4},
            "memory_limit_mb": {
                "package_default": 8192,
                "observed": 8192,
                "note": (
                    "RLIMIT_AS under --runner subprocess; the PHASE-1 limit of "
                    "65,536 MB already killed v2.5 at a 3.9 GB peak, and this one "
                    "is EIGHT TIMES tighter against sixteen times the weight bytes. "
                    "It cannot be enforced on Windows at all (no `resource` "
                    "module), so it is an accounting constraint here, not a test: "
                    "see experiments/2026-08-24_p2_p0/deployment_envelope.json. "
                    "At width 1024 a dense float64 covariance is 8.4 MB per layer "
                    "-- but a K3 tensor would be 8.6 GB, i.e. over the limit on its "
                    "own."
                ),
            },
            "n_scored_public_mlps": {
                "phase1": 50,
                "phase2": None,
                "note": (
                    "Phase 1 scored 50 of 100 mini MLPs publicly. Unverified for "
                    "Phase 2."
                ),
            },
            "how_to_resolve": (
                "a graded Phase-2 report's config_snapshot + runtime_environment "
                "+ per-MLP telemetry. Costs one submission."
            ),
        },
        "round": {
            "id": 1429,
            "name": "Phase 2",
            "note": (
                "polled separately by local_sim/phase2_checklist.py; dataset "
                "publication and round activation are SEPARATE events."
            ),
        },
        "source_hashes": {k: _sha(v) for k, v in raw.items()},
        "source_urls": SOURCES,
    }
    return manifest


def load() -> dict:
    """Read the manifest from disk. The one accessor other scripts should use."""
    with open(OUT, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--offline",
        action="store_true",
        help="read the cached raw sources instead of fetching",
    )
    ap.add_argument(
        "--allow-rebake",
        action="store_true",
        help="accept that the v2-phase2 tag now points at a different commit",
    )
    a = ap.parse_args()

    m = build(offline=a.offline)
    os.makedirs(OUT_DIR, exist_ok=True)

    # The manifest is a LIVE POLL, so rebuilding it is normal and overwriting is
    # correct -- unlike an evidence artifact, which is immutable. The one thing
    # that must never be silent is the dataset moving underneath us: a new
    # commit at the same tag is a re-bake, and every shape-dependent result in
    # the repository is then suspect.
    if os.path.exists(OUT) and not a.allow_rebake:
        with open(OUT, encoding="utf-8") as fh:
            old = json.load(fh)
        if old["dataset"]["commit"] != m["dataset"]["commit"]:
            raise SystemExit(
                "REFUSED: the v2-phase2 tag has MOVED.\n"
                f"  manifest on disk : {old['dataset']['commit']}\n"
                f"  live tag now     : {m['dataset']['commit']}\n"
                "The dataset was re-baked or re-tagged. Every 1024 x 16 result "
                "in this repository was measured against the old commit and "
                "must be re-checked. Rerun with --allow-rebake once that is a "
                "deliberate decision, and record it in experiments/LOG.md."
            )
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(m, indent=2) + "\n")

    p = m["package_parity"]
    b = m["budget"]
    s = m["shape"]
    print("=" * 72)
    print(f"OFFICIAL PHASE-2 MANIFEST  ->  {os.path.relpath(OUT, _ROOT)}")
    print("=" * 72)
    print(f"  dataset      {m['dataset']['repo']}@{m['dataset']['revision_tag']}")
    print(f"  commit       {m['dataset']['commit']}")
    print(
        f"  shape        {s['width']} x {s['depth']}"
        f"   biases={s['biases']}"
        f"   output_layer={s['separate_output_layer']}"
        f"   L/n={s['L_over_n']:.6f}"
    )
    print(
        f"  truth        N = {m['truth']['n_samples_per_mlp']:,}"
        f"   (gt MSE = {m['truth']['gt_mse_identity']})"
    )
    print(
        "  splits       "
        + "  ".join(f"{k}={v['n_mlps']}" for k, v in m["splits"].items())
    )
    print(
        f"  budget       B = {b['flop_budget_per_mlp']:,} = {b['flop_budget_expr']}"
        f"   floor = {b['multiplier_floor']}"
    )
    print(
        f"  per sample   {b['forward_flops_per_sample']:,} FLOPs  ->  "
        f"{b['effective_naive_samples']:,} naive samples"
    )
    print(
        f"  bake stack   whestbench {m['stack']['provenance_versions']['whestbench']}"
        f" / flopscope {m['stack']['provenance_versions']['flopscope']}"
        f"   (legacy *_reported: "
        f"{m['stack']['legacy_version_reported']['whestbench']}"
        f" / {m['stack']['legacy_version_reported']['flopscope']})"
    )
    print(
        f"  installed    whestbench {p['installed_here']['whestbench']}"
        f" / flopscope {p['installed_here']['flopscope']}"
    )
    print()
    print(f"  PACKAGE PARITY: {p['status']}")
    for bl in p["blockers"]:
        print(f"    - {bl}")
    if p["limitation_stamp"]:
        print("    STAMP: " + p["limitation_stamp"])
    print()
    print("  RESOLVED from the bake stack (package-sourced, not graded-report-sourced):")
    print("    lambda = 0.0   -> C == F; residual is GATED at 0.4 s, not priced")
    print("    wall 120 s   residual cap 0.4 s   memory 8,192 MB   setup 5 s")
    print()
    print("  STILL UNKNOWN, and not in the dataset:")
    print("    n scored public MLPs                          -> unverified for Phase 2")
    print("    every limit above as a GRADED fact            -> no graded report publishes them")
    print("    the 8,192 MB limit is UNENFORCEABLE on Windows -> accounting only")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare the stored 2026-08-21 mint-parity evidence row (taken at commit
ce926b7d) against the transcript of the identical check re-run at 6a416fa5.

This is the FUNCTIONAL half of WP0A's claim. The TSV proves the source objects
are byte-identical by hash; this proves that reading one of them at the new
commit reproduces the same weights and the same derived statistics.
"""
import json
import os
import re

ROOT = r"c:\Users\samsu\Documents\arc-whest"
STORED = os.path.join(ROOT, "experiments", "2026-08-21_phase2_manifest",
                      "phase2_mint_parity_shard1_rows4.json")
TXT = os.path.join(ROOT, "experiments", "2026-08-31_wp0a_manifest",
                   "mint_parity_shard1_rows4_at_6a416fa5.txt")
OUT = os.path.join(ROOT, "experiments", "2026-08-31_wp0a_manifest",
                   "mint_parity_source_equivalence.json")

stored = json.load(open(STORED, encoding="utf-8"))
txt = open(TXT, encoding="utf-8").read()


def grab(pat, cast=float):
    m = re.search(pat, txt)
    return cast(m.group(1)) if m else None


rerun = {
    "revision_commit": grab(r"@([0-9a-f]{12})", str),
    "n_rows_checked": grab(r"rows checked: (\d+) of", int),
    "weights_bit_identical": grab(r"published Phase-2 bytes : (\d+)/", int),
    "shape_contract_ok": grab(r"no output layer\)\s+: (\d+)/", int),
    "layer1_closed_form_max_rel": grab(r"worst max relative error\s+: ([\d.e+-]+)"),
    "layer1_z_rms_mean": grab(r"rms z\s+\[bar 0\.9, 1\.1\]\s+: ([\d.]+)"),
    "layer1_z_abs_max": grab(r"worst \|z\| over [\d,]+ neurons\s+: ([\d.]+)"),
    "layer1_z_max_pvalue_exact": grab(r"exact P\(max >= z\) = ([\d.]+)"),
    "gt_noise_floor_exact": grab(r"noise floor, EXACT \(avg_variance / N\)\s+: ([\d.e+-]+)"),
    "max_published_seed_bits": grab(r"widest published mlp_seed\s+: (\d+) bits", int),
    "verdict": "PROVEN" if "MINT PARITY WITH THE PHASE-2 STACK IS PROVEN." in txt else "NOT PROVEN",
}

# The transcript prints 6 or 7 significant figures; compare at that precision,
# and say so rather than claiming bitwise identity the artifact cannot support.
comparisons = []
for key, tol in (
    ("n_rows_checked", 0), ("weights_bit_identical", 0), ("shape_contract_ok", 0),
    ("max_published_seed_bits", 0),
    # Tolerances come from the PRINT FORMAT of local_sim/phase2_mint_parity.py,
    # not from what happens to pass. `%.3e` keeps 4 significant figures, so the
    # relative rounding error is up to 5e-4; `%.4f` on a value near 1.0 is 5e-5;
    # `%.2f` on a value near 4.3 is 1.2e-3; `%.3f` on 0.057 is 8.8e-3; `%.6e` is
    # 5e-7. Each bound below is that figure, rounded up one decade.
    ("layer1_closed_form_max_rel", 1e-3),   # printed %.3e
    ("layer1_z_rms_mean", 1e-4),            # printed %.4f
    ("layer1_z_abs_max", 1e-2),             # printed %.2f
    ("layer1_z_max_pvalue_exact", 1e-2),    # printed %.3f
    ("gt_noise_floor_exact", 1e-6),         # printed %.6e
):
    a, b = stored.get(key), rerun.get(key)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        rel = 0.0 if a == b else abs(a - b) / max(abs(a), 1e-300)
        ok = rel <= tol
    else:
        rel, ok = None, a == b
    comparisons.append({"field": key, "stored_at_ce926b7d": a,
                        "rerun_at_6a416fa5": b, "rel_diff": rel,
                        "printed_precision_tolerance": tol, "agrees": ok})

record = {
    "artifact": "wp0a_mint_parity_source_equivalence",
    "artifact_version": 1,
    "question": ("Does reading the SAME published shard at the NEW v2-phase2 "
                 "commit reproduce the SAME mint-parity result as the stored "
                 "2026-08-21 evidence row taken at the old commit?"),
    "stored_evidence": os.path.relpath(STORED, ROOT).replace("\\", "/"),
    "rerun_transcript": os.path.relpath(TXT, ROOT).replace("\\", "/"),
    "stored_revision": stored["dataset"]["revision_commit"],
    "rerun_revision_prefix": rerun["revision_commit"],
    "shard": stored["dataset"]["shard"],
    "comparisons": comparisons,
    "all_agree": all(c["agrees"] for c in comparisons),
    "verdicts_agree": stored["verdict"] == rerun["verdict"] == "PROVEN",
    "scope": (
        "Numeric fields are compared at the PRINTED precision of the rerun "
        "transcript, not bitwise: the rerun's own JSON row was correctly "
        "REFUSED by the immutability guard because the stored row differs in "
        "its recorded revision and in its package-parity stamp, which were "
        "BLOCKED on 2026-08-21 and are ACHIEVED now. The weights themselves are "
        "compared bitwise BY THE CHECK ITSELF ('weights bit-identical to the "
        "published Phase-2 bytes: 4/4'), against canonical_mint's output."
    ),
    "what_this_does_not_show": (
        "4 of 16 rows of 1 of 7 mini shards. It corroborates the 70-object hash "
        "comparison functionally on a sample; it does not re-derive it."
    ),
}
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(json.dumps(record, indent=2) + "\n")

for c in comparisons:
    print("  %-32s stored=%-22s rerun=%-22s %s"
          % (c["field"], c["stored_at_ce926b7d"], c["rerun_at_6a416fa5"],
             "AGREE" if c["agrees"] else "DIFFER"))
print()
print("all_agree:", record["all_agree"], " verdicts_agree:", record["verdicts_agree"])
print("written:", os.path.relpath(OUT, ROOT))

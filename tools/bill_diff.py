"""Diff two `tools/bill_meter.py` outputs per op family (S10.1: flopscope 0.12.1 vs 0.12.2).

    python tools/bill_diff.py <a.json> <b.json> [--call 1]

Prints total FLOPs / ops for every call of both files, then the per-op-name table for `--call` (default: the
first call), listing every family whose FLOPs or calls moved.
"""
import argparse
import json

U = 2 * 1024 ** 3 - 1024 ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--call", type=int, default=0)
    x = ap.parse_args()
    A, B = json.load(open(x.a)), json.load(open(x.b))
    assert A["sha256"] == B["sha256"], "different candidates"
    print(f"candidate {A['sha256'][:8]}  {A['flopscope_version']} -> {B['flopscope_version']}")
    for ra, rb in zip(A["rows"], B["rows"]):
        d = rb["flops"] - ra["flops"]
        print(f"  call {ra['call']} {ra['net']:<20} F {ra['flops']:>15,} -> {rb['flops']:>15,}  "
              f"d {d:+,} ({100.0 * d / ra['flops']:+.4f} %)  ops {ra['ops']} -> {rb['ops']}  "
              f"preds {'same' if ra['pred_sha'] == rb['pred_sha'] else 'DIFFER'}")
    oa, ob = A["rows"][x.call]["operations"], B["rows"][x.call]["operations"]
    moved = []
    for k in sorted(set(oa) | set(ob)):
        fa, fb = oa.get(k, {}).get("flops", 0.0), ob.get(k, {}).get("flops", 0.0)
        ca, cb = oa.get(k, {}).get("calls", 0.0), ob.get(k, {}).get("calls", 0.0)
        if fa != fb or ca != cb:
            moved.append((k, fa, fb, ca, cb))
    print(f"families that moved (call {x.call}): {len(moved)} of {len(set(oa) | set(ob))}")
    for k, fa, fb, ca, cb in sorted(moved, key=lambda r: -abs(r[2] - r[1])):
        print(f"  {k:<28} F {fa:>16,.0f} -> {fb:>16,.0f} ({(fb - fa) / U:+.4f} u)  calls {ca:.0f} -> {cb:.0f}")


if __name__ == "__main__":
    main()

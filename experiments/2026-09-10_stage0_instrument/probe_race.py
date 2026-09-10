"""Two concurrent claimers on a SINGLE_USE block. Does the ledger refuse both but one?

Drives a SACRIFICIAL ledger file in the scratchpad. Touches nothing real.
"""
import json
import os
import sys
import tempfile
import threading

sys.path.append(os.path.join(os.getcwd(), "local_sim"))
import seed_ledger as sl  # noqa: E402

BLOCK = "P08"                      # SINGLE_USE, per seed_ledger.TIERS
LO, HI = sl.block_span(BLOCK)

results = {"ok": 0, "refused": 0, "errors": []}
lock = threading.Lock()


def claimer(path, i, barrier):
    barrier.wait()
    try:
        sl.claim_range(BLOCK, LO, 8, f"racer-{i}", ts="2026-09-10T00:00:00Z",
                       path=path)
        with lock:
            results["ok"] += 1
    except PermissionError:
        with lock:
            results["refused"] += 1
    except Exception as e:  # noqa: BLE001
        with lock:
            results["errors"].append(repr(e)[:120])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    d = tempfile.mkdtemp(prefix="ledger_race_")
    path = os.path.join(d, "seed_ledger.jsonl")
    open(path, "w").close()
    barrier = threading.Barrier(n)
    ts = [threading.Thread(target=claimer, args=(path, i, barrier))
          for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    lines = [ln for ln in open(path, encoding="utf-8") if ln.strip()]
    claims = [json.loads(ln) for ln in lines]
    print(json.dumps({
        "concurrent_claimers": n,
        "claims_that_SUCCEEDED": results["ok"],
        "claims_REFUSED": results["refused"],
        "lines_actually_written_to_the_ledger": len(lines),
        "distinct_consumers_on_the_block": sorted({c["consumer"] for c in claims}),
        "SINGLE_USE_violated": len(lines) > 1,
        "errors": results["errors"][:3],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

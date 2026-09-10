"""The read register refuses a second look, and counts a crashed one.

Stage 0 repair R3. `docs/research/2026-09-10_validation_audit.md` finding 3.

Two defects, both demonstrated against the code at `0fe164c` before the repair:

  1. `seed_ledger.claim_range` had a read-decide-append window with no lock.
     Twelve concurrent claimers on ONE `SINGLE_USE` block: twelve succeeded,
     zero refused, every run
     (`experiments/2026-09-10_stage0_instrument/defects_before_ledger_race.json`).
  2. Nothing counted READS at all. `SINGLE_USE` protects a block from being
     claimed twice; P08 was claimed once and read three times.

Everything below drives SACRIFICIAL registers and ledgers in a temp directory.
No real block, artifact, ledger or vault is touched.

Run: uv run python local_sim/test_truth_reads.py
"""

import json
import os
import sys
import tempfile
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import seed_ledger as sl  # noqa: E402
import truth_reads as tr  # noqa: E402

RESULTS = []
TS = "2026-09-10T00:00:00Z"


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<66} {'ok ' if ok else 'FAILED'}  {detail}")
    return ok


def head(t):
    print(f"\n{t}\n{'-' * len(t)}")


def _sandbox():
    d = tempfile.mkdtemp(prefix="truth_reads_")
    reads = os.path.join(d, "truth_reads.jsonl")
    ledg = os.path.join(d, "seed_ledger.jsonl")
    open(ledg, "w").close()
    art = os.path.join(d, "truth_sacrificial.npz")
    with open(art, "wb") as fh:
        fh.write(b"not really an npz, and never loaded by these tests")
    return d, reads, ledg, art


#: A SINGLE_USE block, so cap 1. P08 is the real one; the tests use its tier
#: through the real registry but never its real artifact or register.
SU_BLOCK = "P08"
DESIGN_BLOCK = "P01"


def t_cap():
    head("1. a single-use block permits exactly one read")
    d, reads, ledg, art = _sandbox()
    kw = dict(consumer="test", purpose="the one read", ts=TS,
              path=reads, ledger_path=ledg)
    check("the register starts EMPTY", tr.reads(reads) == [],
          "prediction 7: no machine-readable read log existed")
    with tr.reserve(SU_BLOCK, art, **kw):
        pass
    check("the first read is permitted", tr.spent(SU_BLOCK, reads) == 1, "")
    try:
        with tr.reserve(SU_BLOCK, art, **kw):
            pass
        ok = False
        why = "IT WAS PERMITTED"
    except tr.ReadRefused as e:
        ok, why = True, str(e)
    check("the SECOND read is REFUSED", ok, why[:56])
    check("...and the refusal names the prior consumer and its status",
          ok and "test" in why and "completed" in why,
          "a refusal that will not say what spent the block is not auditable")
    check("...and refusing wrote no extra line",
          tr.spent(SU_BLOCK, reads) == 1, f"{tr.spent(SU_BLOCK, reads)} spent")


def t_artifact_identity_binding():
    head("2. a filename's block identity cannot be overridden")
    d, reads, ledg, art = _sandbox()
    named = os.path.join(d, "truth_P08.npz")
    os.replace(art, named)
    try:
        tr.resolve_block(named, "P01")
        mismatch_refused = False
    except tr.ReadRefused:
        mismatch_refused = True
    check("truth_P08.npz cannot be presented as P01", mismatch_refused)
    check("truth_P08.npz resolves to P08 when no id is supplied",
          tr.resolve_block(named) == "P08")
    check("truth-bearing feat_P08.npz also resolves to P08",
          tr.resolve_block(os.path.join(d, "feat_P08.npz")) == "P08")
    check("a vault filename resolves only to V",
          tr.resolve_block(os.path.join(d, "vault_v1.npz")) == "V")
    try:
        tr.resolve_block(os.path.join(d, "vault_v1.npz"), "P01")
        vault_refused = False
    except tr.ReadRefused:
        vault_refused = True
    check("a vault artifact cannot be presented as a design block", vault_refused)
    check("a generic filename may use an explicit registered block",
          tr.resolve_block(os.path.join(d, "generic.npz"), "P01") == "P01")
    try:
        tr.resolve_block(os.path.join(d, "generic.npz"), "P99")
        unknown_refused = False
    except tr.ReadRefused:
        unknown_refused = True
    check("an unknown block is refused", unknown_refused)
    check("identity refusals do not spend a read", tr.reads(reads) == [])


def t_aborted_still_counts():
    head("2. a crashed read still counts — the labels were already exposed")
    d, reads, ledg, art = _sandbox()
    kw = dict(consumer="crasher", purpose="dies mid-read", ts=TS,
              path=reads, ledger_path=ledg)
    try:
        with tr.reserve(SU_BLOCK, art, **kw):
            raise RuntimeError("the instrument fell over after np.load returned")
    except RuntimeError:
        pass
    lines = tr.reads(reads)
    check("the reservation was written BEFORE the body ran",
          any(r.status == "reserved" for r in lines), "reserve-then-look")
    check("...and the crash is recorded as ABORTED",
          any(r.status == "aborted" for r in lines), "")
    check("an aborted read CONSUMES the cap",
          tr.spent(SU_BLOCK, reads) == 1,
          "otherwise 'my script raised' is an unlimited supply of looks")
    try:
        with tr.reserve(SU_BLOCK, art, **kw):
            pass
        ok = False
    except tr.ReadRefused:
        ok = True
    check("...so the next read is refused", ok, "")


def t_uncapped_tier_is_still_counted():
    head("3. a reusable design block is counted, not capped")
    d, reads, ledg, art = _sandbox()
    kw = dict(consumer="design-lab", purpose="exploration", ts=TS,
              path=reads, ledger_path=ledg)
    for _ in range(4):
        with tr.reserve(DESIGN_BLOCK, art, **kw):
            pass
    st = tr.status(DESIGN_BLOCK, reads)
    check("four reads of a DESIGN block are permitted", st["spent"] == 4,
          f"tier {st['tier']}, cap {st['cap']}")
    check("...and none of them is 'exhausted'", st["exhausted"] is False,
          "how much a design corpus has been mined is a different question "
          "from whether it was allowed")


def t_provenance_recorded():
    head("4. every line says who looked, at what, with which instrument")
    d, reads, ledg, art = _sandbox()
    inst = os.path.join(d, "instrument.py")
    with open(inst, "w") as fh:
        fh.write("# the code that did the reading\n")
    with tr.reserve(DESIGN_BLOCK, art, consumer="lab/x.py", purpose="a purpose",
                    ts=TS, instrument=inst, result_artifact="out.json",
                    note="a note", path=reads, ledger_path=ledg):
        pass
    r = tr.reads(reads)[0]
    check("the artifact's sha256 is recorded", len(r.artifact_sha256) == 64,
          r.artifact_sha256[:16] + "...")
    check("the INSTRUMENT's sha256 is recorded", len(r.instrument_sha256) == 64,
          "a repaired instrument cannot be confused with the one that ran")
    check("the consumer, purpose and result artifact are recorded",
          r.consumer == "lab/x.py" and r.purpose == "a purpose"
          and r.result_artifact == "out.json", "")
    check("the register is append-only JSON-Lines, like the claim ledger",
          all(json.loads(ln) for ln in open(reads, encoding="utf-8")), "")


def t_concurrent_reads():
    head("5. two consumers cannot both be handed the LAST read")
    d, reads, ledg, art = _sandbox()
    out = {"ok": 0, "refused": 0, "other": []}
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker(i):
        barrier.wait()
        try:
            with tr.reserve(SU_BLOCK, art, consumer=f"racer-{i}",
                            purpose="race", ts=TS, path=reads,
                            ledger_path=ledg):
                pass
            with lock:
                out["ok"] += 1
        except tr.ReadRefused:
            with lock:
                out["refused"] += 1
        except Exception as e:  # noqa: BLE001
            with lock:
                out["other"].append(repr(e)[:80])

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    check("exactly ONE of eight concurrent readers succeeds", out["ok"] == 1,
          f"{out['ok']} ok, {out['refused']} refused, {len(out['other'])} other")
    check("...and the other seven are REFUSED, not crashed", out["refused"] == 7,
          str(out["other"][:1]))


def t_ledger_race_is_closed():
    head("6. the CLAIM race the register depends on")
    d, reads, ledg, art = _sandbox()
    lo, _hi = sl.block_span(SU_BLOCK)
    out = {"ok": 0, "refused": 0, "other": []}
    lock = threading.Lock()
    barrier = threading.Barrier(12)

    def worker(i):
        barrier.wait()
        try:
            sl.claim_range(SU_BLOCK, lo, 8, f"racer-{i}", ts="2026-09-10",
                           path=ledg)
            with lock:
                out["ok"] += 1
        except PermissionError:
            with lock:
                out["refused"] += 1
        except Exception as e:  # noqa: BLE001
            with lock:
                out["other"].append(repr(e)[:80])

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    n_lines = len([ln for ln in open(ledg, encoding="utf-8") if ln.strip()])
    check("exactly ONE of twelve concurrent claimers succeeds", out["ok"] == 1,
          f"{out['ok']} ok, {out['refused']} refused  (before the lock: 12 ok)")
    check("...and the ledger holds exactly one claim line", n_lines == 1,
          f"{n_lines} lines")
    check("...and no claimer crashed instead of being refused",
          not out["other"], str(out["other"][:1]))


def t_lock_is_released():
    head("7. the lock is released, including when the body raises")
    d, reads, ledg, art = _sandbox()
    lockfile = ledg + ".lock"
    try:
        with sl.ledger_lock(ledg):
            check("the lockfile exists while held", os.path.exists(lockfile), "")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check("...and is gone after the body raises", not os.path.exists(lockfile),
          "a lock leaked by an exception would wedge every later claim")


def t_live_register():
    head("8. the LIVE register — P08's three historical reads")
    st = tr.status("P08")
    check("P08 is SINGLE_USE with a cap of 1", st["tier"] == "SINGLE_USE"
          and st["cap"] == 1, f"tier {st['tier']}, cap {st['cap']}")
    check("...and its register records THREE reads, backfilled",
          st["spent"] == 3 and all(r["backfilled"] for r in st["reads"]),
          "the two broken instrument runs plus the repaired one")
    check("...so P08 reads as EXHAUSTED to anything that asks",
          st["exhausted"] is True,
          "prose in seed_blocks.py became an executable refusal")
    hits = tr.audit_direct_loads()
    check("the direct-load audit finds real call sites", len(hits) > 0,
          f"{len(hits)} sites in local_sim/, none of which reserve")
    check("...and p2_corrector's repaired path is no longer a bypass",
          not any("p2_corrector" in h["file"] for h in hits),
          "the historical P08 reads remain in the register; new runs reserve")


def main():
    print("=" * 96)
    print("THE READ REGISTER — a single-use block is single-use when its LABELS are counted")
    print("=" * 96)
    for fn in (t_cap, t_artifact_identity_binding,
               t_aborted_still_counts, t_uncapped_tier_is_still_counted,
               t_provenance_recorded, t_concurrent_reads, t_ledger_race_is_closed,
               t_lock_is_released, t_live_register):
        fn()
    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

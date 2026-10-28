# L4b — J3 (A4's export + NEW10 + `_SHARED_RANK` 320, refit on fresh nets): NOT RUN, closed on a pre-run check

Handoff: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-21_L4_J3_CHEAP_CONTENT.md`. Session L4, 15:58–16:05 UTC Oct 21.
Decision (Samuel, 16:0x UTC): stop before any pod spend; write it up. **Pod spend: $0** (no pod started or created).

## 0. Handoff prediction and keep rule (as written)

- Prediction: −2.0 % raw vs A4's export (rank 320 alone −1.2 % at step 0; NEW10's refit −0.9 % on A1) ⇒ score-equivalent ≈ −0.15 %.
- Keep: score-equivalent = Δraw + 1.86 % F ≤ −0.6 % (Δraw ≤ −2.46 %, CI excl. 0, ≥ 65/100); canary only if it beats A5s's score-equivalent.

## 1. Why it was not run: NEW10 is already in A4

- A4 = A3's export + 368 fresh-data updates, A3 = A1's export + NEW10 refit (L2). A4 therefore has **NEW10 live and fitted**:
  `2026-10-21_L6/RESULT.md` §planner ("A4 has A3's shape (NEW10, +741 ops)"); A4 vs A3 s368 differ in 35 fitted tables, values only.
- So "NEW10 re-merged on A4's export at default 0" either (a) zeroes A4's fitted NEW10 tables — then the zero control cannot be
  byte-identical to A4's VAL — or (b) keeps them as fitted, and J3 is just **rank 320 + one more fresh-data epoch**.
  Reading (b) is the only consistent one, and it removes the −0.9 % NEW10 term from the prediction: that gain is already in A4.

## 2. Corrected prediction (from measured parts)

| part | measured | source |
|---|---:|---|
| rank 288 → 320, step 0 | −1.20 % raw (85/100) | J step 0 vs A1, `2026-10-21_L4/RESULT.md` §6; L3 ALL-R −1.18 % |
| one more fresh-data epoch | ≈ −0.5 … −0.6 % raw | A4 vs A3 −0.600 % (L6); A5 prediction −0.5 % (L7) |
| NEW10 | 0 (already fitted in A4) | §1 |
| coupling | ≈ 0 | J: parts −3.85 % vs joint −3.90 % (additive) |
| **J3 vs A4's export** | **≈ −1.7 % raw ⇒ score-equiv ≈ +0.15 %** | needs −2.46 % raw; P(clear) ≈ 0.05 |

Even a clear could not ship: the canary condition compares J3 with A5s, which gets the same fresh-data epoch at +0 % F. J3's edge over
A5s is rank 320 alone, −1.2 % raw at +1.86 % F ⇒ **≈ +0.67 % score-equivalent (worse)** — the same verdict J reached for the rank part
("score-equivalent per part ≈ rank +0.67 %", `2026-10-21_L4/RESULT.md`, end).

Clock: the J pod reprovision took ≈ 1.5 h (06:08 start → 07:40 launch) and 276 updates ≈ 2.1 h at 27 s/update, so the earliest
export was ≈ 19:45, after the 19:30 handover to L2.

## 3. Decision

**NO-GO, not run.** Nothing for L2 to bake from L4; the A5s pipeline (L7 → L2) is unaffected. Corpus rows reserved for J3
(shard_ext05/06 rows 1472..2047, disjoint from L7 arm b's 0..1471) stay unused for any post-deadline build.
For the write-up: rank 320 is measured once (on A1, −1.20 % raw, +1.864 % F real bill 292,214,932,424 F) and does not pay for itself
at λ = 0 under score = MSE × F; its transfer to A4 is unmeasured.

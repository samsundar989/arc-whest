# Strategist Packet — 2026-09-04

**Self-contained. Assumes no repository access and no shared chat history.**
Three questions. Everything needed to answer them is below; nothing below
requires you to trust a summary — every number names the artifact it came from.

---

## 0. The situation in six lines

Samuel is a solo entrant (`neopok919`) in the ARC White-Box Estimation Challenge
2026, Phase 2. Given a random He-init ReLU MLP (1024 wide × 16 deep, no biases),
predict every neuron's expected post-ReLU activation under `x ~ N(0, I)`.
Score = `final_layer_MSE × max(0.1, FLOPs / 2^41)`, lower is better. **The prize
is decided by a private re-run of up to two designated submissions on fresh
MLPs**, so public-board rank is not the target. There is a **co-equal $20k
contribution prize** whose deliverable is a write-up.

Deadlines in force: designate by **Oct 10**, close **Oct 17**, write-up
**Oct 24**.

---

## 1. What has happened since your last ruling

Your two rulings were executed. Both produced results, and one produced a
surprise.

### Ruling 1 — the stale certify item, and the canary

Invariant 9 (permutation equivariance) gated on `d_k / scale < 1e-6`, an
absolute bar about ten times float32 epsilon, written when the flow had a
float64 backbone. The live design runs the covariance backbone in f32
deliberately — that is what puts `C/B = 0.0767` on the `0.1` multiplier floor.
The item did not ask a candidate to be correct; it asked it to be expensive.

**Replaced** with "every row's drift, in MSE units, ≤ 1% of the candidate's own
raw MSE". A mutation battery proves the item still rejects nine deliberate
equivariance defects (at 5.3e4×–3.9e7× the bar) while accepting two *equivariant*
ones. **`certify_candidate` now returns 37/37 on the canary.**

An independent Skeptic returned **PROVISIONAL** and found four real things,
including **a live regression I had introduced**: `max()` orders with `>`, and
`nan > x` is `False`, so a NaN anywhere but the first position was dropped before
the finiteness check — the *pre-amendment* code rejected that input. All fixed;
the battery now runs inside `preflight` and `certify` rather than being a file
nothing executed.

**The canary is packaged, certified, and presented. Nothing has been submitted.**

| | |
|---|---|
| archive | `submission_wp18_canary.tar.gz`, 13,858 bytes |
| sha256 | `e40f6b4de6a10cdcef74b729cc8dbc5eae9a930976e53ba37bee78904fb99960` |
| certify | **37/37**, invariants 30/30 |
| Linux 8-GiB gate | 13/13, VmPeak 2266.8 MB |
| local failures | **0 / 100**, residual exhausted 0 |
| multiplier | `C/B = 0.0766870` → **0.100000, the floor**, with the `+2²⁴` server offset |
| local worst-row headroom | **2.23×** (canary bar is ≥1.5×; the *nomination* bar is ≥4×) |

### Ruling 2 — WP20, the last accuracy question

**The question:** does a *carried* three-generation sketch of the discarded
all-distinct block recover the vector slice as well as the matrix slice, and what
is the joint end-to-end drift reduction?

**Pre-registered** (gate ≥4.0×, falsifier <3.0×, statistic, block, predictions,
all committed before data). **Scored on 64 fresh networks: `2.003×`. The
falsifier fired and I recorded the lane closed.**

**An independent Skeptic then found the scored cell was a mis-operationalisation.**
It reconstructed the vector half from leakage in the compressed matrix half
rather than simply *keeping* the vector — a mechanism imported from an earlier
experiment where the vector genuinely was unavailable. Here it costs `n` numbers
a layer and the run had already computed it; the carried vector sat unused two
lines above the code that discarded it. The Skeptic fitted a response surface to
the run's own 512 observations, predicted the corrected cell at **3.18×**, and
said not to publish until it was measured.

**Measured: `3.330×`.** The closure was withdrawn.

---

## 2. The complete picture, and it is a good negative result

Five routes to recovering what the flow discards, each with a **measured
ceiling** rather than a failed attempt:

| route | ceiling | artifact |
|---|---|---|
| select a subspace of what the flow holds (5 rules) | `1.45×`–`2.10×` **worse than not correcting** | WP13 |
| select by the covariance's dominant range | containment `= r/n` — **chance** | WP14 |
| **predict** the vector from carried per-neuron scalars | held out `0.1002`, **ceiling `0.2575`** | WP18 §B |
| **predict** the matrix from 29 equivariant terms | held out `0.1436`, **ceiling `0.1665`** | WP18 §B |
| **keep** three generations of both, exactly | **`3.874×` end-to-end** | WP20-B, `c2_carry3_hybrid_P05.json` |
| *(what `10×` needs: the vector near-exact)* | *`18.873×`* | WP19 |

WP20-B's full table, 16 fresh networks, live shape, one paired run:

| construction, rank 192 of 1024 | end-to-end | vector recovered |
|---|---:|---:|
| no correction | `1.000×` | — |
| random rank-192 sketch (null) | `1.068×` | `-0.038` |
| **compress the combined object, read both halves back off it** | **`0.781×`** | **`-1.702`** |
| reconstruct the vector from the matrix's leakage *(the scored cell)* | `2.154×` | `0.312` |
| **keep the vector, compress only the matrix** | **`3.330×`** | `0.793` |
| keep both exactly — **the family ceiling** | **`3.874×`** | `0.792` |

Three things matter here:

1. **The carried route is the only one that recovers the vector at all** —
   `0.79` against `0.10` and `0.14` for the two learned maps.
2. **The binding limit is the depth of the memory**, not the compression and not
   the predictability. Compressing the matrix to one-eighth rank costs `0.08` of
   recovery; the vector needs no compression.
3. **Compressing the combined object is DESTRUCTIVE** — `0.781×`, worse than not
   correcting, vector recovery `-1.702`. Second time this project has measured
   injecting *correct* information mid-flow making the answer worse.

**Against a `10×` kill-line requirement, `3.874×` is not close.**

---

## 3. THE THREE QUESTIONS

### Q1 — `CONTRACT WP20` says a pass "buys a second contract". Does it?

The result landed in the band the contract wrote down in advance:

> *"The `3.0×`–`4.0×` band, stated in advance so it cannot be argued afterwards.
> The gate fails and the falsifier does not fire. Pre-registered disposition:
> nothing is promoted, no op work is spent on it, and the lane is not reopened by
> me. It becomes a strategist decision with a measured number attached."*

I have executed that disposition exactly: nothing promoted, no op work, lane not
reopened.

**But there is a tension I should not resolve myself.** The contract also says a
*pass* would "buy a second contract", and `3.330×` is a pass on the falsifier
even though it fails the gate. Meanwhile Ruling 2(b) makes the write-up co-equal
and real work from ~Sep 21.

**My read, offered as a recommendation and not a decision:** do not write a
second contract. The kill-line condition is `≥10×`; the family ceiling is
`3.874×` with *both* slices exact and no estimation anywhere. Buying the missing
factor of three means carrying more generations, which is the uncompressed
object the whole design exists to avoid. **I have an incentive to want a second
contract — I built the cell — which is exactly why this is your call.**

### Q2 — 64 of `P06`'s 128 seeds were spent on the mis-operationalised cell. Do we re-score?

`P06[..000..063]` is spent. `P06[..064..127]` is **unspent** and I have committed
in writing not to touch it.

The corrected `3.330×` is **DESIGN tier** (`P05`, 16 networks). It cannot score
the gate — the gate is written on `P06` with pre-registered cells, and the
corrected cell was not one of them.

**Options:** (a) leave it — the number is directionally solid, the disposition is
unchanged either way, and 16 networks at `t = -20.59` on the key contrast is not
a weak measurement; (b) spend `P06[..064..127]` on a properly-tiered re-score of
the corrected cell, which would make the write-up's `3.33×` a DECIDE-tier number.

**My read:** (a). The decision does not turn on it, and a reserved half should not
be spent to upgrade the tier of a number that changes nothing. But I built the
cell, so I am not the right judge of whether the write-up needs the stronger
claim.

### Q3 — Two process items I declined to decide inside the slice that would benefit

**(i) `P06` is registered DESIGN; I used and labelled it DECIDE.** Nothing
mechanically prevents `P06[..000..063]` from being drawn again and quoted as
independent. Changing a block's tier is an IMPORTANT instrument change and I will
not make it inside the slice that benefits from it.

**(ii) The pre-registration protocol has no check that the cell implements the
hypothesis.** This is the generalisable lesson of the session and it belongs in
the binding protocol document. Our pre-registration fixed the gate, the
falsifier, the statistic, the block and the predictions — everything except
whether the code did what the hypothesis said. It did not, and **every check
passed anyway**: the instrument self-test passed, the in-flight algebra probe
passed at `3e-08`, the null sat at chance, an independent cross-run anchor agreed
to `0.5%`, and the promotion statistic was clean on 64 paired networks. Nothing
was broken. A correct implementation of the wrong construction is invisible to
verification and visible only to an adversarial reader who asks what the code
does with each object it was handed.

**Proposed rule, for your ratification:** *a pre-registered contract must name,
for each object the cell is handed, whether the cell uses it — and the Skeptic's
brief must ask that question first.*

---

## 4. What is NOT in question

* `estimator.py` is untouched at **v2.6.1** (`00240eb1`). Nothing submitted,
  nothing designated.
* Ruling 2(a)'s decision tree on the graded worst-row residual stands unchanged
  and is **blocked on the canary being submitted** — that is Samuel's, and it is
  the single highest-value item on the board. The nomination, the op work, and
  the whole ≥4× branch are downstream of one graded number.
* Nominations by **Oct 10**: A = best certified flow with graded worst-row ≥4×,
  B = `v2.7`. Close **Oct 17**, write-up **Oct 24**.
* All closed lanes stay closed. AUGMENT deprioritised. Carrier and output
  corrector closed.

---

## 5. Where to verify anything above

| what | where |
|---|---|
| every contract, result, Skeptic verdict, erratum | `experiments/LOG.md`, newest at the bottom |
| the snapshot a resuming agent reads first | `docs/CURRENT_STATE.md` |
| WP20 scored run, 64 `P06` nets | `experiments/2026-09-03_wp20/c2_carry3_P06.json` |
| WP20-B corrected run, 16 `P05` nets | `experiments/2026-09-03_wp20/c2_carry3_hybrid_P05.json` |
| the f64 numerical control | `experiments/2026-09-03_wp20/c2_carry3_P06_f64.json` |
| the canary's certification | `experiments/2026-09-03_wp18_canary/certify_guarded.json` |
| the binding measurement protocol | `docs/VALIDATION.md` |
| the write-up | `docs/writeup-p2-draft.md` §§7.1–7.5 |

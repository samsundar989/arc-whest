# KERDOCK_MUB_SPEC.md — the width-1024 real-MUB angular carrier

*Phase 1A, written 2026-08-25 **before** any generator existed; the prediction it
answers is in `experiments/LOG.md` under `PHASE 1A ... PREDICTION`. This file is
the specification the implementation is built from and the verifier is checked
against. Where a choice is free, it says so and fixes it.*

> **COMPONENT SPEC, NOT AN ACTIVE PLAN.** The fixed 16-basis family is a bounded
> carrier candidate and diagnostic endpoint. It is not the current architecture,
> not a proven spherical 4-design, and not licensed for a submission by this file.
> Selection and integration are governed only by `GAME_PLAN.md`.

**Nothing here is a score, variance or design-strength claim.** Phase 1A permits
none, and none is made.

---

## 1. Parameters, fixed

| symbol | value | why |
|---|---|---|
| `m` | **9** | must be **odd** — CCKS obtain the *real* extremal line-sets only for odd `m` |
| `n = m + 1` | **10** | must be **even** — a bent function on `F2^n` exists only for even `n` |
| `N = 2^n` | **1024** | the width |
| `R = GR(4, m)` | `Z4[x]/(h(x))` | `h` = the Hensel lift of a primitive degree-9 binary polynomial |
| `T ⊂ R` | Teichmüller set, `|T| = 2^m =` **512** | indexes the bases |
| `K(m)` | Z4-Kerdock code, `4^(m+1) =` **1,048,576** words | **must not be materialised** |
| real MUBs available | `2^m + 1 =` **513** | `= N/2 + 1`, the real bound |
| bases emitted | **16**, a deterministic subset | see §6 |
| entries | `±1/√N = ±1/32` | |
| cross-basis coherence | **exactly `1/32`** | |

`local_sim/carriers.py` is the length-256 case (`m = 7`, `GR(4,7)`). Width 1024
is `m = 9`. This is a different ring and a new implementation, not a parameter
sweep.

**Primary anchor.** A. R. Calderbank, P. J. Cameron, W. M. Kantor, J. J. Seidel,
*Z4-Kerdock Codes, Orthogonal Spreads, and Extremal Euclidean Line-Sets*,
Proc. London Math. Soc. (3) 75 (1997) 436–480.
doi:10.1112/S0024611597000403.

---

## 2. The ring, and why this ring

`R = GR(4, m)` is the Galois ring of characteristic 4 and rank `m`: `Z4[x]/(h)`
with `h` a **basic primitive** polynomial of degree `m` — i.e. `h mod 2` is a
primitive binary polynomial `f`, and `h` is its Hensel lift, the unique monic
degree-`m` lift with `h | x^(2^m − 1) − 1` over `Z4`.

- `|R| = 4^m`.
- `ξ` = the class of `x`, of multiplicative order `2^m − 1`.
- **Teichmüller set** `T = {0} ∪ {ξ^i : 0 ≤ i < 2^m − 1}`, `|T| = 2^m`.
- Every `z ∈ R` is uniquely `z = a + 2b`, `a, b ∈ T` (2-adic representation).
- **Frobenius** `σ(a + 2b) = a² + 2b²`.
- **Trace** `T_R(z) = Σ_{i=0}^{m−1} σ^i(z) ∈ Z4`.

*Implementation choice, fixed:* `f(x) = x^9 + x^4 + 1`, a primitive polynomial of
degree 9 over `F2`. The Hensel lift is computed, not tabulated, and the
implementation **asserts** both defining properties (`h mod 2 == f`, and `ξ` has
order `2^9 − 1 = 511`) rather than trusting the lift.

---

## 3. The code, and the object we must not build

The Z4-Kerdock code is

```text
K(m) = { c_{λ,ε} : λ ∈ R, ε ∈ Z4 },    c_{λ,ε}(x) = T_R(λ x) + ε,  x ∈ T
```

with `|K(m)| = |R| · 4 = 4^(m+1) = 1,048,576` words of length `2^m = 512` over
`Z4`. **We never enumerate it.** Only the `λ ∈ T` of the chosen 16 bases are
touched, and for each we compute one length-512 `Z4` word — 16 words, not a
million.

The `ε` (constant) and the `2b` part of `λ` contribute only affine terms, which
change a basis by a signed permutation and *not* by its span. The bases are
therefore indexed by `λ ∈ T`.

---

## 4. Gray map and coordinate identification

**Gray map** `φ : Z4 → F2²`, in the standard convention

```text
φ(0) = (0,0)   φ(1) = (0,1)   φ(2) = (1,1)   φ(3) = (1,0)
φ(z) = ( ⌊z/2⌋ , ⌊z/2⌋ ⊕ (z mod 2) )
```

Applied coordinatewise to a length-`2^m` `Z4` word it yields a binary vector of
length `2^(m+1) = N`. This is the classical Gray image under which `φ(K(m))` is
the binary Kerdock code.

**Coordinate identification — an implementation choice, and the first version of
it was WRONG.** The `N` positions of the Gray image are indexed by pairs
`(x, t)`, `x ∈ T`, `t ∈ {0,1}`, laid out as

```text
index = 2·ι(x) + t,   ι(x) = the ADDITIVE image of x in R/2R ≅ GF(2^m),
                             read little-endian as m bits
```

Reduction mod 2 is a **bijection** `T → GF(2^m)`, and it is the map that respects
the `F2`-linear structure the Walsh phase `⟨x, y⟩` needs.

> **Corrected 2026-08-25, by the verifier.** This section first specified
> `index = 2·rank(x) + t`, with `rank` the position of `x` in the *multiplicative*
> enumeration `0, ξ^0, ξ^1, …`. That is not `F2`-linear in anything. The
> generator built from it produced Boolean functions of **ANF degree 9 instead of
> 2**, matrix differences of **rank 6, 8 and 10 instead of 10**, and a cross-basis
> spectrum of a spectrum wildly unlike a flat `32` (the executable negative control reports 36 distinct values, max **144**; the `104` first recorded here is from a transient build that is not in git and cannot be reproduced) — while **within-basis
> orthogonality passed perfectly**, exactly as §5 says a weak check would. V3 and
> V7c caught it; V2 could not have. The rank-based identification is recorded
> here rather than deleted, because "the check that would have missed this" is
> the more useful half of the lesson.

This identification is **not forced by the mathematics** — any linear relabelling
of `F2^n` gives an equally valid line-set — but a *wrong* one breaks §5's
identities, so it is fixed in advance and validated exhaustively at small `N`
(§7). It is the one place where a plausible-looking wrong choice is possible, and
it is exactly what K2 and K3 are aimed at.

---

## 5. The bases

For `λ ∈ T` let `g_λ ∈ F2^N` be the Gray image of `c_{λ,0}` under §4. Define the
`N × N` real matrix

```text
B_λ[x, y] = N^(−1/2) · (−1)^( g_λ[x] ⊕ ⟨x, y⟩ )      x, y ∈ F2^n
```

with `⟨x, y⟩` the standard `F2` dot product. The standard basis `I` is also a
member of the family, and is **excluded** from the emitted set: it contributes
coordinate axes, which carry no angular information a dense propagation does not
already have.

**Why the identities hold.**

*Within a basis.* Columns `y ≠ y'`:

```text
⟨B_λ[·,y], B_λ[·,y']⟩ = N^(−1) Σ_x (−1)^(⟨x, y ⊕ y'⟩) = 0
```

— the quadratic phase `g_λ` cancels, so orthogonality is exact and independent of
`λ`. **This is why within-basis orthogonality is a weak check**: it would pass for
a wrong `g_λ`. It is necessary and it discriminates almost nothing.

*Across bases.* For `λ ≠ μ`, with `c = y ⊕ y'`:

```text
⟨B_λ[·,y], B_μ[·,y']⟩ = N^(−1) Σ_x (−1)^( g_λ[x] ⊕ g_μ[x] ⊕ ⟨x, c⟩ )
```

`g_λ ⊕ g_μ` is the quadratic form of the *difference* of the two Kerdock
matrices, which is **nonsingular** — that is the defining property of a Kerdock
set — so it is a **bent** function on `F2^n` (`n` even), and every one of its
Walsh coefficients has magnitude exactly `2^(n/2)`. Hence

```text
|⟨B_λ[·,y], B_μ[·,y']⟩| = N^(−1) · 2^(n/2) = 2^(−n/2) = 1/32
```

**exactly, for every pair, with no exceptions and never zero.** This is the sharp
discriminator: a construction that is subtly wrong produces some zero or some
other magnitude here, and nothing else in the suite would notice.

---

## 6. The emitted set

- **Basis subset:** the first 16 elements of the fixed Teichmüller enumeration,
  `λ ∈ {0, ξ^0, ξ^1, …, ξ^14}`, in that order. Deterministic, stated, and
  independent of any seed.
- **Rows:** for each basis, all `N = 1024` columns, in increasing `y`.
- **Antipodes:** each row `v` is emitted with `−v`.
- **Total:** `16 × 1024 × 2 = 32,768` rows spanning **16,384 distinct lines**.

The fixed sixteen-basis object is a **component choice**, not a claim about any
leader. Current Phase-2 frontier totals near `5.5e11` FLOPs are compatible with
about 16,384 dense rows—eight antipodal bases in this spelling—but equally with
Sobol, lattice, orthogonal or other carriers at the same row count. The verified
nested policies currently use 1, 2, 4, 8 or 16 bases by truncating the same ordered
list; any intermediate panel must be frozen and tested before use.

---

## 7. What the verifier must check, and at what strength

| # | property | strength |
|---|---|---|
| V1 | entries are `±1` before scaling | exact, integer |
| V2 | within-basis correlations are `0` | exact, integer — **weak, see §5** |
| V3 | cross-basis `|correlation| = 32` for **every** pair | exact, integer — **the discriminator** |
| V4 | `32,768` rows, `16,384` distinct lines, closed under negation | exact |
| V5 | rows are pairwise distinct | exact |
| V6 | byte-identical across runs and a fresh process | exact |
| V7 | exhaustive integer Gram matrices at `N = 16, 64, 256` (**not** an independent reference — see the correction below) | **exhaustive**, not sampled |

> **CORRECTED 2026-08-25 — V7 as written above was never implemented, and the
> claim it makes is WITHDRAWN.** This section said "the reference builds the same
> object by a different route … so a shared misunderstanding cannot make both
> agree." **There is no independent reference.** `exhaustive_gram` consumes the
> generator's own output and the verifier imports the generator's own
> `walsh_matrix`. What exists is a strong independent **property checker** — ANF
> degree, rank over `F2`, a second transform implementation, exhaustive integer
> Gram matrices — and the honest scope is that it would not catch a construction
> wrong in a way that satisfies every property checked. The properties checked
> *are* the definition of a Kerdock set, which is the mitigation, not a proof.
>
> The corresponding prediction K6 was withdrawn in `experiments/LOG.md`, the
> verifier's docstring and `docs/CURRENT_STATE.md` — **and this file was missed**,
> so the withdrawn claim survived here in its strongest form for two commits. An
> independent review found it. CCKS Example 9.2 was attempted as a true second
> generator and abandoned rather than tuned into agreement; the reason is recorded
> in the verifier's docstring.

Reduced dimensions are `m = 3, 5, 7` → `N = 16, 64, 256`, all with `m` odd as §1
requires, and at those sizes the Gram matrices are genuinely **exhaustive**.

---

## 8. Radial handling — separate, and separately validated

A bias-free ReLU MLP is **positively homogeneous of degree 1**:
`f(cx) = c·f(x)` for `c > 0`. So for `x ~ N(0, I_N)` the radial and angular
integrals factor exactly:

```text
E[ f(x) ] = E[ ‖x‖ ] · E_{u ~ Uniform(S^{N−1})}[ f(u) ]
```

with `E‖x‖ = √2 · Γ((N+1)/2) / Γ(N/2)`.

The angular carrier estimates the **second** factor. The radial constant is
validated **without** the carrier, against the exact layer-1 identity
`E[ReLU(w·x)] = ‖w‖/√(2π)`, and only then composed. Conflating the two is how a
correct carrier produces a wrong answer.

---

## 9. Cost and memory, predicted before measurement

Dense propagation of all rows through all layers:

```text
2 × 32,768 × 1024 × 1024 × 16 = 1,099,511,627,776 counted FLOPs
```

The `1,099,802,385,000` public figure that motivated this calculation was a
**historical Phase-1 fingerprint**: **`+290,757,224` (`+0.0264%`)** above the dense
propagation term. It remains a useful arithmetic check, not evidence about the
current Phase-2 leaders. The 2026-08-31 leader reports about `5.45e11` mean counted
FLOPs, close to half this row count, but the report has no operation ledger and
does not identify a carrier.

Memory: one `32,768 × 1024` float32 point buffer is **128 MiB**; weights are
`16 × 1024² × 4 B = 64 MiB`. Both far below the 4 GiB single-array cap. **The
dense reference is built first.** Chunking is an optimisation variable, not a
correctness requirement, and any chunked spelling is compared against the dense
reference before it is trusted.

---

## 10. Established mathematics vs. choice vs. inference

**Mathematics** (CCKS, and standard bent-function theory): the existence of
`2^m + 1` real MUBs in `R^(2^(m+1))` for odd `m`; the Kerdock set's
pairwise-nonsingular differences; bentness of the difference form; the exact
`1/√N` coherence.

**Implementation choice** (fixed here, validated at small `N`): the primitive
polynomial `x^9 + x^4 + 1`; the Teichmüller enumeration order; the Gray
convention; the coordinate identification of §4; taking the first 16 `λ`;
excluding the standard basis.

**Historical inference, now withdrawn as an active frontier claim**: mapping a
Phase-1 published FLOP total to sixteen bases, `32,768` directions and antipodes.
The arithmetic is reproducible, but a FLOP total does not identify a private
algorithm. Current Phase-2 telemetry is used only to bound an operating envelope.

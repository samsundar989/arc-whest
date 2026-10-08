# R4 — public footprints of the top Phase-2 teams and follow-ups to ARC's paper (2026-10-07)

*Research agent, web only (WebSearch/WebFetch, public pages, git clone of one public repo). Everything below is data from public sources; nothing was posted or contacted.*

## (a) Direct evidence about top teams
- **No method disclosure** from J2W, marius_binner, Luna, mliston, mk1981, oliver_chen, suliman_tadros, ben3, thegamers, shiv_m, lode_dockx, Masato, oqaris or Puffi (GitHub profiles, LessWrong/AF, Substack, X, Hugging Face, affiliation searches).
- **kaileh57 (rank 17, raw 1.46e-8, cost ≈ 0.17)**: public fork github.com/kaileh57/whest-starterkit, research on `agent/*` branches, Aug 9–25 (nothing pushed after Aug 25; no scores in the repo; current submission unknown). Upstream LICENSE is MIT (AIcrowd); kaileh57's research files carry no headers.
  - Aug 10 checkpoint (256×32): all-distinct κ3 approximated as Σ Sym(a_m ⊗ A_m), A_m ∈ {C, C², C³}, fitted from the (2,1) slice; "κ3 alone fails … The fourth-order state is load-bearing" (K3-only rollout went non-PSD); κ4 ≈ Sym₆(M⊗C) with M n×n, closed under the linear step (M → WᵀMW); bivariate Edgeworth pair map with split Gauss–Legendre at the kink (oracle-moment off-diagonal covariance error 4.5e-3 Gaussian → 4.8e-5 with K3 → 6.6e-6 with K3+K4).
  - Aug 24–25 (1024×16), five separate small estimators: memoryless "goal-directed" corrections contracted against next-layer weight columns — κ3 (1,1,2) tree via P = C·(g1∘W) (one n³ per layer), κ4 star 4Σ g3·w·P³ (no extra product) and path 12Σ F∘(C·F), F = g2∘W∘P (one more n³), mean Δμ = κ3 f3/6 + κ4 f4/24 + κ3² f6/72; rank-capped factored K3 (on Paul Rosu's public MIT SIMPLE code); a "signed recompression" of factored K3 columns by OMP on downstream-weighted features (tested only at width 24–32); Price covariance series to k = 2/4/6; Kerdock/MUB spherical cubature. No measured 1024×16 numbers survive (CI logs need login).
- **AndreasHad04 (rank 9)**: flopscope issues #275 (`out=` larger than inputs under-billed 33–513×) and **#285** (Oct 5: ufuncs with `where=` keep the symmetric tag, `exp` bills ~49 % low). Organizers promised a fixed release and a re-evaluation; v0.12.1 (Aug 29) is still the latest tag.
- Mid-table, for context: Hisernberg/arc-whitebox (public, Oct 7): 504aldo V29 + a learned cross-neuron corrector, raw 1.965e-8 at 0.205 B (#334627, 4.045e-9).

## (b) Research follow-ups to arXiv:2605.05179
- Paper still v2 (May 14); OpenAlex cited_by_count 0. Semantic Scholar rate-limited (429) and arXiv/Scholar search blocked by robots.txt, so the citation check is incomplete.
- ARC blog: newest post Jun 9; nothing on depth-independent methods, Borel resummation or low-rank allocation.
- ARC repo `main` unchanged since Aug 4 (93d091a). Unmerged July branches: `max` (a stabiliser that propagates low-variance coordinates as z → z + (ReLU(μ) − μ), disables the factored path) and an Edgeworth/treewidth estimator for E[max] (different target).

## (c) Searched, nothing found
AIcrowd Notebooks/Showcase (empty), forum posts since Oct 1 (#18237, #18238 memory accumulator question, #18240, #18118), Hugging Face discussions, GitHub web searches and ~35 username/repo probes, LessWrong/AF threads, Substack/X/LinkedIn.

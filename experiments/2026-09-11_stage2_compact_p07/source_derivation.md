# Independent bounded K4 source derivation (2026-09-12)

Scope: read-only mathematical audit of frozen `frozen_arms_metadata_protocol/arm_coherent.py`, SHA-256 `d3ec837b72f5dacb4f8f622394d3d635268649c741289fb9754197e993343665`, and `local_sim/test_p2_k4_coherent.py`. No protected arrays, live labels, or heavy evaluation were used. This is not the formal fresh Skeptic verdict or an accuracy result.

## Exact source invariants

The first preactivation `z=W0 x` is jointly Gaussian because `x~N(0,I)` and `W0` is fixed. Therefore its entire cumulant tensor of order four is **zero**, even when two `z` coordinates are correlated. A fourth-order consumer or induced birth attributed to that tensor must vanish at this layer.

An independent-coordinate fixture is obtained by choosing orthogonal rows of `W0` of norm one. Then `Y_a=ReLU(z_a)` are independent, with `z_a~N(0,1)`. Write `m=1/sqrt(2π)`, `v=1/2-m²`. The exact moments give

```
E[Y]=m, E[Y²]=1/2, E[Y³]=2m, E[Y⁴]=3/2,
M4=E[(Y-m)⁴]=3/2-5m²-3m⁴ = 0.6282343968087699,
κ4(Y)=M4-3v²=3/4-2m²-6m⁴ = 0.2797083383527027.
```

Independence means the only nonzero input fourth cumulants are `K_aaaa=κ4_a`. For any next dense row matrix `B`, multilinearity gives the exact transported slices

```
D4_i  = Σ_a B_ia⁴ κ4_a,
D22_ij= Σ_a B_ia² B_ja² κ4_a,
D31_ij= Σ_a B_ia³ B_ja κ4_a.
```

In particular, input `K_aabb=0` for `a≠b`. With equal `κ4_a=κ4` and a row whose entries are `±sqrt(2/n)`, `D4_i=4κ4/n`. This is an exact deterministic identity for that fixture; it is not an inference from the contest distribution.

## What the frozen source constructs

At the first layer `D4=0`, but lines 220–227 call `M4` `k4v`: the required subtraction `3*var_out**2` is absent. They set `k22_ij=(k4v_i+k4v_j)/6` for `i≠j`, even when the ReLU coordinates are independent and the exact mixed cumulant is zero. The contraction/projection at lines 228–231 then maps equal `M4` to `gcore_i=M4/2`. For independent input covariance, the `λ` pieces in the diagonal transport (lines 181–184) cancel. The next equal-magnitude He row consequently receives `D4_i=2 M4=1.2564687936175398`, rather than `4κ4/n`; at `n=1024` the factor is about `1150`. Subtracting `3v²` and setting independent `k22` to zero removes this particular width-scale error, but still does not make the matrix core an exact representation: for equal `κ4`, contraction projection gives `gcore_i=3κ4/[2(n+2)]` and the same row gives `D4_i=6κ4/(n+2)`, whereas the exact value is `4κ4/n`.

The represented core is `T_abcd=(2/6) Sym_6(G_ab δ_cd)`, with `G=diag(g)+λ C_off`, as enumerated in the test at lines 29–45 and 117–123. For distinct `i,j`, its slices are `D4_i=2G_ii`, `D22_ij=(G_ii+G_jj)/3`, and `D31_ij=G_ij`. Hence a nonzero diagonal core necessarily produces nonzero mixed `D22` in general. A diagonal fourth-cumulant tensor of independent non-Gaussian coordinates cannot equal this core: `K_iijj=0` for all distinct pairs while `K_iiii>0`. The diagonal cumulant must be retained separately or the core must explicitly be called an approximation with its residual and contraction error measured.

The test checks these slices only for a **supplied** `G`, masks the `D22/D31` diagonals, checks transport of `diag(W G Wᵀ)` only, and transports an induced cubic tensor built from that same supplied core (test lines 117–172). It does not establish that the source produced a valid fourth cumulant, or that an off-diagonal `D31` was transported. On the diagonal, the computed `D22_ii=D4_i/3` and `D31_ii=0` are not fourth-tensor slices; their use is masked, so this is a domain restriction rather than an observed off-diagonal covariance error.

## Transport, zero, symmetry, and births

The diagonal expression at frozen lines 181–184 is exactly `diag(B G_old Bᵀ)` for the assumed `G_old=diag(gcore)+λ C_old,off`. Its off-diagonal is **not** generally `λ C_pre,off` as line 188 assumes. An exact 2×2 signed example, with orthogonal `B` so the represented metric remains unchanged, is

```
C_old=I, G_old=diag(1,0), λ=1/3,
B=(1/sqrt(2))*[[1,1],[1,-1]],
C_pre=B C_old Bᵀ=I,
B G_old Bᵀ=[[1/2,1/2],[1/2,1/2]].
```

The represented off-diagonal `D31_12` is `1/2`, while frozen line 188 gives zero. More generally the omitted matrix is `off(B diag(gcore-λ diag(C_old)) Bᵀ)`. For a nonorthogonal `B`, full fourth-tensor transport also changes the metric from `I` to `B Bᵀ`; freezing the metric at `I` is an additional approximation. The simple scaling `B=tI` makes the exact fourth cumulant scale as `t⁴`, whereas this fixed-metric `G` transport and its `D4` scale as `t²`.

The Gaussian L1 zero-source rule is violated in two directly inspectable places. With `gcore=None`, frozen line 180 makes `dG=0`, but line 188 still sets `D31=λ C_off`, nonzero for correlated Gaussian preactivations. This particular L1 `K11` contribution is suppressed because `mu_pre=0` makes `w3=0`; it does not restore a zero fourth-cumulant source. In the birth path (lines 239–248), `dG=0` leaves

```
feed21_ij=(2/3)y_i a_ij c1_j
         =(λ/3) w2_i w1_i w1_j C_off,ij,
```

which is nonzero when `C_off,ij≠0`, although no preactivation K4 exists. `feed3` is zero here. Thus the supplied-core cubic oracle cannot certify this actual L1 birth source.
Under a positive rescaling `z→t z` with fixed `λ`, this term scales as `t` (`C_off→t² C_off`, `w2→w2/t`), whereas a third cumulant of `ReLU(tz)` scales as `t³`. Likewise `D31=λ C_off` scales as `t²`, although every fourth cumulant scales as `t⁴`.

Finally, the `D31` term in `K11` (lines 204–206) is generally asymmetric. Its antisymmetric difference is

```
K11_ij-K11_ji = (D31_ij+D31_ji)
               *(w3_i*w1_j-w3_j*w1_i)/12.
```

Take `σ_i=σ_j=1`, `mu_pre,i=0`, `mu_pre,j=1`, and symmetric nonzero `D31_ij`: then `w3_i=0`, `w3_j=-φ(1)`, and the difference is nonzero. A covariance must be symmetric. For a genuine fourth tensor the first-order mixed correction has the symmetric orientation `D31_ij*w3_i*w1_j/6 + D31_ji*w1_i*w3_j/6` (alongside the `D22` term). A numerical symmetrization of the frozen result would hide the error without validating this coefficient or source.

## Repair boundary and inference limit

The minimum defensible mathematical boundary is to construct actual marginal fourth **cumulants**, enforce zero fourth source for a Gaussian preactivation and zero mixed source for independent post-ReLU coordinates, and either retain the diagonal K4 component under dense transport (formulas above) or record exactly which contractions a projected core approximates. `D4`, `D22`, `D31`, and all K3 births must be derived from that same declared tensor; the `D31` covariance update must preserve symmetry. A tiny fixture can falsify these algebraic identities and prove a repaired implementation matches them on its chosen inputs. It cannot establish live-width accuracy, residual headroom, counted-cost feasibility of a richer representation, a promotion, or a scientific kill of coherent K4.

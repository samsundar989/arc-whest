# Partitioned positive quadrature for the fixed two-layer increment

Design only, 2026-09-12. This memo does not read the increment fixture's result, change the committed `1e99494` contract or its gates, or assert a measured outcome. The proposed calculation concerns the fixed `rho=0.5`, `X=ReLU(Z)`, `U=BX`, `Y=ReLU(U)`, `B=[[0.8,-0.3],[-0.55,1.4]]`, and declared old carry `K(U)=B^{otimes 4}K(X)`. It does not define the real later-layer response or a width-1024 source.

## Split the actual pushforward measure

The law of `X` has four support pieces, not merely a positive-quadrant density. For standard correlated Gaussian `Z` with correlation `rho`, its atom at `(0,0)` has mass `m00=1/4+asin(rho)/(2*pi)`; each open axis has mass `m10=m01=1/4-asin(rho)/(2*pi)`; the open quadrant has mass `m11=m00`. At `rho=0.5` these are respectively `1/3`, `1/6`, `1/6`, `1/3`. The first-axis density for `x>0` is

```
f10(x) = phi(x) Phi(-rho*x/sqrt(1-rho^2)),
```

and the second-axis density is the same with coordinates exchanged. Keep the origin as an explicit positive-weight node and each axis as its own positive-weight node set. On the first axis `Y=(0.8*x,0)`; on the second `Y=(0,1.4*x)`; at the origin `Y=0`. Thus all three singular pieces contribute to means, covariances and fourth cumulants even though the origin contributes zero to positive-degree *raw* moments.

For the open quadrant write `x1=r*cos(theta)`, `x2=r*sin(theta)`. The two `B`-row zero rays are `theta_low=atan(11/28)` for `U2=0` and `theta_high=atan(8/3)` for `U1=0`. Split the angle into `[0,theta_low]`, `[theta_low,theta_high]`, `[theta_high,pi/2]`. In these sectors the active outputs are respectively `(Y1,0)`, `(Y1,Y2)`, `(0,Y2)`. The density times area element is

```
[r/(2*pi*sqrt(1-rho^2))] * exp(-q(theta)*r^2/2) dr dtheta,
q(theta) = (1-rho*sin(2*theta))/(1-rho^2).
```

Within each sector every raw monomial of `X`, `U`, or `Y` of total degree at most four is a polynomial in `r` with a smooth angular coefficient. The oblique ReLU kinks now lie on integration boundaries, including the two axis boundaries.

## One positive node bank for every discrete identity

Use Gauss-Legendre `N` on each of the three fixed angle intervals and on `u in (0,1)` for each axis, with `x=u/(1-u)` and its positive Jacobian. At an angular node, set `t=sqrt(q)*r`. The normalized conditional radial weight is `t*exp(-t^2/2) dt` on `(0,infinity)`. Its moments are `mu_k=2^(k/2)*Gamma(1+k/2)`. A three-node Gaussian rule for this weight has positive nodes and weights and integrates powers `t^k` exactly through degree five, so all required degree-four radial polynomials are exact at each angle. Construct it reproducibly from `H0[i,j]=mu_(i+j)` and `H1[i,j]=mu_(i+j+1)` for `i,j=0..2`: Cholesky `H0=L L^T`, diagonalize the symmetric `L^-1 H1 L^-T`, take eigenvalues as nodes and squared first eigenvector components as weights. Check nodes and weights are positive, sum of weights is one, and moments through degree five agree with `mu_k` before using the bank.

The unscaled quadrant weight for angle node `(theta,wtheta)` and radial node `(t,wt)` is `wtheta*wt/[2*pi*sqrt(1-rho^2)*q(theta)]`; its location is `X=(t/sqrt(q))*(cos(theta),sin(theta))`. Axis weights are Gauss-Legendre weights times `f10(x)/(1-u)^2` at `(x,0)` and `(0,x)`. Scale the quadrant weights together to sum to exact `m11`, each axis set separately to exact `m10` or `m01`, and set the origin weight to exact `m00`. This preserves positivity, normalization and the four support masses. Record the three pre-scaling mass errors as quadrature diagnostics; scaling must not hide a poorly resolved region.

Form `U=BX` and `Y=max(U,0)` from this *one* immutable `X,p` bank. Center `X`, `U`, and `Y` separately with the same `p`. The old component is the positive centered-`U` fourth factor and its negative same-`U` covariance pairings. The later component uses centered `Y` and its own covariance pairings. The increment is later minus old, so both fourth factors and both pairing tensors retain their respective signs. Check `K(U)=B^{otimes 4}K(X)` and `K(Y)-K(U)` against separately enumerated raw-moment partitions on that same bank, then transport both components through the fixture's Hadamard and signed second maps. Recheck all D4/D22/D31 entries, oriented covariance and the declared quadratic K3 feed. These are exact-for-a-chosen-discrete-measure identities; they should stay within the existing `1e-10` absolute gate at every order, independent of continuous quadrature convergence. Reuse the identity and negative-identity `B` controls and the naive old-plus-whole-later discriminator unchanged.

Use fixed `N=32,48,64` for this diagnostic sequence. The bank has `9N+2N+1=11N+1` nodes: 353, 529, and 705, compared with the existing full-sign product's 4,096 and 9,216 nodes at half-orders 32 and 48. A float64 `p` and three `2 x 705` arrays for `X,U,Y` occupy under 40 KiB; centered copies, raw moments and 2-by-2 tensors keep the live array payload comfortably under 1 MiB. That is an array estimate, not a process-RAM measurement. Building nodes requires `3N` angular points, `2N` axis CDF evaluations, three radial nodes reused at all angles, and `O(11N)` work per raw monomial. This is a tiny CPU-only fixture; no challenge FLOP, residual-time or rank claim follows.

## Independently check continuous moments

Implement a separate conditional-Gaussian reference, without calling the node-bank construction or its factor/partition helpers. For `z1>0`, use `Z2 | Z1=z1 ~ N(rho*z1,1-rho^2)` and split its integral at `0`, `(11/28)*z1`, and `(8/3)*z1`. The four intervals are the first axis and the three open-quadrant sectors. For `z1<=0`, split at `z2=0` into the origin and second-axis contribution. In each interval `X`, `U`, and `Y` are fixed linear polynomials in `z1,z2`; expand each requested raw monomial of total degree at most four and integrate powers of `z2` with truncated-normal moments. For standard-normal limits `a,b`, use `J0=Phi(b)-Phi(a)`, `J1=phi(a)-phi(b)`, and `Jk=(k-1)J_(k-2)+a^(k-1)phi(a)-b^(k-1)phi(b)` through degree four, with zero infinite-boundary terms. Binomially shift and scale for the conditional mean and variance. Integrate the remaining positive `z1` dimension adaptively at higher precision; compute the second-axis piece separately by its one-dimensional density. Construct continuous cumulants from independently coded raw set partitions, not from discrete centered factors.

An additional cross-check can use the polar sectors above but integrate radius analytically rather than through radial nodes: for a homogeneous degree-`d` monomial, `int_0^infinity r^(d+1) exp(-q*r^2/2) dr = 2^(d/2) Gamma(1+d/2) q^(-(d+2)/2)`. A separately coded one-dimensional adaptive angular integral then checks the conditional reference without sharing its truncated-normal recurrence. Check support masses and the known first moments `E[X_i]=1/sqrt(2*pi)`, second moments `E[X_i^2]=1/2`, plus the correlated `E[X1 X2]` ReLU kernel as low-degree controls.

Compare raw moments, `K(X)`, `K(U)`, `K(Y)`, their difference, and the two-map consumers separately at 32/48 and 48/64 and against the conditional reference. Preserve the contract's `1e-8` relative gate for entries above `1e-6` and `1e-8` absolute otherwise; do not promote a cancellation in `Delta` based only on its apparently stable difference. Record worst absolute and material-relative errors for each family, support-mass errors before scaling, radial moment defects and reference-integration precision/doubling error. If the fixed gate still fails, report that failure; the method is a diagnostic proposal, not a revised verdict. No protected labels, seed block, public split or vault is involved.

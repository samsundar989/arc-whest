# Full-polar bank for the failed three-layer nonlinear fixture (proposal)

This is a design only. The frozen fixture at `39f21de` remains FAIL 54/56:
order-48/64 and order-48/direct positive-law layer-three moment gaps are
about `3.6e-5` and `3.1e-5`, versus its `1e-8` gate. Its signed-law identities
on that discrete bank are separate. No corrected bank has been measured.

The original two-layer bank partitions the first ReLU's origin/axes and the
middle B map, but the final `D ReLU(Bx)` introduces two more zero boundaries
inside a middle-map sector. A Gauss-Legendre angular rule crossing such a
fourth-moment kink has no smooth-integrand convergence guarantee. Define a
new positive Gaussian bank from the *original* correlated preactivation
`Z~N(0, Sigma)`, with `Sigma=[[1,.5],[.5,1]]`. Partition the full polar
angle at every zero of the two coordinates of `Z`, `B ReLU(Z)`, and
`D ReLU(B ReLU(Z))`, using the already frozen B and D. On each interval
the entire nested biasless map is a fixed linear function of radius. The
breakpoints are computed from fixed matrix rows and sign patterns, with
each root included once and interval endpoints audited; they must not be
chosen after inspecting moment errors.

At angle `theta`, write `e=(cos(theta),sin(theta))` and
`q=e.T Sigma^-1 e`. For a normalized Rayleigh radial variable with measure
`t exp(-t*t/2) dt` on `t>0`, its moments are
`mu_k=2^(k/2) Gamma(1+k/2)` for `k=0,...,5`. A positive three-point
Hankel/Gaussian rule from `H0[i,j]=mu_(i+j)` and
`H1[i,j]=mu_(i+j+1)`, `i,j=0,1,2`, integrates every radial monomial through
degree five. For each angle quadrature node with weight `wtheta`, form
three preactivation particles `Z=(t/sqrt(q))*e` and positive weights
`wtheta*wradial/(2*pi*sqrt(det(Sigma))*q)`. Map particles through all three
ReLUs; retain coincident mapped particles as indexed atoms for subsequent
source reweighting. Check positive weights, full mass, six radial moments,
sector boundaries and maximum single array size *before* trusting results.
There is no a posteriori rescaling of quadrant/axis masses.

Use fixed angular orders 32/48/64 on *each* fully split interval. The
independent direct continuous oracle integrates Gaussian radius analytically
and the remaining angle with adaptive scalar quadrature; it should not use
the proposed particle bank or reweighting code. The unsplit order-48 bank
is a negative control. Compare every unreweighted raw monomial through
degree four at layers two and three, including degree-one and oriented
`(3,1)` entries. A new contract would freeze `1e-8` material-relative
positive-law gates, scalar integration bound below `1e-10`, and require the
unsplit negative control to retain a gap above `1e-6`. Recompute the same
fixed weighted-nullspace fourth-order contrasts and four counterfactual
source identities on the new bank with their original `1e-11`, `1e-10`,
`1e-6` and relative `1e-3` gates; preserve any newly failed discriminator.

This bank looks ahead at fixed B and D to make a *toy diagnostic* smooth.
It is not a practical way to propagate an unknown width-1024 network and
does not prove a K4-only nonlinear closure: a signed atomic perturbation
contains higher moments, and ReLU can create lower-moment leakage. If an
independent Skeptic accepts the old failure diagnosis, preregister this new
intervention and its falsifiers in `experiments/LOG.md` before coding or
running it. A fresh Skeptic must then inspect the raw numerical result.

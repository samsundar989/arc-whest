# Exact-moment axis rule for the two-layer source fixture (proposal)

This is a design derivation, not a measured result or an accepted source
closure. The registered boundary fixture at `f4f1305` failed its 32/48
convergence gate. Its mapped half-axis rule resolves total mass but not the
fourth axis moment at order 32. A new contract must precede any new test.

For a correlated standard Gaussian pair with `rho=1/2`, the horizontal
axis of `X=ReLU(Z)` has density

`f(x) = phi(x) Phi(-rho*x/sqrt(1-rho**2))`, `x>0`.

The vertical axis is symmetric. Define `mu[k] = integral_0^infinity
x**k*f(x) dx` for `k=0,...,5`; `mu[0]=1/6`. Form the positive Hankel
matrices `H0[i,j]=mu[i+j]` and `H1[i,j]=mu[i+j+1]` for `i,j=0,1,2`.
With `L=chol(H0)`, diagonalize the symmetric matrix
`J=L**(-1) H1 L**(-T)`. Its three eigenvalues are axis nodes, and
`mu[0]` times the squared first components of its orthonormal eigenvectors
are their weights. For this positive half-line measure the nodes and
weights are positive; the three-node Gaussian rule integrates polynomials
through degree five exactly in exact arithmetic. Check all six moments,
positivity, and the *pre-scaling* mass numerically before use. Recording
the independent one-dimensional integration error prevents a moment match
against its own inaccurately computed input from being mistaken for truth.

Keep the origin atom, three oblique-boundary quadrant sectors, positive
three-node radial rule, conditional-Gaussian reference, structural controls,
and fixed angular orders 32/48/64. Only the two half-axis components change
from mapped Gauss-Legendre to this exact-moment rule. The bank sizes become
`7+9*N` nodes: 295, 439, and 583. If the quadrant angular rule already
converges at order 32, exact axis moments should remove the observed
high-moment gap without changing the `1e-8` numerical or `1e-10`
structural thresholds. This is a falsifiable prediction, not a pass.

The construction certifies at most raw degree-four moments and their
finite-measure fourth-cumulant algebra for the fixed two-coordinate toy
law. It says nothing about an affordable width-1024 source, later
nonlinear old-history response, multi-source rank, or challenge score.

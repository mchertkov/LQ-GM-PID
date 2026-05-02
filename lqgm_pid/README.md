## `lqgm_pid/`

Core PyTorch API for the analytic LQ-GM-PID backbone used in the paper: "Analytic Bridge Diffusions with Controlled Path Generation" (released May 2, 2026). The package implements piecewise-constant linear--quadratic protocols, Riccati/Green-function coefficient propagation, closed-form optimal control evaluation, exact intermediate Gaussian-mixture marginals, and Euler--Maruyama simulation for diagnostics and visualization.

The main public entry point is `LQGMPID`: construct a `MatrixPWCProtocol`, specify a Gaussian-mixture target, set the deterministic source point `x0`, run `precompute()`, and then query the closed-form score/control field, log potential, responsibilities, marginals, or simulated trajectories.

### Main files

- `core.py`  
  Dataclasses for the API:
  - `TimeDomain`: endpoint-clamped time interval handling;
  - `GaussianMixture`: weights, means, covariances, and precisions;
  - `MatrixPWCProtocol`: piecewise-constant protocol  
    $$
    \Gamma_t=(\beta_t,\nu_t,\sigma_t,\kappa_t)
    $$
    with full matrix-valued `beta` and `sigma`;
  - `CoeffState`: Green-function coefficient snapshot  
    $$
    (A,B,C,\theta_x,\theta_y,\zeta).
    $$

- `hamiltonian.py`  
  Builds the augmented Hamiltonian matrices used for Riccati and linear-coefficient propagation on each PWC interval. This includes backward/forward Hamiltonians for the quadratic coefficients, augmented systems for `C`, and augmented systems for the linear terms.

- `coeff_propagator.py`  
  Single-interval propagation of Green-function coefficients. Provides:
  - `backward_interval`;
  - `forward_interval`;
  - `delta_bc`.

  The implementation uses matrix-exponential propagation for the general matrix case and includes optimized analytic branches for important special cases such as zero drift with diagonal or SPD `beta`.

- `sweep.py`  
  Full backward and forward sweeps over all PWC intervals:
  - `backward_sweep(protocol)`;
  - `forward_sweep(protocol)`;
  - `full_sweep(protocol)`.

  These return lists of `CoeffState` objects at the protocol breakpoints.

- `control.py`  
  Evaluation routines for the closed-form LQ-GM-PID control/score:
  - `eval_bwd`;
  - `eval_fwd`;
  - `gmm_control`.

  `gmm_control` evaluates the optimal drift $u_t^*(x)$, the log potential/log normalizer, and the Gaussian-mixture responsibilities at query points.

- `density.py`  
  Exact intermediate marginal evaluation. The main routine,
  `exact_marginal_gmm`, returns the Gaussian-mixture representation of the instantaneous marginal \(p_t(x)\), including component weights, means, precisions/covariances, and related diagnostic quantities.

- `pid.py`  
  High-level wrapper class `LQGMPID`. It caches the forward/backward sweeps and exposes the convenient user-facing methods:
  - `precompute()`;
  - `control(t, x)`;
  - `control_full(t, x)`;
  - `log_psi(t, x)`;
  - `bwd_at(t)`;
  - `fwd_at(t)`;
  - `simulate(...)`.

- `__init__.py`  
  Public API exports for the package.

### Typical usage

```python
from lqgm_pid import LQGMPID, MatrixPWCProtocol, GaussianMixture

protocol = MatrixPWCProtocol.from_scalar_beta(breaks, beta_scalars, nu)
target = GaussianMixture(weights, means, covs)

pid = LQGMPID(protocol=protocol, target=target, x0=x0)
pid.precompute()

u = pid.control(t, x)                 # closed-form optimal drift / score
u, log_psi, rho = pid.control_full(t, x)
result = pid.simulate(B=256, n_steps=2000)

## `lqgm_pid/`

Core Python API for LQ-GM-PID. This package implements the analytic backbone used by the paper: piecewise-constant protocols, forward/backward Riccati coefficient propagation, Gaussian-mixture Green-function algebra, closed-form score evaluation, exact intermediate marginal evaluation, and density-level diagnostics/objectives used by the experiment notebooks.

Typical contents:

- `coeff_propagator.py`  
  Riccati and linear-coefficient propagation for the LQ-GM-PID forward/backward Green functions under piecewise-constant protocols.

- `green_functions.py`  
  Gaussian Green-function utilities and algebra used to assemble bridge scores, responsibilities, and marginal Gaussian-mixture components.

- `score.py` / `control.py`  
  Evaluation of the optimal drift / score field $u_t^*(x)$ from the propagated coefficients.

- `marginals.py`  
  Closed-form intermediate marginal \(p_t(x)\) as a Gaussian mixture, including component weights, means, and precisions.

- `protocols.py`  
  Data structures and helper constructors for protocols $\Gamma_t=(\beta_t,\nu_t,\sigma_t,\kappa_t)$, including scalar, anisotropic, and block-structured matrix schedules.

- `gmm.py` / `specs.py`  
  Gaussian-mixture target/source specifications used in the corridor, multi-entrance, and high-dimensional trunk--branch--local experiments.

- `objectives.py` / `metrics.py`  
  Density-level path objectives and diagnostics, including corridor adherence, guide cost, kinetic/protocol penalties, subspace variance traces, and terminal mode-allocation checks.

- `sampling.py`  
  Euler--Maruyama trajectory simulation using the closed-form score, used only for visualization and empirical diagnostics, not for density-level optimization.

The API is designed so that experiment notebooks can construct a source/target GMM and a protocol, run the Riccati precompute once, and then query scores, marginals, objectives, gradients, and diagnostic quantities without an inner stochastic simulation loop.

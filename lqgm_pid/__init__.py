"""lqgm_pid — LQ-GM-PID optimal control for Gaussian-mixture diffusion bridges.

Public API
----------
    LQGMPID              high-level controller (precompute + control + simulate)
    MatrixPWCProtocol    piecewise-constant drift/potential schedule
    GaussianMixture      terminal target distribution
    TimeDomain           time-axis parameters (eps)
    CoeffState           raw Riccati coefficient struct (advanced use)
    backward_sweep       raw backward Riccati sweep
    forward_sweep        raw forward Riccati sweep
    gmm_control          raw batched control evaluation
    exact_marginal_gmm   exact instantaneous marginal as a Gaussian mixture
    make_standard_forward_sweep  forward sweep with standard BC (for density)
"""

from .core import (
    CoeffState,
    GaussianMixture,
    MatrixPWCProtocol,
    TimeDomain,
)
from .sweep import backward_sweep, forward_sweep
from .control import eval_bwd, eval_fwd, gmm_control
from .pid import LQGMPID
from .density import exact_marginal_gmm, make_standard_forward_sweep

__all__ = [
    # High-level
    "LQGMPID",
    # Data types
    "CoeffState",
    "GaussianMixture",
    "MatrixPWCProtocol",
    "TimeDomain",
    # Sweeps
    "backward_sweep",
    "forward_sweep",
    # Control
    "eval_bwd",
    "eval_fwd",
    "gmm_control",
    # Density
    "exact_marginal_gmm",
    "make_standard_forward_sweep",
]

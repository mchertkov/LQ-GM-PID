# %% [markdown]
# # Common utilities for LQ-GM-PID paper experiments
#
# This module provides:
# - reproducibility helpers,
# - standard scenario builders,
# - 2D centerline / corridor construction,
# - protocol construction wrappers for `lqgm_pid`,
# - target / initial Gaussian-mixture builders,
# - a thin simulation wrapper.
#
# The design goal is to keep all experiment scripts concise and consistent.

# %%
from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

# Robust project-root import so the same file works as a script and after
# `jupytext --to notebook`.
if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    ROOT = Path.cwd().resolve()
    if not (ROOT / "lqgm_pid").exists():
        for parent in [ROOT] + list(ROOT.parents):
            if (parent / "lqgm_pid").exists():
                ROOT = parent
                break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lqgm_pid import GaussianMixture, LQGMPID, MatrixPWCProtocol, TimeDomain


# %% [markdown]
# ## Reproducibility and filesystem helpers

# %%
def ensure_dir(path: str) -> str:
    """Create a directory if needed and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Set Python / NumPy / Torch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> str:
    """Use CUDA if available, otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_dtype() -> torch.dtype:
    """Default experiment dtype."""
    return torch.float64


def make_time_grid(n_steps: int, *, dtype: torch.dtype | None = None,
                   device: str | torch.device | None = None) -> torch.Tensor:
    """Standard time grid on [0, 1]."""
    if dtype is None:
        dtype = default_dtype()
    if device is None:
        device = "cpu"
    return torch.linspace(0.0, 1.0, n_steps + 1, dtype=dtype, device=device)


def save_metrics_json(path: str, metrics_dict: Dict) -> None:
    """Save metrics dictionary as JSON, converting tensors/arrays to lists."""
    ensure_dir(os.path.dirname(path) or ".")
    serializable = _to_jsonable(metrics_dict)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)


def save_csv(path: str, rows: Sequence[Dict]) -> None:
    """Save a list of dictionaries to CSV."""
    import csv

    ensure_dir(os.path.dirname(path) or ".")
    rows = list(rows)
    if not rows:
        raise ValueError("rows must be non-empty")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_jsonable(obj):
    """Convert nested tensors/arrays/dtypes to JSON-safe objects."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# %% [markdown]
# ## Lightweight scenario containers

# %%
@dataclass
class Centerline2D:
    """Discrete 2D centerline and associated geometric data."""
    t: torch.Tensor            # (T,)
    xy: torch.Tensor           # (T, 2)
    tangent: torch.Tensor      # (T, 2)
    normal: torch.Tensor       # (T, 2)
    width: torch.Tensor        # (T,)


@dataclass
class ScenarioConfig:
    """Standard experiment configuration."""
    seed: int = 12345
    B: int = 2000
    n_steps: int = 400
    K: int = 6
    bc_eps: float = 1e-6
    td_eps: float = 1e-3
    dtype: torch.dtype = torch.float64
    device: str = "cpu"


# %% [markdown]
# ## Gaussian-mixture builders

# %%
def make_standard_target_gmm(
    case: str = "triangular_3mode",
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> GaussianMixture:
    """
    Build a standard 2D terminal target GMM used across experiments.

    Cases
    -----
    triangular_3mode : three moderately separated modes
    bimodal_right    : two modes near the right side of the domain
    single           : one Gaussian mode
    """
    if dtype is None:
        dtype = default_dtype()
    if device is None:
        device = "cpu"

    if case == "triangular_3mode":
        weights = torch.tensor([0.35, 0.30, 0.35], dtype=dtype, device=device)
        means = torch.tensor(
            [[2.2,  0.0],
             [3.1,  0.9],
             [3.1, -0.9]],
            dtype=dtype, device=device
        )
        covs = torch.stack([
            torch.tensor([[0.10, 0.00], [0.00, 0.12]], dtype=dtype, device=device),
            torch.tensor([[0.12, 0.02], [0.02, 0.10]], dtype=dtype, device=device),
            torch.tensor([[0.12, -0.02], [-0.02, 0.10]], dtype=dtype, device=device),
        ], dim=0)

    elif case == "bimodal_right":
        weights = torch.tensor([0.5, 0.5], dtype=dtype, device=device)
        means = torch.tensor(
            [[2.8,  0.7],
             [2.8, -0.7]],
            dtype=dtype, device=device
        )
        covs = torch.stack([
            torch.tensor([[0.12, 0.00], [0.00, 0.10]], dtype=dtype, device=device),
            torch.tensor([[0.12, 0.00], [0.00, 0.10]], dtype=dtype, device=device),
        ], dim=0)

    elif case == "single":
        weights = torch.tensor([1.0], dtype=dtype, device=device)
        means = torch.tensor([[3.0, 0.0]], dtype=dtype, device=device)
        covs = torch.tensor([[[0.15, 0.0], [0.0, 0.15]]], dtype=dtype, device=device)

    else:
        raise ValueError(f"Unknown target GMM case: {case}")

    return GaussianMixture(weights=weights, means=means, covs=covs)


def make_initial_point(
    x0: Sequence[float] = (0.0, 0.0),
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Deterministic 2D starting point."""
    if dtype is None:
        dtype = default_dtype()
    if device is None:
        device = "cpu"
    return torch.tensor(x0, dtype=dtype, device=device)


def make_initial_gmm(
    case: str = "two_entrances",
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> GaussianMixture:
    """
    Build initial GMM for E3.

    Cases
    -----
    two_entrances : two narrow modes above/below a common corridor entry
    three_entrances : optional three-mode variant
    """
    if dtype is None:
        dtype = default_dtype()
    if device is None:
        device = "cpu"

    if case == "two_entrances":
        weights = torch.tensor([0.5, 0.5], dtype=dtype, device=device)
        means = torch.tensor(
            [[-1.0,  0.8],
             [-1.0, -0.8]],
            dtype=dtype, device=device
        )
        cov = torch.tensor([[0.06, 0.0], [0.0, 0.06]], dtype=dtype, device=device)
        covs = torch.stack([cov, cov], dim=0)

    elif case == "three_entrances":
        weights = torch.tensor([0.3, 0.4, 0.3], dtype=dtype, device=device)
        means = torch.tensor(
            [[-1.1,  1.0],
             [-1.2,  0.0],
             [-1.1, -1.0]],
            dtype=dtype, device=device
        )
        cov = torch.tensor([[0.05, 0.0], [0.0, 0.05]], dtype=dtype, device=device)
        covs = torch.stack([cov, cov, cov], dim=0)

    else:
        raise ValueError(f"Unknown initial GMM case: {case}")

    return GaussianMixture(weights=weights, means=means, covs=covs)


def sample_gmm(
    gmm: GaussianMixture,
    n_samples: int,
    *,
    seed: int = 0,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Sample from a GaussianMixture using torch.distributions.

    Returns
    -------
    samples : (n_samples, d)
    """
    if dtype is None:
        dtype = gmm.weights.dtype
    if device is None:
        device = gmm.weights.device

    gmm = gmm.to(device=device, dtype=dtype)
    gen = torch.Generator(device=torch.device(str(device)) if str(device) != "cpu" else torch.device("cpu"))
    gen.manual_seed(seed)

    comp_ids = torch.multinomial(gmm.weights, num_samples=n_samples, replacement=True, generator=gen)
    d = gmm.d
    out = torch.empty(n_samples, d, dtype=dtype, device=device)

    for m in range(gmm.M):
        idx = torch.where(comp_ids == m)[0]
        if idx.numel() == 0:
            continue
        mean = gmm.means[m]
        cov = gmm.covs[m]
        chol = torch.linalg.cholesky(cov)
        z = torch.randn((idx.numel(), d), generator=gen, dtype=dtype, device=device)
        out[idx] = mean + z @ chol.T

    return out


# %% [markdown]
# ## 2D centerlines and corridor geometry

# %%
def _normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize row vectors."""
    nrm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / nrm


def _finite_difference_tangent(xy: torch.Tensor) -> torch.Tensor:
    """Centered finite-difference tangent field."""
    dx = torch.zeros_like(xy)
    dx[1:-1] = 0.5 * (xy[2:] - xy[:-2])
    dx[0] = xy[1] - xy[0]
    dx[-1] = xy[-1] - xy[-2]
    return _normalize_rows(dx)


def make_centerline(
    case: str,
    t_grid: torch.Tensor,
    *,
    width: float | Callable[[torch.Tensor], torch.Tensor] = 0.25,
) -> Centerline2D:
    """
    Construct standard 2D centerlines.

    Cases
    -----
    straight
    curved_arc
    curved_s
    corridor_main
    """
    t = t_grid
    x = -0.8 + 3.8 * t

    if case == "straight":
        y = torch.zeros_like(t)
    elif case == "curved_arc":
        y = 0.8 * torch.sin(0.9 * math.pi * t)
    elif case == "curved_s":
        y = 0.9 * torch.sin(2.0 * math.pi * (t - 0.15)) * torch.exp(-0.3 * (t - 0.5) ** 2)
    elif case == "corridor_main":
        y = 0.65 * torch.sin(math.pi * (t - 0.10)) - 0.25 * torch.sin(2.0 * math.pi * (t - 0.10))
    else:
        raise ValueError(f"Unknown centerline case: {case}")

    xy = torch.stack([x, y], dim=-1)
    tangent = _finite_difference_tangent(xy)
    normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)

    if callable(width):
        width_t = width(t)
    elif isinstance(width, (torch.Tensor, np.ndarray, list, tuple)):
        width_t = torch.as_tensor(width, dtype=t.dtype, device=t.device)
        if width_t.shape != t.shape:
            raise ValueError(
                f"Width profile must have shape {tuple(t.shape)}, got {tuple(width_t.shape)}"
            )
    else:
        width_t = torch.full_like(t, float(width))

    return Centerline2D(t=t, xy=xy, tangent=tangent, normal=normal, width=width_t)


def make_width_profile(
    t_grid: torch.Tensor,
    *,
    kind: str = "constant",
    w0: float = 0.25,
    w1: float = 0.18,
) -> torch.Tensor:
    """
    Build standard corridor half-width profiles.
    """
    t = t_grid
    if kind == "constant":
        return torch.full_like(t, w0)
    if kind == "narrowing":
        return w0 + (w1 - w0) * t
    if kind == "bottleneck":
        return w0 - (w0 - w1) * torch.exp(-((t - 0.55) / 0.18) ** 2)
    raise ValueError(f"Unknown width profile kind: {kind}")


def _match_times_to_grid(query_t: torch.Tensor, grid_t: torch.Tensor) -> torch.Tensor:
    """Match query times to nearest indices on a fixed grid."""
    dist = torch.abs(query_t.unsqueeze(-1) - grid_t.unsqueeze(0))
    return torch.argmin(dist, dim=-1)


def make_oriented_beta_field(
    centerline: Centerline2D,
    breaks: torch.Tensor,
    *,
    b_parallel: float,
    b_perp: float,
) -> torch.Tensor:
    """
    Build a piecewise-constant SPD matrix field aligned with the local
    tangent/normal directions of the centerline.
    """
    device = breaks.device
    dtype = breaks.dtype
    K = breaks.numel() - 1
    beta = torch.empty(K, 2, 2, dtype=dtype, device=device)

    t_mid = 0.5 * (breaks[:-1] + breaks[1:])
    idx = _match_times_to_grid(t_mid, centerline.t)

    for k in range(K):
        tau = centerline.tangent[idx[k]]
        nor = centerline.normal[idx[k]]
        R = torch.stack([tau, nor], dim=-1)
        D = torch.diag(torch.tensor([b_parallel, b_perp], dtype=dtype, device=device))
        beta[k] = R @ D @ R.T

    return beta


def make_scalar_beta_field(
    breaks: torch.Tensor,
    *,
    beta_scalar: float,
) -> torch.Tensor:
    """Build isotropic piecewise-constant beta matrices."""
    K = breaks.numel() - 1
    dtype = breaks.dtype
    device = breaks.device
    I = torch.eye(2, dtype=dtype, device=device)
    return beta_scalar * I.unsqueeze(0).repeat(K, 1, 1)


def sample_centerline_at_breaks(centerline: Centerline2D, breaks: torch.Tensor) -> torch.Tensor:
    """Produce piecewise-constant protocol centers nu_k from centerline geometry."""
    t_mid = 0.5 * (breaks[:-1] + breaks[1:])
    idx = _match_times_to_grid(t_mid, centerline.t)
    return centerline.xy[idx]


# %% [markdown]
# ## Sigma builders

# %%
def make_sigma_matrix(
    case: str = "none",
    strength: float = 0.5,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Build a 2x2 linear drift matrix sigma.
    """
    if dtype is None:
        dtype = default_dtype()
    if device is None:
        device = "cpu"

    z = torch.zeros((2, 2), dtype=dtype, device=device)
    if case == "none":
        return z
    if case == "rotation":
        w = float(strength)
        return torch.tensor([[0.0, -w], [w, 0.0]], dtype=dtype, device=device)
    if case == "stretch":
        a = float(strength)
        return torch.tensor([[a, 0.0], [0.0, -a]], dtype=dtype, device=device)
    if case == "shear":
        s = float(strength)
        return torch.tensor([[0.0, s], [0.0, 0.0]], dtype=dtype, device=device)
    raise ValueError(f"Unknown sigma case: {case}")


def expand_sigma_over_breaks(breaks: torch.Tensor, sigma_matrix: torch.Tensor) -> torch.Tensor:
    """Repeat a single 2x2 sigma matrix over all PWC intervals."""
    K = breaks.numel() - 1
    return sigma_matrix.unsqueeze(0).repeat(K, 1, 1)


# %% [markdown]
# ## Protocol builders

# %%
def make_breaks(
    K: int,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Uniform PWC breaks on [0,1]."""
    if dtype is None:
        dtype = default_dtype()
    if device is None:
        device = "cpu"
    return torch.linspace(0.0, 1.0, K + 1, dtype=dtype, device=device)


def make_protocol_path_shaping(
    *,
    K: int,
    centerline_case: str,
    beta_scalar: float,
    sigma_case: str = "none",
    sigma_strength: float = 0.0,
    td_eps: float = 1e-3,
    n_geom_steps: int = 401,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> Tuple[MatrixPWCProtocol, Centerline2D]:
    """Construct a simple isotropic path-shaping protocol for E1."""
    if dtype is None:
        dtype = default_dtype()
    if device is None:
        device = "cpu"

    breaks = make_breaks(K, dtype=dtype, device=device)
    t_geom = make_time_grid(n_geom_steps - 1, dtype=dtype, device=device)
    centerline = make_centerline(centerline_case, t_geom, width=0.25)
    nu = sample_centerline_at_breaks(centerline, breaks)
    beta = make_scalar_beta_field(breaks, beta_scalar=beta_scalar)
    sigma = expand_sigma_over_breaks(
        breaks, make_sigma_matrix(sigma_case, sigma_strength, dtype=dtype, device=device)
    )
    protocol = MatrixPWCProtocol(
        breaks=breaks,
        sigma=sigma,
        beta=beta,
        nu=nu,
        time_domain=TimeDomain(eps=td_eps),
    )
    return protocol, centerline


def make_protocol_corridor(
    *,
    K: int,
    centerline_case: str = "corridor_main",
    width_kind: str = "bottleneck",
    width0: float = 0.28,
    width1: float = 0.16,
    isotropic_beta: Optional[float] = None,
    b_parallel: Optional[float] = None,
    b_perp: Optional[float] = None,
    sigma_case: str = "none",
    sigma_strength: float = 0.0,
    td_eps: float = 1e-3,
    n_geom_steps: int = 401,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> Tuple[MatrixPWCProtocol, Centerline2D]:
    """
    Construct a corridor-following protocol.
    """
    if dtype is None:
        dtype = default_dtype()
    if device is None:
        device = "cpu"

    if (isotropic_beta is None) == (b_parallel is None or b_perp is None):
        raise ValueError("Specify either isotropic_beta, or both b_parallel and b_perp.")

    breaks = make_breaks(K, dtype=dtype, device=device)
    t_geom = make_time_grid(n_geom_steps - 1, dtype=dtype, device=device)
    width = make_width_profile(t_geom, kind=width_kind, w0=width0, w1=width1)
    centerline = make_centerline(centerline_case, t_geom, width=width)
    nu = sample_centerline_at_breaks(centerline, breaks)

    if isotropic_beta is not None:
        beta = make_scalar_beta_field(breaks, beta_scalar=float(isotropic_beta))
    else:
        beta = make_oriented_beta_field(centerline, breaks, b_parallel=float(b_parallel), b_perp=float(b_perp))

    sigma = expand_sigma_over_breaks(
        breaks, make_sigma_matrix(sigma_case, sigma_strength, dtype=dtype, device=device)
    )

    protocol = MatrixPWCProtocol(
        breaks=breaks,
        sigma=sigma,
        beta=beta,
        nu=nu,
        time_domain=TimeDomain(eps=td_eps),
    )
    return protocol, centerline


# %% [markdown]
# ## Controller / simulation wrappers

# %%
def build_controller(
    *,
    protocol: MatrixPWCProtocol,
    target: GaussianMixture,
    x0: Sequence[float] | torch.Tensor,
    bc_eps: float = 1e-6,
) -> LQGMPID:
    """Build an LQGMPID controller for a deterministic initial point."""
    if not isinstance(x0, torch.Tensor):
        x0 = torch.tensor(x0, dtype=protocol.dtype, device=protocol.device)
    else:
        x0 = x0.to(device=protocol.device, dtype=protocol.dtype)

    target = target.to(device=protocol.device, dtype=protocol.dtype)
    return LQGMPID(protocol=protocol, target=target, x0=x0, bc_eps=bc_eps)


def run_simulation(
    controller: LQGMPID,
    *,
    B: int = 2000,
    n_steps: int = 400,
    seed: int = 12345,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
):
    """
    Standard simulation wrapper.
    Returns times, traj, xT, sim.
    """
    if dtype is None:
        dtype = controller.protocol.dtype
    if device is None:
        device = controller.protocol.device

    sim = controller.simulate(B=B, n_steps=n_steps, seed=seed, dtype=dtype, device=str(device))
    times = sim.times.detach().cpu()
    traj = sim.traj.detach().cpu()
    xT = traj[-1]
    return times, traj, xT, sim


def run_simulation_from_multiple_starts(
    *,
    protocol: MatrixPWCProtocol,
    target: GaussianMixture,
    x0_samples: torch.Tensor,
    n_steps: int = 400,
    seed: int = 12345,
    bc_eps: float = 1e-6,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper for E3 when the starting points are sampled from an
    initial law and we still use the deterministic-start controller interface.
    """
    if dtype is None:
        dtype = protocol.dtype
    if device is None:
        device = protocol.device

    x0_samples = x0_samples.to(device=device, dtype=dtype)
    B, d = x0_samples.shape
    all_traj = []
    times_ref = None

    for b in range(B):
        controller = build_controller(protocol=protocol, target=target, x0=x0_samples[b], bc_eps=bc_eps)
        times, traj, _, _ = run_simulation(controller, B=1, n_steps=n_steps, seed=seed + b, dtype=dtype, device=device)
        if times_ref is None:
            times_ref = times
        all_traj.append(traj[:, 0, :])

    traj = torch.stack(all_traj, dim=1)
    xT = traj[-1]
    return times_ref, traj, xT

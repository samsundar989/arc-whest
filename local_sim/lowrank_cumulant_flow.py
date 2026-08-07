"""Fenced low-rank cumulant-flow adapter (GPU research only).

This module adapts ARC's ``mlp_cumulant_propagation`` implementation without
changing ``estimator.py``.  Orders one and two remain full-width.  After every
nonlinearity, orders three and four are projected onto the leading eigenspace
of the propagated covariance.  Optional layerwise scalar renormalizers are
applied only to those higher-order cumulants.

The upstream package is intentionally a research-only dependency.  Point
``MLP_KPROP_SRC`` at its ``src`` directory, or install it in the GPU venv.
The adapter prefers the upstream factored representations, so a rank-r
directional restriction normally does not materialize an n^4 tensor.  It cannot
always: when upstream hands back an order-4 ``HTensor`` rather than a
``FactoredTensor4`` there is no factored form to project, and densifying it at
challenge width would be 256^4 doubles = 34 GB.  That case is now refused with a
diagnostic instead of exhausting the device (MOONSHOT G0, 2026-08-06); it is
reached at ``k_max=3``, where upstream still carries a degree-4 slot that the
order-3 flow has no business projecting at all.

This is infrastructure for the pre-registered D08/D09 gates in
``docs/MOONSHOT.md``.  It is not a score claim and must never be imported by the
submission estimator.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class LayerProjection:
    """Cheap audit record emitted for every projected activation layer."""

    layer: int
    requested_rank: int
    effective_rank: int
    covariance_trace: float
    retained_covariance_fraction: float
    k3_representation: str | None
    k4_representation: str | None
    k3_factor_columns: int | None
    k4_factor_columns: int | None
    k3_scale: float
    k4_scale: float


def _load_upstream() -> dict[str, Any]:
    src = os.environ.get("MLP_KPROP_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)
    try:
        from mlp_kprop.factor_k3 import FactoredTensor
        from mlp_kprop.factor_k4 import FactoredTensor4
        from mlp_kprop.harmonic import HTensor
        from mlp_kprop.kprop_harmonic import (
            SIMPLE,
            clone_tower,
            coerce_input,
            linear_kprop,
            metric_needed,
            nonlin_kprop,
        )
        from mlp_kprop.wick import WICK_COEF_D
    except ImportError as exc:  # pragma: no cover - exercised by CLI setup
        raise RuntimeError(
            "ARC mlp_cumulant_propagation is unavailable. Install it in the "
            "GPU venv or set MLP_KPROP_SRC to the checkout's src directory."
        ) from exc
    return {
        "FactoredTensor": FactoredTensor,
        "FactoredTensor4": FactoredTensor4,
        "HTensor": HTensor,
        "SIMPLE": SIMPLE,
        "clone_tower": clone_tower,
        "coerce_input": coerce_input,
        "linear_kprop": linear_kprop,
        "metric_needed": metric_needed,
        "nonlin_kprop": nonlin_kprop,
        "WICK_COEF_D": WICK_COEF_D,
    }


def leading_covariance_basis(covariance: torch.Tensor, rank: int) -> tuple[torch.Tensor, float]:
    """Return top covariance eigenvectors and their fraction of positive trace."""

    covariance = (covariance + covariance.mT) * 0.5
    evals, evecs = torch.linalg.eigh(covariance)
    effective_rank = min(rank, covariance.shape[0])
    basis = evecs[:, -effective_rank:]
    positive = torch.clamp(evals, min=0.0)
    total = positive.sum()
    retained = positive[-effective_rank:].sum()
    total_value = float(total.detach())
    fraction = 1.0 if total_value == 0.0 else float((retained / total).detach())
    return basis, fraction


def _project_matrix_factors(factors: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Apply U U^T to both spatial axes of (n,n,R) factors."""

    reduced = torch.einsum("ia,ijq,jb->abq", basis, factors, basis)
    return torch.einsum("ia,abq,jb->ijq", basis, reduced, basis)


def _project_dense(tensor: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Orthogonally project every axis of a small order-3/4 dense tensor."""

    if tensor.ndim == 3:
        core = torch.einsum("ia,jb,kc,ijk->abc", basis, basis, basis, tensor)
        return torch.einsum("ia,jb,kc,abc->ijk", basis, basis, basis, core)
    if tensor.ndim == 4:
        core = torch.einsum("ia,jb,kc,ld,ijkl->abcd", basis, basis, basis, basis, tensor)
        return torch.einsum("ia,jb,kc,ld,abcd->ijkl", basis, basis, basis, basis, core)
    raise ValueError(f"only order-3/4 tensors are supported, got order {tensor.ndim}")


def _refuse_dense(degree: int, width: int, max_elems: int) -> None:
    """Fail loudly rather than OOM when a dense projection cannot fit.

    Discovered at MOONSHOT G0: at ``k_max=3`` upstream's tower still carries a
    degree-4 entry, and it arrives as an ``HTensor``, so the dense fallback below
    tried to build a 256^4 tensor (34 GB) and took the device down. Silent OOM in
    a research lane is worse than an exception, because it looks like "the method
    does not scale" when the truth is "this adapter projected something it should
    not have touched".
    """
    elems = width ** degree
    if elems > max_elems:
        raise RuntimeError(
            f"refusing to densify an order-{degree} cumulant at width {width}: "
            f"{elems:,} elements exceeds max_dense_elems={max_elems:,}. "
            "Upstream returned an unfactored representation; either raise the "
            "cap deliberately or restrict the projection with k_max."
        )


def project_higher_cumulants(
    tower: Mapping[int, Any],
    rank: int,
    *,
    k3_scale: float = 1.0,
    k4_scale: float = 1.0,
    k_max: int = 4,
    max_dense_elems: int = 1 << 28,
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Project K3/K4 into the top-r eigenspace of the full K2 covariance.

    The returned tower keeps ARC-compatible objects, allowing its next exact
    linear/nonlinear step to run unchanged.  Factored K4 is projected without
    ever constructing its n^4 dense form.

    ``k_max`` bounds which degrees are touched.  This matters: upstream's tower
    can carry a degree-4 slot even when the flow is order-3, and projecting it
    is both outside the method's definition and, at challenge width, impossible.
    """

    up = _load_upstream()
    FactoredTensor = up["FactoredTensor"]
    FactoredTensor4 = up["FactoredTensor4"]
    HTensor = up["HTensor"]

    out = {degree: value.clone() for degree, value in tower.items()}
    covariance = out[2].to_tensor() if hasattr(out[2], "to_tensor") else out[2]
    basis, retained = leading_covariance_basis(covariance, rank)
    diag: dict[str, Any] = {
        "effective_rank": basis.shape[1],
        "covariance_trace": float(torch.trace(covariance).detach()),
        "retained_covariance_fraction": retained,
        "k3_representation": None,
        "k4_representation": None,
        "k3_factor_columns": None,
        "k4_factor_columns": None,
    }

    if 3 in out and k_max >= 3:
        k3 = out[3]
        if isinstance(k3, FactoredTensor):
            projected = [basis @ (basis.mT @ factor) for factor in k3.factors]
            projected[0] = projected[0] * k3_scale
            out[3] = FactoredTensor(
                n=k3.n,
                d=3,
                factors=tuple(projected),
                device=k3.device,
                dtype=k3.dtype,
            )
            diag["k3_representation"] = "factored-cp"
            diag["k3_factor_columns"] = projected[0].shape[1]
        elif isinstance(k3, HTensor):
            _refuse_dense(3, covariance.shape[0], max_dense_elems)
            dense = _project_dense(k3.to_tensor(), basis) * k3_scale
            out[3] = HTensor(core=dense, r=0)
            diag["k3_representation"] = "dense-projected"
        else:
            _refuse_dense(3, covariance.shape[0], max_dense_elems)
            out[3] = _project_dense(torch.as_tensor(k3), basis) * k3_scale
            diag["k3_representation"] = "dense-projected"

    if 4 in out and k_max >= 4:
        k4 = out[4]
        if isinstance(k4, FactoredTensor4):
            A, B = k4.factors
            A = _project_matrix_factors(A, basis) * k4_scale
            B = _project_matrix_factors(B, basis)
            out[4] = FactoredTensor4(
                n=k4.n,
                factors=(A, B),
                device=k4.device,
                dtype=k4.dtype,
                assume_symmetric=True,
            )
            diag["k4_representation"] = "factored-pair"
            diag["k4_factor_columns"] = A.shape[2]
        elif isinstance(k4, HTensor):
            _refuse_dense(4, covariance.shape[0], max_dense_elems)
            dense = _project_dense(k4.to_tensor(), basis) * k4_scale
            out[4] = HTensor(core=dense, r=0)
            diag["k4_representation"] = "dense-projected"
        else:
            _refuse_dense(4, covariance.shape[0], max_dense_elems)
            out[4] = _project_dense(torch.as_tensor(k4), basis) * k4_scale
            diag["k4_representation"] = "dense-projected"

    return out, diag


def lowrank_mlp_kprop(
    mlp: Any,
    K_in: Mapping[int, torch.Tensor],
    *,
    rank: int,
    k_max: int = 4,
    k3_scales: Sequence[float] | None = None,
    k4_scales: Sequence[float] | None = None,
    kind: Any | None = None,
    use_avg_metric: bool = True,
    output_all: bool = False,
) -> tuple[Any, list[LayerProjection]]:
    """Run ARC k-propagation with a projection/renormalization after each ReLU.

    ``k_max=2`` is admitted for **layer-convention parity only** (MOONSHOT G0).
    At order two there are no higher cumulants, so the projection and the
    renormalizers are no-ops and the run reduces to ARC's own Gaussian closure —
    which is exactly what makes it a clean test of whether this adapter feeds ARC
    the same linear map that ``estimator.py`` propagates. It is not a candidate
    configuration: at ``k_max=2`` the moonshot's whole mechanism is switched off.
    """

    if k_max not in (2, 3, 4):
        raise ValueError("the fenced lane supports k_max 2 (parity only), 3 or 4")
    if rank <= 0:
        raise ValueError("rank must be positive")
    if getattr(mlp, "layernorm", False):
        raise NotImplementedError("upstream cumulant flow does not support layernorm")

    up = _load_upstream()
    if kind is None:
        kind = up["SIMPLE"]
    skip_metric = not up["metric_needed"](k_max, kind)
    K = up["coerce_input"](K_in, k_max=k_max, kind=kind)
    by_layer: OrderedDict[str, Any] = OrderedDict()
    records: list[LayerProjection] = []
    nonlinear_count = len(mlp.nonlins)

    def scale_at(values: Sequence[float] | None, layer: int) -> float:
        return 1.0 if values is None else float(values[layer])

    for layer, W_module in enumerate(mlp.Ws):
        W = W_module.weight
        bias = W_module.bias
        if layer == nonlinear_count:
            K = up["linear_kprop"](K, W, k_max=k_max, bias=bias)
            if output_all:
                by_layer[f"pre{layer}"] = up["clone_tower"](K)
            break

        metric = mlp.init_scale[layer] if use_avg_metric else (1.0 if skip_metric else None)
        K = up["linear_kprop"](K, W, k_max=k_max, set_metric=metric, bias=bias)
        if output_all:
            by_layer[f"pre{layer}"] = up["clone_tower"](K)
        K = up["nonlin_kprop"](
            K,
            nonlin_wick_coef=up["WICK_COEF_D"][mlp.nonlin_names[layer]],
            k_max=k_max,
            kind=kind,
            use_pK=True,
            factor=True,
        )
        K, raw = project_higher_cumulants(
            K,
            rank,
            k3_scale=scale_at(k3_scales, layer),
            k4_scale=scale_at(k4_scales, layer),
            k_max=k_max,
        )
        records.append(
            LayerProjection(
                layer=layer,
                requested_rank=rank,
                k3_scale=scale_at(k3_scales, layer),
                k4_scale=scale_at(k4_scales, layer),
                **raw,
            )
        )
        if output_all:
            by_layer[f"act{layer}"] = up["clone_tower"](K)

    return (by_layer if output_all else K), records


def lowrank_challenge_flow(
    weights: torch.Tensor,
    *,
    rank: int,
    k_max: int = 4,
    k3_scales: Sequence[float] | None = None,
    k4_scales: Sequence[float] | None = None,
    output_all: bool = False,
) -> tuple[Any, list[LayerProjection]]:
    """Challenge-shaped convenience wrapper: every square layer has a ReLU.

    ``weights`` follows whestbench's row-vector convention ``h @ W``; ARC's
    propagation uses column vectors, so each matrix is transposed.  The input is
    standard normal and minted matrices have entry variance ``2 / width``.
    """

    if weights.ndim != 3 or weights.shape[1] != weights.shape[2]:
        raise ValueError("weights must have shape (depth, width, width)")
    depth, width, _ = weights.shape
    layers = [SimpleNamespace(weight=W.mT, bias=None) for W in weights]
    challenge = SimpleNamespace(
        Ws=layers,
        nonlins=[None] * depth,
        nonlin_names=["relu"] * depth,
        init_scale=[2.0] * depth,
        layernorm=False,
    )
    K_in = {
        1: torch.zeros(width, device=weights.device, dtype=weights.dtype),
        2: torch.eye(width, device=weights.device, dtype=weights.dtype),
    }
    return lowrank_mlp_kprop(
        challenge,
        K_in,
        rank=rank,
        k_max=k_max,
        k3_scales=k3_scales,
        k4_scales=k4_scales,
        output_all=output_all,
    )
def _smoke(rank: int, width: int) -> list[dict[str, Any]]:
    """Check that projected factors have no component outside span(U)."""

    up = _load_upstream()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    gen = torch.Generator(device=device).manual_seed(20260806)
    q, _ = torch.linalg.qr(torch.randn(width, width, generator=gen, device=device, dtype=dtype))
    spectrum = torch.linspace(2.0, 0.1, width, device=device, dtype=dtype)
    cov = (q * spectrum) @ q.mT
    columns = min(7, width)
    k3_factors = tuple(
        torch.randn(width, columns, generator=gen, device=device, dtype=dtype)
        for _ in range(3)
    )
    A = torch.randn(width, width, columns, generator=gen, device=device, dtype=dtype)
    B = torch.randn(width, width, columns, generator=gen, device=device, dtype=dtype)
    tower = {
        1: up["HTensor"](core=torch.zeros(width, device=device, dtype=dtype), r=0),
        2: up["HTensor"](core=cov, r=0),
        3: up["FactoredTensor"](n=width, d=3, factors=k3_factors),
        4: up["FactoredTensor4"](n=width, factors=(A, B)),
    }
    projected, diag = project_higher_cumulants(tower, rank, k3_scale=0.9, k4_scale=1.1)
    basis, _ = leading_covariance_basis(cov, rank)
    P = basis @ basis.mT
    errors = []
    for degree in (3, 4):
        obj = projected[degree]
        factors = obj.factors
        for factor in factors:
            if factor.ndim == 2:
                residual = factor - P @ factor
            else:
                residual = factor - _project_matrix_factors(factor, basis)
            errors.append(float(torch.linalg.norm(residual) / torch.clamp(torch.linalg.norm(factor), min=1e-30)))
    if max(errors) > 1e-10:
        raise AssertionError(f"projection residual too large: {max(errors):.3e}")
    diag["max_out_of_basis_relative_norm"] = max(errors)
    diag["device"] = str(device)
    return [diag]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run projection invariants only")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--width", type=int, default=16)
    args = parser.parse_args()
    if not args.smoke:
        parser.error("only --smoke is exposed; D08/D09 runners are added after calibration data minting")
    for result in _smoke(args.rank, args.width):
        print(asdict(result) if isinstance(result, LayerProjection) else result)


if __name__ == "__main__":
    main()

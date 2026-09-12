"""Truth-free four-coordinate Gaussian/ReLU source and signed transport fixture.

The fixed one-factor conditional integral and direct support-aware marginal
density quadrature use independent density and moment implementations.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad


A = np.array([0.4, 0.5, 0.6, 0.7], dtype=float)
R = np.diag(1.0 - A*A) + np.outer(A, A)
ORDERS = (32, 48, 64)
LIMIT = 12.0
TOL_QUAD = 1e-8
TOL_ID = 1e-10
MATERIAL = 1e-6
TAIL = math.sqrt(420.0 * 0.5 * math.erfc(12.0 / math.sqrt(2.0)))
MULTI = tuple(p for p in itertools.product(range(5), repeat=4) if sum(p) <= 4)
FOUR = tuple(itertools.product(range(4), repeat=4))
H = np.array([[1, 1, 1, 1], [1, -1, 1, -1],
              [1, 1, -1, -1], [1, -1, -1, 1]], dtype=float) / 2.0


def half_normal_powers(b: float, s: float) -> tuple[float, ...]:
    t = b / s
    q0 = 0.5 * math.erfc(-t / math.sqrt(2.0))
    q1 = b*q0 + s*math.exp(-t*t/2.0)/math.sqrt(2.0*math.pi)
    out = [1.0, q1]
    previous, current = q0, q1
    for power in range(2, 5):
        next_value = b*current + (power-1)*s*s*previous
        out.append(next_value)
        previous, current = current, next_value
    return tuple(out)


def latent_raw() -> tuple[dict[tuple[int, ...], float], dict[tuple[int, ...], float]]:
    scales = np.sqrt(1.0 - A*A)
    def value(u: float, powers: tuple[int, ...]) -> float:
        factor = math.exp(-u*u/2.0)/math.sqrt(2.0*math.pi)
        for i, power in enumerate(powers):
            if power:
                factor *= half_normal_powers(A[i]*u, scales[i])[power]
        return factor
    moments = {}
    errors = {}
    for powers in MULTI:
        if not any(powers):
            moments[powers], errors[powers] = 1.0, 0.0
            continue
        neg, eneg = quad(value, -np.inf, 0.0, args=(powers,),
                         epsabs=1e-13, epsrel=1e-13)
        pos, epos = quad(value, 0.0, np.inf, args=(powers,),
                         epsabs=1e-13, epsrel=1e-13)
        moments[powers] = neg+pos
        errors[powers] = eneg+epos
    return moments, errors


def latent_composite(powers: tuple[int, ...], order: int) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    scales = np.sqrt(1.0-A*A)
    total = 0.0
    for left, right in ((-12, -6), (-6, -2), (-2, 0),
                        (0, 2), (2, 6), (6, 12)):
        u = (right-left)*nodes/2.0 + (right+left)/2.0
        for x, weight in zip(u, weights):
            factor = math.exp(-x*x/2.0)/math.sqrt(2.0*math.pi)
            for i, power in enumerate(powers):
                if power:
                    factor *= half_normal_powers(A[i]*x, scales[i])[power]
            total += weight*(right-left)/2.0*factor
    return total


def active_powers(support: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(p for p in MULTI if all((p[i] > 0) == (i in support) for i in range(4)))


def density_grid(x: np.ndarray, q: np.ndarray, norm: float) -> np.ndarray:
    # x has one row per point. This route never uses the latent factor A.
    quadratic = np.einsum("ni,ij,nj->n", x, q, x, optimize=True)
    return norm*np.exp(-quadratic/2.0)


def direct_raw(order: int, limit: float = LIMIT) -> tuple[dict[tuple[int, ...], float], int]:
    nodes0, weights0 = np.polynomial.legendre.leggauss(order)
    nodes = limit*(nodes0+1.0)/2.0
    weights = limit*weights0/2.0
    output = {(0, 0, 0, 0): 1.0}
    point_count = 0
    for size in range(1, 5):
        for support in itertools.combinations(range(4), size):
            selected = active_powers(support)
            sub = R[np.ix_(support, support)]
            inverse = np.linalg.inv(sub)
            norm = 1.0 / ((2.0*math.pi)**(size/2.0)*math.sqrt(np.linalg.det(sub)))
            totals = np.zeros(len(selected), dtype=float)
            compensation = np.zeros(len(selected), dtype=float)
            # The final one or two dimensions fit a small tile. Every tile has
            # at most 64**2=4096 nodes, well below the registered 65536 cap.
            trailing = min(size, 2)
            tail_index = list(itertools.product(range(order), repeat=trailing))
            tail_x = nodes[np.asarray(tail_index, dtype=int)]
            tail_w = np.prod(weights[np.asarray(tail_index, dtype=int)], axis=1)
            prefix_size = size-trailing
            for prefix in itertools.product(range(order), repeat=prefix_size):
                x = np.empty((len(tail_w), size), dtype=float)
                for axis, index in enumerate(prefix):
                    x[:, axis] = nodes[index]
                x[:, prefix_size:] = tail_x
                mass = density_grid(x, inverse, norm)*tail_w
                for index in prefix:
                    mass *= weights[index]
                for slot, powers in enumerate(selected):
                    monomial = np.ones(len(mass), dtype=float)
                    for axis, global_index in enumerate(support):
                        monomial *= x[:, axis]**powers[global_index]
                    tile_sum = math.fsum((mass*monomial).tolist())
                    y = tile_sum-compensation[slot]
                    t = totals[slot]+y
                    compensation[slot] = (t-totals[slot])-y
                    totals[slot] = t
            point_count += order**size
            for powers, total in zip(selected, totals):
                output[powers] = float(total)
    assert len(output) == len(MULTI) == 70
    return output, point_count


def key(indices: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(indices.count(i) for i in range(4))


def partitions(items: tuple[int, ...]) -> list[tuple[tuple[int, ...], ...]]:
    if not items:
        return [()]
    first, rest = items[0], items[1:]
    result = []
    for partition in partitions(rest):
        result.append(((first,),)+partition)
        for position in range(len(partition)):
            updated = list(partition)
            updated[position] = (first,)+updated[position]
            result.append(tuple(updated))
    return result


PARTITIONS = partitions((0, 1, 2, 3))


def cumulant_partitions(raw: dict, labels: tuple[int, ...]) -> float:
    terms = []
    for blocks in PARTITIONS:
        term = (-1.0)**(len(blocks)-1)*math.factorial(len(blocks)-1)
        for block in blocks:
            term *= raw[key(tuple(labels[position] for position in block))]
        terms.append(term)
    return math.fsum(terms)


def cumulant_central(raw: dict, labels: tuple[int, ...]) -> float:
    means = [raw[key((label,))] for label in labels]
    centered = 0.0
    for mask in range(16):
        subset = tuple(labels[i] for i in range(4) if mask & (1 << i))
        complement = [means[i] for i in range(4) if not mask & (1 << i)]
        centered += (-1.0)**len(complement)*math.prod(complement)*raw[key(subset)]
    def covariance(i: int, j: int) -> float:
        return raw[key((labels[i], labels[j]))] - means[i]*means[j]
    return centered - covariance(0, 1)*covariance(2, 3) - (
        covariance(0, 2)*covariance(1, 3)) - covariance(0, 3)*covariance(1, 2)


def tensor(raw: dict, method) -> np.ndarray:
    result = np.empty((4,)*4)
    for labels in FOUR:
        result[labels] = method(raw, labels)
    return result


def moment_error_bound(raw: dict, uncertainty: dict, labels: tuple[int, ...]) -> float:
    # Interval-product sensitivity over all partition terms. The uncertainty
    # includes the observed 48/64 change, adaptive error, and analytic tail.
    total = 0.0
    for blocks in PARTITIONS:
        coefficient = math.factorial(len(blocks)-1)
        values = [key(tuple(labels[position] for position in block)) for block in blocks]
        high = math.prod(abs(raw[p])+uncertainty[p] for p in values)
        low = math.prod(abs(raw[p]) for p in values)
        total += coefficient*(high-low)
    return total


def scalar_cumulant_from_raw(raw: dict, row: np.ndarray) -> float:
    moments = [1.0]
    for degree in range(1, 5):
        value = 0.0
        for labels in itertools.product(range(4), repeat=degree):
            value += math.prod(row[i] for i in labels)*raw[key(labels)]
        moments.append(value)
    mu = moments[1]
    variance = moments[2]-mu*mu
    central_fourth = moments[4]-4*mu*moments[3]+6*mu*mu*moments[2]-3*mu**4
    return central_fourth-3*variance*variance


def record(checks: dict, name: str, actual: float, expected: float,
           tolerance: float, material: float = MATERIAL) -> None:
    difference = abs(actual-expected)
    relative = difference/abs(expected) if abs(expected) > material else None
    checks[name] = {"actual": float(actual), "expected": float(expected),
                    "absolute_error": float(difference), "relative_error": relative,
                    "tolerance": tolerance,
                    "pass": bool(math.isfinite(actual) and
                                 (relative <= tolerance if relative is not None else difference <= tolerance))}


def lower(checks: dict, name: str, value: float, bound: float) -> None:
    checks[name] = {"actual": float(value), "strict_lower_bound": bound,
                    "pass": bool(value > bound)}


def run() -> dict:
    checks = {}
    latent, adaptive_error = latent_raw()
    direct = {}
    counts = {}
    for order in ORDERS:
        direct[order], counts[order] = direct_raw(order)
    selected = [key((0,)), key((1, 2)), key((0, 0, 1)),
                key((0, 0, 1, 2)), key((0, 1, 2, 3))]
    for powers in selected:
        label = "".join(map(str, powers))
        record(checks, f"raw_{label}_48_64", direct[48][powers], direct[64][powers], TOL_QUAD)
        record(checks, f"raw_{label}_latent", direct[64][powers], latent[powers], TOL_QUAD)
        composite = [latent_composite(powers, n) for n in (48, 72, 96)]
        record(checks, f"raw_{label}_latent_72_96", composite[1], composite[2], TOL_QUAD)
        record(checks, f"raw_{label}_latent_adaptive", composite[2], latent[powers], TOL_QUAD)
    for p in MULTI:
        record(checks, f"all_raw_{''.join(map(str,p))}_latent", direct[64][p], latent[p], TOL_QUAD)
    direct_part = tensor(direct[64], cumulant_partitions)
    direct_cent = tensor(direct[64], cumulant_central)
    latent_part = tensor(latent, cumulant_partitions)
    latent_cent = tensor(latent, cumulant_central)
    for labels in FOUR:
        suffix = "".join(map(str, labels))
        record(checks, f"cumulant_{suffix}_central", direct_cent[labels], direct_part[labels], TOL_ID, 0.0)
        record(checks, f"cumulant_{suffix}_latent_central", latent_cent[labels], latent_part[labels], TOL_ID, 0.0)
    distinct3 = {}
    for repeated in range(4):
        others = [i for i in range(4) if i != repeated]
        for j, k in itertools.combinations(others, 2):
            labels = (repeated, repeated, j, k)
            name = "".join(map(str, labels))
            distinct3[name] = float(direct_part[labels])
            record(checks, f"k_{name}_latent", direct_part[labels], latent_part[labels], TOL_QUAD)
            for permutation in set(itertools.permutations(labels)):
                record(checks, f"k_{name}_perm_{''.join(map(str,permutation))}",
                       direct_part[permutation], direct_part[labels], TOL_ID, 0.0)
    for permutation in itertools.permutations((0, 1, 2, 3)):
        record(checks, f"k0123_perm_{''.join(map(str,permutation))}",
               direct_part[permutation], direct_part[0, 1, 2, 3], TOL_ID, 0.0)
    for labels in ((0, 0, 1, 2), (0, 1, 2, 3)):
        suffix = "".join(map(str, labels))
        record(checks, f"k_{suffix}_latent", direct_part[labels], latent_part[labels], TOL_QUAD)
        # Full cumulant convergence is separately checked; this error bound
        # conservatively propagates estimated raw errors and analytic tail.
        record(checks, f"k_{suffix}_48_64", cumulant_partitions(direct[48], labels),
               direct_part[labels], TOL_QUAD)
    lower(checks, "k0012_positive", float(direct_part[0, 0, 1, 2]), 1e-4)
    lower(checks, "k0123_positive", float(direct_part[0, 1, 2, 3]), 1e-5)
    transported = np.einsum("ia,jb,kc,ld,abcd->ijkl", H, H, H, H, direct_part, optimize=True)
    consumer = np.empty((4,)*4)
    for output_labels in FOUR:
        consumer[output_labels] = math.fsum(
            math.prod(H[output_labels[axis], input_labels[axis]] for axis in range(4))*direct_part[input_labels]
            for input_labels in FOUR)
    for labels in FOUR:
        record(checks, f"transport_{''.join(map(str,labels))}", transported[labels],
               consumer[labels], TOL_ID, 0.0)
    for i in range(4):
        record(checks, f"signed_scalar_d4_{i}", transported[i, i, i, i],
               scalar_cumulant_from_raw(direct[64], H[i]), TOL_ID, 0.0)
        for j in range(4):
            record(checks, f"signed_d22_{i}{j}", transported[i, i, j, j], consumer[i, i, j, j], TOL_ID, 0.0)
            record(checks, f"signed_d31_{i}{j}", transported[i, i, i, j], consumer[i, i, i, j], TOL_ID, 0.0)
    all_distinct = np.zeros_like(direct_part)
    for permutation in itertools.permutations((0, 1, 2, 3)):
        all_distinct[permutation] = direct_part[0, 1, 2, 3]
    all_d4 = np.einsum("ia,ib,ic,id,abcd->i", H, H, H, H, all_distinct, optimize=True)
    for i in range(4):
        expected = 24*math.prod(H[i, j] for j in range(4))*direct_part[0, 1, 2, 3]
        record(checks, f"all_distinct_d4_{i}", all_d4[i], expected, TOL_ID, 0.0)
    lower(checks, "all_distinct_hadamard_gap", float(np.max(np.abs(all_d4))), 1e-5)
    # The signed row with one minus explicitly checks the memo's +/-1.5 factor.
    plus = np.ones(4)/2.0
    minus = np.array([1, 1, 1, -1])/2.0
    signed_readouts = {}
    for name, row in (("plus", plus), ("minus", minus)):
        contracted = math.fsum(math.prod(row[i] for i in labels)*direct_part[labels] for labels in FOUR)
        scalar = scalar_cumulant_from_raw(direct[64], row)
        isolated = 24*math.prod(row)*direct_part[0, 1, 2, 3]
        record(checks, f"{name}_scalar_consumer", contracted, scalar, TOL_ID, 0.0)
        record(checks, f"{name}_isolated", isolated,
               (1.5 if name == "plus" else -1.5)*direct_part[0, 1, 2, 3], TOL_ID, 0.0)
        signed_readouts[name] = {"full": contracted, "isolated_all_distinct": isolated}
    uncertainty = {p: abs(direct[64][p]-direct[48][p])+adaptive_error[p]+TAIL+1e-14 for p in MULTI}
    uncertainty[(0, 0, 0, 0)] = 0.0
    decisive = {}
    for labels in ((0, 0, 1, 2), (0, 1, 2, 3)):
        suffix = "".join(map(str, labels))
        decisive[suffix] = {"direct": float(direct_part[labels]), "latent": float(latent_part[labels]),
                            "direct_48": float(cumulant_partitions(direct[48], labels)),
                            "estimated_cumulant_absolute_error_bound": moment_error_bound(direct[64], uncertainty, labels)}
    failures = [name for name, check in checks.items() if not check["pass"]]
    return {"status": "PASS" if not failures else "FAIL", "checks_count": len(checks),
            "failures": failures, "checks": checks, "a": A.tolist(), "R": R.tolist(),
            "orders": list(ORDERS), "limit": LIMIT, "grid_points_by_order": counts,
            "tail_analytic_bound_per_raw_moment": TAIL,
            "estimated_error_note": "48/64 difference + adaptive estimate + analytic tail + 1e-14 roundoff allowance; convergence estimate is not a certified quadrature bound",
            "raw_direct_64": {"".join(map(str,p)): v for p, v in direct[64].items()},
            "raw_latent": {"".join(map(str,p)): v for p, v in latent.items()},
            "three_distinct": distinct3, "decisive": decisive,
            "hadamard": H.tolist(), "signed_d4": [float(transported[i,i,i,i]) for i in range(4)],
            "signed_d22": [[float(transported[i,i,j,j]) for j in range(4)] for i in range(4)],
            "signed_d31": [[float(transported[i,i,i,j]) for j in range(4)] for i in range(4)],
            "all_distinct_d4": all_d4.tolist(), "signed_readouts": signed_readouts,
            "tolerances": {"quadrature": TOL_QUAD, "identity_absolute": TOL_ID,
                           "material": MATERIAL},
            "scope": "fixed n=4 first-layer correlated Gaussian/ReLU source and exact tiny linear transport; no score, compactness, or deployment claim"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8", newline="\n")
    else:
        print(content, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

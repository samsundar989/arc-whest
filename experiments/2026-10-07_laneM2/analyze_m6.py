"""M6 tables: one-step oracle (m6_oracle.json + .proxy.npz) and end-to-end (m6_e2e.json).

    uv run python experiments/2026-10-07_laneM2/analyze_m6.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LS = ("4", "8", "12", "16")


def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f}"


def oracle() -> list[str]:
    ps = sorted(HERE.glob("m6_oracle_*.json"))
    if not ps:
        return ["(oracle not run)"]
    nets = [n for p in ps for n in json.loads(p.read_text())["nets"]]
    out = [f"### One-step oracle ({len(nets)} DEV nets, n = 1024; mean ± sd over nets)", "",
           "Sizes (relative Frobenius norms):", "",
           "| layer L | E3 ΔD21/D21 (pre L) | E3 ΔD3/D3 | E21a Δcore/core_off | E21b Δcore/core_off | "
           "post-act L ΔD21/D21: E3 | E21a | E21b | **set** | c² |", "|---|---|---|---|---|---|---|---|---|---|"]
    for L in LS:
        g = lambda k: [n["layers"][L][k] for n in nets]  # noqa: E731
        pr = lambda k: [n["layers"][L]["post_rel"][k] for n in nets]  # noqa: E731
        out.append(f"| {L} | {ms(g('e3_d21_rel'))} | {ms(g('e3_d3_rel'))} | {ms(g('e21a_core_rel'))} | "
                   f"{ms(g('e21b_core_rel'))} | {ms(pr('E3'))} | {ms(pr('E21a'))} | {ms(pr('E21b'))} | "
                   f"**{ms(pr('set'))}** | {np.mean(g('c2')):.4f} |")
    out += ["", "Capture = 1 − ‖sub − exact‖²/‖exact‖² (cos² in brackets = capture after the best scalar):", "",
            "| layer L | family | spike1 | spike4 | spike16 | spike64 | gain (β = 0.7) |", "|---|---|---|---|---|---|---|"]
    for L in LS:
        for fam, key in (("E3 ΔD21", "E3_d21_cap"), ("E3 ΔD3", "E3_d3_cap"), ("E21a core", "E21a_cap"),
                         ("E21b core", "E21b_cap")):
            cells = []
            for v in ("spike1", "spike4", "spike16", "spike64", "gain"):
                c = [n["layers"][L][key][v]["capture"] for n in nets]
                s = [n["layers"][L][key][v]["cos2"] for n in nets]
                cells.append(f"{np.mean(c):+.2f} ({np.mean(s):.2f})")
            out.append(f"| {L} | {fam} | " + " | ".join(cells) + " |")
    out += ["", "Set at post-act L (one step later):", "", "| layer L | set capture gain | set capture spike16 | "
            "E21b capture gain | E21b capture spike16 |", "|---|---|---|---|---|"]
    for L in LS:
        c = lambda k: np.mean([n["layers"][L][k]["capture"] for n in nets])  # noqa: E731
        out.append(f"| {L} | {c('post_set_cap_gain'):+.2f} | {c('post_set_cap_spike16'):+.2f} | "
                   f"{c('post_E21b_cap_gain'):+.2f} | {c('post_E21b_cap_spike16'):+.2f} |")
    # ridge proxy, leave-one-net-out
    pzs = sorted(HERE.glob("m6_oracle_*.proxy.npz"))
    if pzs:
        d = {}
        for pz in pzs:
            z = np.load(pz)
            d.update({k: z[k] for k in z.files})
        ids = sorted({int(k.split("_")[0][1:]) for k in d})
        out += ["", "Ridge proxy, held-out R² (leave-one-net-out; 12 features: D21, D21ᵀ, D21∘C (both), "
                "its row/col sums, w2-weighted D21 (both sides), top-4-mode projections, C):", "",
                "| layer L | target E3 ΔD21 (pre L) | target set ΔD21 (post L) | target E21b ΔD21 (post L) |",
                "|---|---|---|---|"]
        for L in LS:
            cells = []
            for y in ("y_e3_pre", "y_set_post", "y_e21b_post"):
                r2 = []
                for h in ids:
                    tr = [i for i in ids if i != h]
                    X = np.concatenate([d[f"n{i}_L{L}_X"] for i in tr])
                    Y = np.concatenate([d[f"n{i}_L{L}_{y}"] for i in tr])
                    mu, sd = X.mean(0), X.std(0) + 1e-30
                    Z = (X - mu) / sd
                    lam = 1e-3 * len(Z)
                    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean()))
                    Xh, Yh = d[f"n{h}_L{L}_X"], d[f"n{h}_L{L}_{y}"]
                    pred = ((Xh - mu) / sd) @ beta + Y.mean()
                    r2.append(1 - ((Yh - pred) ** 2).sum() / ((Yh - Yh.mean()) ** 2).sum())
                cells.append(f"{np.mean(r2):.3f} (min {np.min(r2):.3f})")
            out.append(f"| {L} | " + " | ".join(cells) + " |")
    out += ["", f"materialise: R = {[n['layers']['16']['R'] for n in nets][0]} columns at layer 16, "
            f"{np.mean([n['layers']['16']['materialize_s'] for n in nets]):.1f} s; gain closed form vs dense "
            f"max rel err {max(n['layers'][L]['e21b_gain_closed_rel_err'] for n in nets for L in LS):.1e}"]
    return out


def e2e() -> list[str]:
    ps = sorted(HERE.glob("m6_e2e_[0-9]*.json"))
    if not ps:
        return ["(e2e not run)"]
    ds = [json.loads(p.read_text()) for p in ps]
    nets, arms = [n for d in ds for n in d["nets"]], ds[0]["arms"]
    b = np.array([n["b"]["mse_final"] for n in nets])
    full = np.array([n["E21"]["mse_final"] for n in nets]) if "E21" in arms else None
    out = [f"### End-to-end in ARC's factored chain ({len(nets)} DEV nets, n = 1024)", "",
           "| arm | final MSE | Δ vs b (paired mean) | wins vs b | share of exact-E21 gain | all-layer MSE |",
           "|---|---|---|---|---|---|"]
    for k in arms:
        x = np.array([n[k]["mse_final"] for n in nets])
        dr = (x - b) / b
        sh = (b.mean() - x.mean()) / (b.mean() - full.mean()) if full is not None else np.nan
        xa = np.mean([n[k]["mse_all"] for n in nets])
        out.append(f"| {k} | {x.mean():.3e} | {dr.mean() * 100:+.1f} % ± {2.13 * dr.std(ddof=1) / np.sqrt(len(dr)) * 100:.1f} | "
                   f"{(x < b).sum()}/{len(x)} | {sh * 100:.0f} % | {xa:.3e} |")
    return out


def main() -> int:
    txt = "\n".join(oracle() + [""] + e2e())
    print(txt)
    (HERE / "m6_table.md").write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

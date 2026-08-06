"""
Render the write-up's three headline figures from `figure_data.py`'s output.

Palette and mark conventions follow the project's data-viz reference instance
(validated: adjacent CVD dE 24.7, normal-vision dE 33.6, contrast >= 3:1 against
the light surface). Print figures, so light mode only; one measure per axis —
where two measures of different scale appear they get two panels, never two
y-scales.

Run: uv run python local_sim/make_figures.py --data <figdata.npz> --out docs/figures
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
S1 = "#2a78d6"   # categorical slot 1 (blue)
S2 = "#eb6834"   # categorical slot 2 (orange)
BAND = "#cde2fb"  # blue-100, for the spread band
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    ax.grid(True, color=MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_effective_rank(d, out):
    er = d["eff_rank"]
    layers = np.arange(1, er.shape[1] + 1)
    med = np.median(er, axis=0)
    lo, hi = np.percentile(er, 10, axis=0), np.percentile(er, 90, axis=0)

    fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=200, facecolor=SURFACE)
    style(ax)
    ax.fill_between(layers, lo, hi, color=BAND, alpha=0.9, linewidth=0)
    ax.plot(layers, med, color=S1, linewidth=2.0, solid_capstyle="round")
    ax.set_yscale("log")
    ax.set_xlabel("layer", color=INK2, fontsize=10)
    ax.set_ylabel("effective rank  (participation ratio)", color=INK2, fontsize=10)
    ax.set_title("Activation fluctuations collapse onto a few directions with depth",
                 color=INK, fontsize=11.5, pad=10, loc="left")
    ax.set_xlim(1, layers[-1])
    ax.set_yticks([2, 5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100", "200"])
    for L in (1, 8, 16, 24, 32):
        v = med[L - 1]
        ax.plot([L], [v], "o", color=S1, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2)
        off, ha = {1: ((10, -2), "left"), 32: ((-10, -2), "right")}.get(L, ((0, 11), "center"))
        ax.annotate(f"{v:.0f}", (L, v), textcoords="offset points", xytext=off,
                    ha=ha, fontsize=9, color=INK)
    ax.annotate("shaded band: 10th–90th percentile over 100 MLPs",
                (0.99, 0.93), xycoords="axes fraction", ha="right", fontsize=8.5, color=INK2)
    fig.tight_layout()
    p = os.path.join(out, "fig1_effective_rank.png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_error_vs_n(d, out):
    ns = d["ns"].astype(float)
    mse = d["mse_n"].mean(axis=0)
    cb = ns * float(d["cost"].mean()) / float(d["budget"])
    adj = mse * np.maximum(0.1, cb)
    slope, icpt = np.polyfit(np.log(ns), np.log(mse), 1)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.0), dpi=200, sharex=True, facecolor=SURFACE)
    a, b = axes
    for ax in axes:
        style(ax)

    a.plot(ns, np.exp(icpt) * ns**slope, "--", color=S2, linewidth=1.8,
           label=f"fit:  MSE $\\propto N^{{{slope:.3f}}}$")
    a.plot(ns, mse, "-o", color=S1, linewidth=2.0, markersize=6,
           markeredgecolor=SURFACE, markeredgewidth=1.2, label="measured")
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_ylabel("final-layer MSE", color=INK2, fontsize=10)
    a.set_title("Error falls as $1/N$, so the adjusted score is flat above the floor",
                color=INK, fontsize=11.5, pad=10, loc="left")
    leg = a.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)

    n_floor = 0.1 * float(d["budget"]) / float(d["cost"].mean())
    b.axvspan(ns[0] * 0.85, n_floor, color=BAND, alpha=0.55, linewidth=0)
    b.plot(ns, adj, "-o", color=S1, linewidth=2.0, markersize=6,
           markeredgecolor=SURFACE, markeredgewidth=1.2)
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.set_xlabel("samples per MLP, $N$", color=INK2, fontsize=10)
    b.set_ylabel("adjusted score", color=INK2, fontsize=10)
    b.set_xticks(ns)
    b.set_xticklabels([f"{int(v):,}" for v in ns])
    b.annotate("multiplier pinned at the 0.1 floor:\nextra samples are free",
               (ns[0] * 1.05, adj[0] * 0.55), fontsize=9, color=INK2, ha="left")
    b.annotate("above the floor $N$ cancels:\nadjusted $= V\\!\\cdot\\!c/B$",
               (ns[-1] * 0.95, adj[-1] * 2.6), fontsize=9, color=INK, ha="right")
    b.axvline(n_floor, color=MUTED, linewidth=1.0, linestyle=":")
    for x, y in zip(ns[2:], adj[2:]):
        b.annotate(f"{y:.2e}", (x, y), textcoords="offset points", xytext=(0, 12),
                   ha="center", fontsize=8.5, color=INK)
    b.set_ylim(adj.min() * 0.80, adj.max() * 1.6)
    fig.tight_layout()
    p = os.path.join(out, "fig2_error_vs_n.png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_alpha_by_layer(d, out):
    """Top: per-layer density of alpha, each column scaled to its own peak so the
    late-layer shape is visible at all. Bottom: the three populations the
    estimator actually acts on."""
    al = d["alphas"]                      # (M, depth, width)
    depth = al.shape[1]
    edges = np.linspace(-8, 8, 97)
    H = np.zeros((len(edges) - 1, depth))
    for li in range(depth):
        h, _ = np.histogram(al[:, li, :].ravel(), bins=edges)
        H[:, li] = h / max(h.max(), 1)
    layers = np.arange(1, depth + 1)
    a = al.reshape(al.shape[0], depth, -1)
    dead = (a <= -2.5).mean(axis=(0, 2))
    on = (a > 3.0).mean(axis=(0, 2))
    kink = 1.0 - dead - on
    cmap = LinearSegmentedColormap.from_list("seqblue", ["#ffffff"] + SEQ)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.4, 6.4), dpi=200, sharex=True,
                                 facecolor=SURFACE, gridspec_kw={"height_ratios": [1.25, 1]})
    for x in (ax, bx):
        style(x)
    ax.grid(False)
    im = ax.imshow(H, aspect="auto", origin="lower", cmap=cmap,
                   extent=[0.5, depth + 0.5, edges[0], edges[-1]], vmin=0, vmax=1)
    ax.axhline(-2.5, color=S2, linewidth=1.6, linestyle="--")
    ax.annotate(r"prune below $\tau=-2.5$", (depth * 0.99, -2.4), fontsize=9,
                color=S2, ha="right", va="bottom")
    ax.axhline(3.0, color=INK, linewidth=1.2, linestyle=":")
    ax.annotate(r"$\alpha>3$: ReLU is the identity", (depth * 0.99, 3.2), fontsize=9,
                color=INK, ha="right", va="bottom")
    ax.set_ylabel(r"firing index  $\alpha=\mu_{\rm pre}/\sigma_{\rm pre}$", color=INK2, fontsize=10)
    ax.set_title("Depth widens the firing index — the tails fill, a third stays in the kink",
                 color=INK, fontsize=11.5, pad=10, loc="left")
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("density (per layer, scaled to peak)", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)

    bx.stackplot(layers, dead * 100, kink * 100, on * 100,
                 colors=[S2, "#cfcfca", S1], edgecolor=SURFACE, linewidth=0.6)
    bx.set_xlim(1, depth)
    bx.set_ylim(0, 100)
    bx.set_xlabel("layer", color=INK2, fontsize=10)
    bx.set_ylabel("share of neurons (%)", color=INK2, fontsize=10)
    bx.annotate(f"dead  $\\alpha\\leq-2.5$\n{dead[-1] * 100:.0f}% by layer {depth}",
                (depth * 0.985, dead[-1] * 50), fontsize=9, color=INK, ha="right", va="center")
    bx.annotate(f"kink — must be sampled\n{kink[-1] * 100:.0f}%",
                (depth * 0.985, (dead[-1] + kink[-1] / 2) * 100), fontsize=9,
                color=INK, ha="right", va="center")
    bx.annotate(f"on  $\\alpha>3$\n{on[-1] * 100:.0f}%",
                (depth * 0.985, (1 - on[-1] / 2) * 100), fontsize=9,
                color="#ffffff", ha="right", va="center")
    fig.colorbar(im, ax=bx, pad=0.02, fraction=0.045).ax.set_visible(False)
    fig.tight_layout()
    p = os.path.join(out, "fig3_alpha_by_layer.png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.data)
    for p in (fig_effective_rank(d, args.out), fig_error_vs_n(d, args.out),
              fig_alpha_by_layer(d, args.out)):
        print(f"wrote {p} ({os.path.getsize(p) / 1000:.0f} kB)")


if __name__ == "__main__":
    main()

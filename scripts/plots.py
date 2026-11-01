"""Make the Level 1, 2 and 5 figures from results/*.csv and traces/*.hex."""
import csv
import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ref_model import load_trace  # noqa: E402

RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
WL = [("llm", "LLM decode"), ("cnn", "CNN 3x3 conv"), ("rec", "Recommender")]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {4: ("TAPS", "#2a78d6"), 1: ("Next-line", "#eb6834"), 2: ("Stride", "#1baf7a")}
STREAM_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def read(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def get(rows, w, cfg, key, **match):
    for r in rows:
        if r["workload"] == w and int(r["cfg"]) == cfg and all(int(r[k]) == v for k, v in match.items()):
            return float(r[key])
    raise KeyError((w, cfg, key, match))


def bar_group(ax, values, labels, fmt="{:.2f}"):
    """values: {cfg: [per-workload]}"""
    x = np.arange(len(labels))
    keys = list(values)
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        name, col = SERIES[k]
        bars = ax.bar(x + (i - (len(keys) - 1) / 2) * w, values[k], w - 0.03, color=col, label=name)
        for b, v in zip(bars, values[k]):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=INK2)
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)


# ------------------------------------------------------------------- Level 1
def level1():
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    names = {"llm": ["weights", "act / query", "K cache", "V cache"],
             "cnn": ["input fmap", "output fmap"],
             "rec": ["MLP weights", "embeddings", "dense input"]}
    for j, (w, title) in enumerate(WL):
        sids, lines = load_trace(os.path.join(ROOT, "traces", f"trace_{w}.hex"))
        sids, lines = np.array(sids), np.array(lines)
        step = max(1, len(lines) // 4000)
        ax = axes[0, j]
        for s in sorted(set(sids)):
            idx = np.where(sids == s)[0][::step]
            base = lines[sids == s].min()
            ax.scatter(idx, lines[idx] - base, s=3, color=STREAM_COL[s], label=names[w][s])
        ax.set_title(f"{title}: all {len(lines):,} accesses")
        ax.set_xlabel("access index")
        ax.set_ylabel("line offset within its tensor")
        ax.legend(loc="best", fontsize=8, markerscale=4, framealpha=0.9)

        ax = axes[1, j]
        rows = []
        for s in sorted(set(sids)):
            seq = lines[sids == s]
            d = np.diff(seq)
            d = d[d != 0]
            top = Counter(d.tolist()).most_common(3)
            for delta, c in top:
                if c < 0.01 * len(d):
                    continue
                rows.append((f"{names[w][s]}\n{delta:+d}", 100 * c / max(len(d), 1), STREAM_COL[s]))
        ax.barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1],
                color=[r[2] for r in rows][::-1], height=0.7)
        ax.set_xlim(0, 105)
        ax.set_title("Top deltas per stream (% of that stream's steps)")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Level 1: access patterns, separated by stream", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "level1_trace_analysis.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------- Level 2
def level2():
    rows = read("level2_sweep.csv")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    way_col = {1: "#8a8983", 2: "#eda100", 4: "#2a78d6", 8: "#1baf7a", 16: "#6250d6"}
    for j, (w, title) in enumerate(WL):
        ax = axes[j]
        for ways in (1, 2, 4, 8, 16):
            pol = "lru" if ways == 1 else "plru"
            pts = sorted((int(r["size_kb"]), float(r["hit_rate"])) for r in rows
                         if r["workload"] == w and int(r["ways"]) == ways and r["policy"] == pol)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", lw=2, ms=5,
                    color=way_col[ways], label=f"{ways}-way")
        ax.axvline(16, color=INK2, lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32, 64], ["4", "8", "16", "32", "64"])
        ax.set_title(f"{title}")
        ax.set_xlabel("cache size (KB)")
        ax.set_ylabel("hit rate, no prefetch (%)")
    axes[0].legend(fontsize=8)

    ax = axes[3]
    pols = [("lru", "LRU", "#2a78d6"), ("plru", "tree-PLRU", "#eb6834"), ("random", "random", "#1baf7a")]
    x = np.arange(3)
    for i, (p, lab, col) in enumerate(pols):
        v = [next(float(r["hit_rate"]) for r in rows if r["workload"] == w and r["size_kb"] == "16"
                  and r["ways"] == "4" and r["policy"] == p) for w, _ in WL]
        bars = ax.bar(x + (i - 1) * 0.27, v, 0.25, color=col, label=lab)
        for b, val in zip(bars, v):
            ax.annotate(f"{val:.1f}", (b.get_x() + b.get_width() / 2, val), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [t for _, t in WL])
    ax.set_title("Replacement policy at 16 KB, 4-way")
    ax.set_ylabel("hit rate, no prefetch (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8)
    fig.suptitle("Level 2: cache organisation (reference model, dashed line = chosen 16 KB)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "level2_cache_sweep.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------- Level 5
def level5():
    rows = read("rtl_results.csv")
    labels = [t for _, t in WL]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    cfgs = [1, 2, 4]

    sp = {c: [get(rows, w, 0, "cycles") / get(rows, w, c, "cycles") for w, _ in WL] for c in cfgs}
    bar_group(axes[0], sp, labels, "{:.2f}x")
    axes[0].axhline(1, color=INK2, lw=1)
    axes[0].set_title("Speedup over no prefetching (higher is better)")
    axes[0].set_ylabel("total cycles, no-prefetch / config")

    dr = {c: [get(rows, w, c, "dram_reads") / get(rows, w, 0, "dram_reads") for w, _ in WL] for c in cfgs}
    bar_group(axes[1], dr, labels, "{:.2f}")
    axes[1].axhline(1, color=INK2, lw=1)
    axes[1].set_title("DRAM traffic vs no prefetching (lower is better)")
    axes[1].set_ylabel("DRAM line reads, normalised")

    ac = {c: [get(rows, w, c, "pf_accuracy") for w, _ in WL] for c in cfgs}
    bar_group(axes[2], ac, labels, "{:.0f}")
    axes[2].set_ylim(0, 110)
    axes[2].set_title("Prefetch accuracy (higher is better)")
    axes[2].set_ylabel("useful / issued prefetches (%)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Level 5: RTL results (DRAM 32 cycles, 8 compute cycles between requests, degree 4)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "level5_performance.png"), dpi=150)
    plt.close(fig)


def sweeps():
    specs = [("sweep_latency.csv", "dram_lat", "DRAM latency (cycles)"),
             ("sweep_bandwidth.csv", "issue_gap", "cycles between DRAM requests (less bandwidth ->)"),
             ("sweep_degree.csv", "degree", "prefetch degree (lines ahead)")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for i, (fname, key, xl) in enumerate(specs):
        rows = read(fname)
        xs = sorted({int(r[key]) for r in rows})
        for j, (w, title) in enumerate(WL):
            ax = axes[i, j]
            for c in (4, 1, 2):
                name, col = SERIES[c]
                ys = [get(rows, w, 0, "cycles", **{key: x}) / get(rows, w, c, "cycles", **{key: x}) for x in xs]
                ax.plot(xs, ys, marker="o", lw=2, ms=6, color=col, label=name)
                ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK2)
            ax.axhline(1, color=INK2, lw=1)
            ax.set_xticks(xs)
            ax.set_xlim(xs[0] - 0.05 * (xs[-1] - xs[0]), xs[-1] + 0.22 * (xs[-1] - xs[0]))
            ax.set_xlabel(xl)
            ax.set_ylabel("speedup over no prefetch")
            if i == 0:
                ax.set_title(title)
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Level 5: sensitivity (one parameter varied, others at default)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "level5_sweeps.png"), dpi=150)
    plt.close(fig)


def summary():
    rows = read("rtl_results.csv")
    names = ["No prefetch", "Next-line", "Stride", "Stride + anti-pollution", "TAPS",
             "TAPS + anti-pollution"]
    out = ["| Workload | Config | Hit rate | Avg latency (cyc) | Total cycles | Speedup | "
           "DRAM reads | PF accuracy | Late | Useless |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for w, title in WL:
        base = get(rows, w, 0, "cycles")
        for c in range(6):
            g = lambda k: get(rows, w, c, k)
            acc = f"{g('pf_accuracy'):.1f}%" if c else "-"
            out.append(f"| {title} | {names[c]} | {g('hit_rate'):.1f}% | {g('avg_latency'):.2f} | "
                       f"{int(g('cycles')):,} | {base / g('cycles'):.2f}x | {int(g('dram_reads')):,} | "
                       f"{acc} | {int(g('late'))} | {int(g('pf_useless'))} |")
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    summary()
    level1()
    level2()
    level5()
    sweeps()
    print("figures written to figures/")

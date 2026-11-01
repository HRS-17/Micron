import csv
import numpy as np
import matplotlib.pyplot as plt

data = {}
with open("rtl_results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        wl = row["workload"].strip()
        mode = int(row["mode"])
        if wl not in data:
            data[wl] = {}
        data[wl][mode] = {
            "hit_rate": float(row["hit_rate"]),
            "latency": float(row["avg_latency"]),
            "accuracy": float(row["pf_accuracy"]),
        }

workloads = list(data.keys())
modes = [
    (0, "No Prefetch", "#94a3b8"),
    (1, "Sequential (+1)", "#38bdf8"),
    (2, "Standard Stride", "#fb923c"),
    (3, "TAPS AI-Aware (Ours)", "#9333ea"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=180)
fig.suptitle("Level 5 Hardware RTL Validation: TAPS AI-Aware Cache & Prefetch Engine", fontsize=14, fontweight="bold")

metrics = [
    ("hit_rate", "Cache Hit Rate (%) ↑ [Higher is Better]", (0, 108)),
    ("latency", "Avg Memory Access Latency (Cycles) ↓ [Lower is Better]", (0, 20)),
    ("accuracy", "Prefetch Accuracy (%) ↑ [Higher = Less DRAM Waste]", (0, 108)),
]

x = np.arange(len(workloads))
width = 0.19

for m_idx, (metric_key, title, ylim) in enumerate(metrics):
    ax = axes[m_idx]
    for mode_id, label, color in modes:
        vals = [data[wl][mode_id][metric_key] for wl in workloads]
        offset = (mode_id - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="black", linewidth=0.7)
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels([w.replace("_", " ") for w in workloads], fontweight="bold")
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.3)
    if m_idx == 0:
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

plt.tight_layout()
plt.savefig("level5_performance_validation.png")
print("[OK] Saved Level 5 validation chart -> level5_performance_validation.png")
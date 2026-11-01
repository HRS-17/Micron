import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
LINE_SIZE = 64  # 64-byte cache line (offset = 6 bits)

def save_hex(filename, line_addrs):
    with open(filename, "w") as f:
        for addr in line_addrs:
            byte_addr = int(addr) * LINE_SIZE
            f.write(f"{byte_addr:08x}\n")

# ---------------------------------------------------------
# 1. LLM Workload: Weight Streaming (+1) + KV-Cache Strided (+4) + Block Reuse
# ---------------------------------------------------------
llm_trace = []
# Phase A: Layer weight streaming (stride +1) with short local reuse
for block in range(4):
    base = 0x100 + block * 32
    for i in range(24):
        llm_trace.append(base + i)
        if i % 4 == 3:
            llm_trace.append(base + i - 1)  # activation reuse hit
# Phase B: Multi-Head Attention KV-Cache strided walk (stride +4)
kv_base = 0x400
for i in range(136):
    llm_trace.append(kv_base + i * 4)
llm_trace = llm_trace[:256]

# ---------------------------------------------------------
# 2. CNN 2D Tensor Workload: 2D Tiled Sliding Window
#    4 columns of stride +1, then row-wrap jump of +29 (next row of tensor)
# ---------------------------------------------------------
cnn_trace = []
tensor_base = 0x800
for row in range(64):
    row_start = tensor_base + row * 32
    for col in range(4):  # 3 steps of +1, then wrap jump of +29
        cnn_trace.append(row_start + col)
cnn_trace = cnn_trace[:256]

# ---------------------------------------------------------
# 3. Recommender Workload: Hot Working-Set Embeddings + Random Sparse Lookups + MLP Burst
# ---------------------------------------------------------
rec_trace = []
hot_embeddings = [0x1200 + (i % 4) * 16 + (i // 4) for i in range(16)]  # fits in 4-way cache
# Warm up hot embedding rows
for addr in hot_embeddings:
    rec_trace.append(addr)
# Mix random sparse cold embedding lookups (70%) with hot embedding hits (30%)
for i in range(190):
    if i % 3 == 0:
        rec_trace.append(hot_embeddings[i % len(hot_embeddings)])
    else:
        # Random sparse lookup that maps to the same cache sets (threatens cache pollution!)
        rand_tag = np.random.randint(80, 900)
        rand_set = i % 16
        rec_trace.append((rand_tag << 4) | rand_set)
# Final Dense Bottom-MLP layer (sequential stride +1)
for i in range(50):
    rec_trace.append(0x3000 + i)
rec_trace = rec_trace[:256]

save_hex("trace_llm.hex", llm_trace)
save_hex("trace_cnn.hex", cnn_trace)
save_hex("trace_rec.hex", rec_trace)
print("[OK] Generated trace_llm.hex, trace_cnn.hex, trace_rec.hex (256 accesses each)")

# ---------------------------------------------------------
# Plot Level 1 Analysis Figure
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), dpi=180)
fig.suptitle("Level 1: AI Memory Trace Access Behavior & Stride Delta Analysis", fontsize=14, fontweight="bold")

workloads = [
    ("LLM (Weight Stream + KV Stride)", llm_trace, "#2563eb"),
    ("CNN (2D Tiled Sliding Window)", cnn_trace, "#9333ea"),
    ("Recommender (Sparse + Hot Reuse)", rec_trace, "#dc2626"),
]

for idx, (title, trace, color) in enumerate(workloads):
    arr = np.array(trace)
    deltas = np.diff(arr)

    # Top row: Line Address vs Time
    ax_top = axes[0, idx]
    ax_top.plot(arr[:120], marker=".", markersize=4, linestyle="-", linewidth=1, color=color)
    ax_top.set_title(title, fontweight="bold", fontsize=11)
    ax_top.set_xlabel("Access Index (first 120 reqs)")
    ax_top.set_ylabel("Cache Line Address")
    ax_top.grid(True, alpha=0.3)

    # Bottom row: Stride Delta Distribution
    ax_bot = axes[1, idx]
    unique_d, counts_d = np.unique(deltas, return_counts=True)
    top_indices = np.argsort(counts_d)[::-1][:5]
    labels = [f"Δ={unique_d[i]:+d}" for i in top_indices]
    vals = [counts_d[i] / len(deltas) * 100 for i in top_indices]
    ax_bot.bar(labels, vals, color=color, alpha=0.85, edgecolor="black")
    ax_bot.set_title("Top 5 Address Deltas (% of accesses)", fontsize=10)
    ax_bot.set_ylabel("Frequency (%)")
    ax_bot.set_ylim(0, 100)
    ax_bot.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("level1_trace_analysis.png")
print("[OK] Saved Level 1 chart -> level1_trace_analysis.png")
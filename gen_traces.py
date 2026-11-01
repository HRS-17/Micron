"""Generate memory traces from the loop nests of three AI kernels.

Each trace line is 10 hex digits: 2 digits of stream ID + 8 digits of byte address.
The stream ID says which tensor the access belongs to (input, weights, KV, ...).
An accelerator's DMA/load unit knows this, the same way a CPU prefetcher knows the PC.

Line size is 64 B and every tensor is fp16, so one line = 32 elements.
"""
import argparse
import numpy as np

LINE = 64
EL = 2  # bytes per fp16


def lines(nbytes):
    return (nbytes + LINE - 1) // LINE


class Trace:
    def __init__(self):
        self.sid = []
        self.addr = []

    def rd(self, sid, byte_addr):
        self.sid.append(sid)
        self.addr.append(byte_addr)

    def rd_lines(self, sid, base, first_line, n):
        for i in range(n):
            self.rd(sid, base + (first_line + i) * LINE)

    def save(self, path):
        with open(path, "w") as f:
            for s, a in zip(self.sid, self.addr):
                f.write(f"{s:02x}{a:08x}\n")


# --------------------------------------------------------------------------
# LLM decode step for one layer (one new token)
#   sid 0: weights (streamed, no reuse)    sid 1: activations / query (reused)
#   sid 2: K cache                         sid 3: V cache
# KV cache is token-major: K[t][head][hd], so all heads of a token are contiguous
# (the layout you get when each new token is appended). Reading one head across
# all tokens gives runs of HD lines separated by a jump of one token row.
# --------------------------------------------------------------------------
def llm_trace(d_model=1024, n_heads=8, ctx=96):
    t = Trace()
    hd = d_model // n_heads                   # 128, as in most current LLMs
    head_lines = lines(hd * EL)                 # 4 lines per head per token
    tok_lines = n_heads * head_lines            # 32 lines per token row
    x_lines = lines(d_model * EL)                # 32 lines

    W_QKV, W_O, X, K, V = 0x0010_0000, 0x0020_0000, 0x0030_0000, 0x0040_0000, 0x0050_0000

    def gemv(w_base, rows_out):
        # y = W x, W stored row-major [out][in]; each output row is x_lines long.
        # x is re-read for every row (it lives in cache after the first pass).
        for r in range(rows_out):
            for k in range(x_lines):
                t.rd(1, X + k * LINE)
                t.rd(0, w_base + (r * x_lines + k) * LINE)

    gemv(W_QKV, 3 * d_model // 64)              # scaled-down QKV projection
    for h in range(n_heads):
        q = X + h * head_lines * LINE
        for kv_base, sid in ((K, 2), (V, 3)):
            for tok in range(ctx):
                t.rd(1, q)                      # query / score reuse
                for i in range(head_lines):
                    t.rd(sid, kv_base + (tok * tok_lines + h * head_lines + i) * LINE)
    gemv(W_O, d_model // 64)
    return t


# --------------------------------------------------------------------------
# CNN: 3x3 conv, NHWC, C=64 fp16 (2 lines/pixel), 32x32 output, padded input 34x34.
# Output-stationary with tiles of 8 output columns: for every output row the
# engine reads a 10-pixel segment from 3 input rows, then writes 8 output pixels.
#   sid 0: input feature map   sid 1: output feature map
# Weights sit in a local weight buffer (weight-stationary), so they are not traced.
# --------------------------------------------------------------------------
def cnn_trace(H=32, W=32, C=64, TX=8):
    t = Trace()
    px_lines = lines(C * EL)
    in_w = W + 2
    IN, OUT = 0x0100_0000, 0x0180_0000
    for xt in range(W // TX):
        for y in range(H):
            for ky in range(3):
                row = y + ky
                t.rd_lines(0, IN, (row * in_w + xt * TX) * px_lines, (TX + 2) * px_lines)
            t.rd_lines(1, OUT, (y * W + xt * TX) * px_lines, TX * px_lines)
    return t


# --------------------------------------------------------------------------
# Recommender (DLRM-style), two batches.
#   sid 0: MLP weights (streamed)   sid 1: embedding gathers (2 lines per row)
#   sid 2: dense features
# Embedding indices follow a Zipf distribution, so a few rows are hot.
# --------------------------------------------------------------------------
def rec_trace(batches=2, batch=64, tables=8, rows=100_000, dim=64, zipf_a=1.1, seed=7):
    rng = np.random.default_rng(seed)
    t = Trace()
    row_lines = lines(dim * EL)
    MLP_B, MLP_T, EMB, DENSE = 0x0200_0000, 0x0210_0000, 0x0300_0000, 0x0280_0000
    table_bytes = rows * row_lines * LINE
    for b in range(batches):
        # bottom MLP: dense input + weights
        for i in range(256):
            if i % 8 == 0:
                t.rd(2, DENSE + (b * 32 + i // 8) * LINE)
            t.rd(0, MLP_B + i * LINE)
        # embedding gathers
        for s in range(batch):
            for tb in range(tables):
                idx = (rng.zipf(zipf_a) - 1) % rows
                base = EMB + tb * table_bytes
                t.rd_lines(1, base, idx * row_lines, row_lines)
        # top MLP
        t.rd_lines(0, MLP_T, 0, 512)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="traces")
    args = ap.parse_args()
    import os
    os.makedirs(args.outdir, exist_ok=True)
    for name, tr in (("llm", llm_trace()), ("cnn", cnn_trace()), ("rec", rec_trace())):
        tr.save(f"{args.outdir}/trace_{name}.hex")
        print(f"trace_{name}.hex: {len(tr.addr)} accesses, streams {sorted(set(tr.sid))}")

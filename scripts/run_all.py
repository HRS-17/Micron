"""Generate traces, run the RTL, check it against the reference model, run the sweeps.

    python3 scripts/run_all.py            # everything
    python3 scripts/run_all.py --quick    # main results + golden check only
"""
import argparse
import csv
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ref_model import load_trace, hits  # noqa: E402

WORKLOADS = ["llm", "cnn", "rec"]
DEFAULT = dict(DRAM_LAT=32, GAP=8, DEGREE=4, ISSUE_GAP=2)
HEADER = ("workload,cfg,dram_lat,gap,degree,issue_gap,accesses,hits,misses,late,hit_rate,"
          "avg_latency,cycles,pf_issued,pf_dropped,pf_useful,pf_useless,pf_accuracy,dram_reads\n")
BUILD = os.path.join(ROOT, "build")
RES = os.path.join(ROOT, "results")


def compile_sim(params):
    tag = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    out = os.path.join(BUILD, f"sim_{tag}")
    if not os.path.exists(out):
        cmd = ["iverilog", "-g2012", "-o", out]
        cmd += [f"-Ptb_mem.{k}={v}" for k, v in params.items()]
        cmd += [os.path.join(ROOT, "tb", "tb_mem.v")]
        cmd += sorted(os.path.join(ROOT, "rtl", f) for f in os.listdir(os.path.join(ROOT, "rtl")))
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    return out


def run(params, workload, csv_path):
    sim = compile_sim(params)
    part = f"{csv_path}.{workload}.{os.getpid()}.{id(params)}.part"
    subprocess.run(["vvp", "-n", sim, f"+trace={ROOT}/traces/trace_{workload}.hex",
                    f"+name={workload}", f"+csv={part}"], check=True, stdout=subprocess.DEVNULL)
    with open(part) as f:
        rows = f.read()
    os.remove(part)
    return rows


def run_set(points, csv_name, jobs):
    csv_path = os.path.join(RES, csv_name)
    for p in points:                       # compile serially
        compile_sim(p)
    tasks = [(p, w) for p in points for w in WORKLOADS]
    with ThreadPoolExecutor(jobs) as ex:
        rows = list(ex.map(lambda pw: run(pw[0], pw[1], csv_path), tasks))
    with open(csv_path, "w") as f:
        f.write(HEADER)
        f.write("".join(rows))
    print(f"  wrote results/{csv_name} ({len(tasks)} runs)")


def golden_check():
    with open(os.path.join(RES, "rtl_results.csv")) as f:
        rtl = {r["workload"]: int(r["hits"]) for r in csv.DictReader(f) if r["cfg"] == "0"}
    lines, ok = [], True
    for w in WORKLOADS:
        _, l = load_trace(os.path.join(ROOT, "traces", f"trace_{w}.hex"))
        ref = hits(l)
        good = ref == rtl[w]
        ok &= good
        lines.append(f"{w}: reference model {ref} hits, RTL {rtl[w]} hits -> {'MATCH' if good else 'MISMATCH'}")
    with open(os.path.join(RES, "golden_check.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  " + "\n  ".join(lines))
    if not ok:
        sys.exit("golden check failed")


def level2_sweep():
    rows = ["workload,size_kb,ways,policy,hit_rate\n"]
    for w in WORKLOADS:
        _, l = load_trace(os.path.join(ROOT, "traces", f"trace_{w}.hex"))
        for kb, ways, pol in itertools.product([4, 8, 16, 32, 64], [1, 2, 4, 8, 16],
                                               ["lru", "plru", "random"]):
            sets = kb * 1024 // 64 // ways
            if ways == 1 and pol != "lru":
                continue
            rows.append(f"{w},{kb},{ways},{pol},{100.0 * hits(l, sets, ways, pol) / len(l):.3f}\n")
    with open(os.path.join(RES, "level2_sweep.csv"), "w") as f:
        f.writelines(rows)
    print("  wrote results/level2_sweep.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    os.makedirs(BUILD, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    print("traces")
    subprocess.run([sys.executable, os.path.join(ROOT, "gen_traces.py"),
                    "--outdir", os.path.join(ROOT, "traces")], check=True)
    print("main run")
    run_set([DEFAULT], "rtl_results.csv", args.jobs)
    print("golden check")
    golden_check()
    if args.quick:
        sys.exit(0)

    print("sweeps")
    run_set([{**DEFAULT, "DRAM_LAT": v} for v in (16, 32, 64, 128)], "sweep_latency.csv", args.jobs)
    run_set([{**DEFAULT, "ISSUE_GAP": v} for v in (1, 2, 4, 8)], "sweep_bandwidth.csv", args.jobs)
    run_set([{**DEFAULT, "DEGREE": v} for v in (1, 2, 4, 8)], "sweep_degree.csv", args.jobs)
    run_set([{**DEFAULT, "GAP": v} for v in (2, 8, 16)], "sweep_gap.csv", args.jobs)
    print("level 2")
    level2_sweep()

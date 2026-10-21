"""
Scaling curve — what happens to recall, latency and memory as N grows 100x.
==========================================================================
Reads the per-N result files written by vdb_build_100m.py and turns them into
one table, one chart and one projection.

The point it makes: at a FIXED nlist, every 10x in N multiplies the vectors per
inverted list by 10, so a query at the same nprobe scans 10x more candidates.
Latency therefore grows with N, and the fix at a billion is to grow nlist with
sqrt(N) (or to shard), not to scan harder. Memory, meanwhile, is exactly linear
and that is the part you can read straight off the chart.

Run:  .venv/bin/python vdb_scaling_curve.py
Out:  results_scaling.json, results_scaling.png, and a markdown table on stdout
      ready to paste into README.md.
"""

import json
import os

import numpy as np

RUNS = [
    ("results_100m_curve_1M.json", 1_000_000),
    ("results_100m_curve_10M.json", 10_000_000),
    ("results_100m_curve_100M.json", 100_000_000),
]
PIVOT_NPROBE = 64          # the operating point we compare across N
CEILING_NPROBE = 128       # the most lists we scan -- recall here is the PQ ceiling
GB = 1024 ** 3

# NOTE ON WHAT THIS MEASURES. vdb_build_100m.py never keeps the raw vectors --
# it streams them, encodes them to 16 bytes, and throws them away. So there is
# nothing to re-rank against, and every recall number here is the COMPRESSED
# index alone. That is why recall plateaus near 0.6 instead of the 0.959 in the
# README: 0.959 is IVF-PQ plus exact re-ranking of the top 100, measured where
# the raw vectors were on disk. Both are true, they answer different questions.


def load():
    """Return the runs that actually exist on disk, smallest N first."""
    out = []
    for path, n_expected in RUNS:
        if not os.path.exists(path):
            print(f"  (skipping {path} -- not built yet)")
            continue
        d = json.load(open(path))
        d["_path"] = path
        d["_n_expected"] = n_expected
        out.append(d)
    return sorted(out, key=lambda d: d["N"])


def row_at(run, nprobe):
    for r in run["rows"]:
        if r["nprobe"] == nprobe:
            return r
    return None


def ceiling(run):
    """Recall at the widest scan we measured -- the PQ precision ceiling."""
    return row_at(run, CEILING_NPROBE)


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def markdown(runs):
    lines = []
    lines.append(f"**At nprobe {PIVOT_NPROBE}, no re-ranking** "
                 f"(`{runs[0]['index_factory']}`, single thread, "
                 f"{runs[0]['NQ']:,} BigANN queries)\n")
    lines.append("| N | index on disk | raw uint8 | recall@10 | p50 ms | p99 ms | batch QPS |")
    lines.append("|---|---|---|---|---|---|---|")
    for run in runs:
        r = row_at(run, PIVOT_NPROBE)
        if r is None:
            continue
        raw = run["N"] * run["DIM"]          # BigANN is uint8, 128 bytes/vector
        lines.append(
            f"| {run['N'] // 1_000_000}M | {human(run['index_bytes'])} | {human(raw)} | "
            f"{r['recall']:.3f} | {r['p50_ms']:.2f} | {r['p99_ms']:.2f} | {r['qps_batch']:,} |"
        )

    lines.append(f"\n**Ceiling: widest scan we measured (nprobe {CEILING_NPROBE})**\n")
    lines.append("| N | recall@10 | p99 ms | cost of the last 10x |")
    lines.append("|---|---|---|---|")
    prev = None
    for run in runs:
        r = ceiling(run)
        grow = "—" if prev is None else f"{r['p99_ms'] / prev:.1f}x slower"
        lines.append(f"| {run['N'] // 1_000_000}M | {r['recall']:.3f} | "
                     f"{r['p99_ms']:.2f} | {grow} |")
        prev = r["p99_ms"]
    return "\n".join(lines)


def project(runs):
    """Fit index bytes and p99 against N, then read off the 1B estimate."""
    ns = np.array([r["N"] for r in runs], dtype=float)
    out = {}

    # memory: index_bytes = per_vector * N + fixed. The fixed part is the coarse
    # quantizer (nlist * DIM * 4 bytes) and it dominates at 1M, so taking a simple
    # ratio there overstates the cost at a billion. Fit both terms instead.
    ib = np.array([r["index_bytes"] for r in runs], dtype=float)
    if len(runs) >= 2:
        per_vec, fixed = np.polyfit(ns, ib, 1)
    else:
        per_vec, fixed = ib[0] / ns[0], 0.0
    out["index_bytes_per_vector"] = round(float(per_vec), 2)
    out["index_fixed_overhead_mb"] = round(float(fixed) / 2 ** 20, 1)
    out["projected_1B_index_gb"] = round(float((per_vec * 1e9 + fixed) / GB), 1)
    out["raw_uint8_1B_gb"] = round(1e9 * 128 / GB, 1)

    # latency: fit log(p50) against log(N). p50, not p99 -- the tail is noise at
    # this scale. Slope 1 means linear in N, slope 0 means flat. Expect something
    # in between while the fixed coarse-quantizer cost still matters, trending
    # toward 1 as the list scan takes over.
    if len(runs) >= 2:
        p50 = np.array([row_at(r, PIVOT_NPROBE)["p50_ms"] for r in runs])
        out["p50_growth_per_10x"] = [
            f"{int(ns[i-1]) // 10**6}M->{int(ns[i]) // 10**6}M: {p50[i] / p50[i-1]:.2f}x"
            for i in range(1, len(runs))
        ]
        slope, intercept = np.polyfit(np.log10(ns), np.log10(p50), 1)
        out["p50_loglog_slope"] = round(float(slope), 3)
        out["projected_1B_p50_ms_fixed_nlist"] = round(float(10 ** (intercept + slope * 9)), 1)
        out["note"] = (
            "The 1B latency projection holds nlist fixed at the benchmarked value, so it is "
            "a worst case: it assumes lists keep growing. Growing nlist with sqrt(N), or "
            "sharding 4 ways as vdb_day7_router.py already does, keeps the scanned candidate "
            "count flat instead."
        )
    return out


def chart(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # p50, not p99: at sub-millisecond scale the tail is noise and the curve
    # doubles back on itself. Sorted so the line traces the tradeoff frontier.
    for run in runs:
        pts = sorted(run["rows"], key=lambda r: r["p50_ms"])
        ax1.plot([r["p50_ms"] for r in pts], [r["recall"] for r in pts], "o-",
                 label=f"{run['N'] // 1_000_000}M")
    ax1.set_xscale("log")
    ax1.set_xlabel("p50 latency, ms (single thread)")
    ax1.set_ylabel("recall@10")
    ax1.set_title(f"Recall/latency tradeoff, nprobe 1 to {CEILING_NPROBE}")
    ax1.legend(title="vectors")
    ax1.grid(alpha=0.3)

    # latency and memory live on different scales, so give memory its own axis.
    ns = [r["N"] for r in runs]
    ax2.plot(ns, [row_at(r, PIVOT_NPROBE)["p50_ms"] for r in runs], "o-",
             color="tab:blue", label=f"p50 ms @ nprobe {PIVOT_NPROBE}")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("vectors")
    ax2.set_ylabel("latency, ms", color="tab:blue")
    ax2.grid(alpha=0.3, which="both")

    ax3 = ax2.twinx()
    ax3.plot(ns, [r["index_bytes"] / 2 ** 20 for r in runs], "s--",
             color="tab:orange", label="index size, MB")
    ax3.set_yscale("log")
    ax3.set_ylabel("index size, MB", color="tab:orange")

    ax2.set_title("Cost vs N (log-log)")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    fig.savefig("results_scaling.png", dpi=130)
    print("saved results_scaling.png")


def main():
    runs = load()
    if not runs:
        raise SystemExit("no result files yet -- run vdb_build_100m.py first")

    table = markdown(runs)
    proj = project(runs)

    json.dump({"hardware": runs[0]["hardware"],
               "index_factory": runs[0]["index_factory"],
               "pivot_nprobe": PIVOT_NPROBE,
               "runs": [{"N": r["N"], "index_bytes": r["index_bytes"], "rows": r["rows"]}
                        for r in runs],
               "projection": proj},
              open("results_scaling.json", "w"), indent=1)
    print("saved results_scaling.json")

    chart(runs)

    print("\n" + "=" * 70)
    print(table)
    print("\nprojection:")
    for k, v in proj.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

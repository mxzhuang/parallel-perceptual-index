#!/usr/bin/env python3
"""Check every parallel configuration against the sequential baseline.

For each configuration, pi_eval is run on the test images and its feature dump is compared
with the dump of the sequential baseline (pi_eval_seq):

  * every image of the baseline must be present, valid and free of NaN;
  * features, forest outputs and scores must agree within the tolerances below, which were
    fixed on the calibration images in tests/images before any module was parallelized;
  * the designed versions (intra, graph) must produce bit-identical dumps for every thread
    count, schedule and design switch, because their reductions use a fixed chunk grid.

Usage: check_parallel.py [--bin ./pi_eval] [--seq ./pi_eval_seq] [--threads 1,2,4,N]
                         [--quick] [images ...]
Exit status 0 if every configuration passes.
"""
import argparse
import glob
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import load  # noqa: E402

import numpy as np  # noqa: E402

# Tolerances against the sequential baseline. Reordered reductions (M3 QR, M5 second moments,
# M6 means) change rounding at the 1e-13 level; the bounds leave a wide margin and are still
# far below the 1e-4 agreement with the official code.
TOL_FEATURE_REL = 1e-9
TOL_FOREST = 1e-9
TOL_NIQE = 1e-9
TOL_PI = 1e-9


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\nexit {p.returncode}\n{p.stdout[-400:]}\n{p.stderr[-400:]}")


def rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def compare(ref, got):
    """Return (ok, worst feature rel, worst |ds|, worst |dNIQE|, worst |dPI|, message)."""
    worst = [0.0, 0.0, 0.0, 0.0]
    for name, r in ref.items():
        if name not in got:
            return False, *worst, f"missing image {name}"
        c = got[name]
        vals = [c["ma"], c["niqe"], *c["s"], *np.concatenate(c["f"])]
        if any(np.isnan(v) for v in vals):
            return False, *worst, f"NaN in {name}"
        if any(len(c["f"][k]) != len(r["f"][k]) for k in range(3)):
            return False, *worst, f"feature count differs in {name}"
        worst[0] = max(worst[0], max(rel(c["f"][k], r["f"][k]) for k in range(3)))
        worst[1] = max(worst[1], float(np.max(np.abs(c["s"] - r["s"]))))
        worst[2] = max(worst[2], abs(c["niqe"] - r["niqe"]))
        worst[3] = max(worst[3], abs(((10 - c["ma"]) + c["niqe"]) / 2 - ((10 - r["ma"]) + r["niqe"]) / 2))
    ok = worst[0] <= TOL_FEATURE_REL and worst[1] <= TOL_FOREST and worst[2] <= TOL_NIQE and worst[3] <= TOL_PI
    return ok, *worst, "" if ok else "above tolerance"


def configurations(threads, quick):
    """(label, arguments, is a designed version)"""
    out = [("seq (OpenMP build)", ["--mode", "seq"], False)]
    for t in threads:
        out.append((f"naive t={t}", ["--mode", "naive", "--threads", str(t)], False))
        for mode in ("intra", "graph"):
            out.append((f"{mode} t={t}", ["--mode", mode, "--threads", str(t)], True))
    t = str(max(threads))
    switches = [
        ["--m1", "plain"],
        ["--m2-sched", "dynamic,8"], ["--m2-sched", "static,1"], ["--m2-sched", "guided"],
        ["--m7-sched", "dynamic,1"],
        ["--m4-level", "bands"],
        ["--m5-gather", "materialized"],
        ["--m6-ssim", "tasks"], ["--m6-ssim", "inner"],
        ["--rf-order", "tree"],
        ["--image-workers", "2"], ["--image-workers", t, "--prefetch"],
    ]
    if not quick:
        for sw in switches:
            for mode in ("intra", "graph"):
                out.append((f"{mode} t={t} {' '.join(sw)}", ["--mode", mode, "--threads", t, *sw], True))
        out.append((f"graph t={t} --light serial", ["--mode", "graph", "--threads", t, "--light", "serial"], True))
    # the OpenMP reduction is allowed to depend on the thread count: tolerance only
    out.append((f"intra t={t} --rf-sum reduction", ["--mode", "intra", "--threads", t, "--rf-sum", "reduction"], False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./pi_eval")
    ap.add_argument("--seq", default="./pi_eval_seq")
    ap.add_argument("--threads", default=None, help="thread counts to test (default: 1, 2, 4 and all CPUs)")
    ap.add_argument("--quick", action="store_true", help="modes and thread counts only, no design switches")
    ap.add_argument("--pirm", action="store_true", help="pass --pirm (for RGB inputs)")
    ap.add_argument("images", nargs="*")
    a = ap.parse_args()
    images = a.images or sorted(glob.glob("tests/images/*_y.png"))
    if a.threads:
        threads = [int(t) for t in a.threads.split(",")]
    else:
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        threads = sorted({1, 2, 4, cpus})
    extra = ["--pirm"] if a.pirm else []
    os.makedirs("tests/out", exist_ok=True)

    ref_path = "tests/out/seq.txt"
    run([a.seq, "--mode", "seq", "--quiet", "--dump", ref_path, *extra, *images])
    ref = load(ref_path)
    if len(ref) != len(images):
        print(f"baseline scored {len(ref)} of {len(images)} images")
        return 1

    failures = 0
    designed_hash = None
    print(f"{'configuration':44s} {'feat rel':>9s} {'|ds|':>9s} {'|dNIQE|':>9s} {'|dPI|':>9s}  result")
    for i, (label, args, designed) in enumerate(configurations(threads, a.quick)):
        path = f"tests/out/cfg{i}.txt"
        try:
            run([a.bin, *args, "--quiet", "--dump", path, *extra, *images])
            ok, fr, ds, dn, dp, msg = compare(ref, load(path))
        except RuntimeError as e:
            ok, fr, ds, dn, dp, msg = False, 0, 0, 0, 0, str(e).splitlines()[-1]
        if ok and designed:
            digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if designed_hash is None:
                designed_hash = digest
            elif digest != designed_hash:
                ok, msg = False, "not bit-identical to the other designed configurations"
        failures += not ok
        print(f"{label:44s} {fr:9.1e} {ds:9.1e} {dn:9.1e} {dp:9.1e}  {'ok' if ok else 'FAIL: ' + msg}")
    print(f"{failures} failure(s)" if failures else "all parallel configurations match the sequential baseline")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

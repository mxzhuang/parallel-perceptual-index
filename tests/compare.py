#!/usr/bin/env python3
"""Compare a C++ dump (pi_eval --dump) with the official reference (reference/run_reference.m).

Reports, per image, the score differences and the largest relative difference inside each
feature group (f1 = M2 block-DCT, f2 = M4-M6 subband statistics, f3 = M3 patch SVD),
and the difference of each random-forest output (M8).
"""
import sys

import numpy as np


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t or not line.endswith("\n"):
                continue  # skip an incomplete last line (reference still being written)
            name = t[0]
            vals = [float(x) for x in t[1:6]]
            pos = 6
            groups = []
            try:
                for _ in range(3):
                    n = int(t[pos]); pos += 1
                    groups.append(np.array([float(x) for x in t[pos:pos + n]])); pos += n
            except (IndexError, ValueError):
                continue
            if any(len(g) == 0 for g in groups) or pos != len(t):
                continue
            rows[name] = dict(ma=vals[0], niqe=vals[1], s=np.array(vals[2:5]), f=groups)
    return rows


def rel(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))


def main(ref_path, cpp_path):
    ref, cpp = load(ref_path), load(cpp_path)
    names = [n for n in ref if n in cpp]
    print(f"{'image':28s} {'|dMa|':>10s} {'|dNIQE|':>10s} {'|dPI|':>10s} {'f1 rel':>9s} {'f2 rel':>9s} {'f3 rel':>9s} {'|ds| max':>9s}")
    worst = 0.0
    for n in names:
        r, c = ref[n], cpp[n]
        dma, dnq = abs(c["ma"] - r["ma"]), abs(c["niqe"] - r["niqe"])
        dpi = abs(((10 - c["ma"]) + c["niqe"]) / 2 - ((10 - r["ma"]) + r["niqe"]) / 2)
        worst = max(worst, dpi)
        fr = [rel(c["f"][k], r["f"][k]) for k in range(3)]
        print(f"{n:28s} {dma:10.2e} {dnq:10.2e} {dpi:10.2e} {fr[0]:9.1e} {fr[1]:9.1e} {fr[2]:9.1e} {np.max(np.abs(c['s'] - r['s'])):9.1e}")
    print(f"images compared: {len(names)}, largest |dPI| = {worst:.2e}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

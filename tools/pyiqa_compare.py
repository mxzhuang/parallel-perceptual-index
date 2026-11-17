#!/usr/bin/env python3
"""Time the Perceptual Index of the pyiqa package on the first fixed input (optional, E8).

  python3 tools/pyiqa_compare.py --manifest data/inputs/manifest.json --out results/<run>/pyiqa

Writes <out>/pyiqa.json: one entry per device (cpu, and cuda when available) with the
seconds of every repetition after one warm-up call, and the PI value pyiqa returns.
pyiqa downloads its model weights on first use. Its PI uses its own MATLAB-compatible
resizing and colour conversion, so the value is expected to be close to, not identical
with, the value of pi_eval.
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    import pyiqa
    import torch

    inp = json.load(open(a.manifest))["inputs"][0]
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    runs = []
    for device in devices:
        try:
            metric = pyiqa.create_metric("pi", device=torch.device(device))
            metric(inp["path"])  # warm-up: weights, thread pool, CUDA context
            seconds = []
            value = float("nan")
            for _ in range(a.reps):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                value = float(metric(inp["path"]))
                if device == "cuda":
                    torch.cuda.synchronize()
                seconds.append(time.perf_counter() - t0)
            runs.append({"input": inp["id"], "device": device, "seconds": seconds, "pi": value,
                         "torch_threads": torch.get_num_threads()})
            print(f"pyiqa PI on {device}: {min(seconds):.3f} s (best of {a.reps}), PI {value:.4f}")
        except Exception as e:  # noqa: BLE001 - report and continue with the next device
            print(f"pyiqa on {device} failed: {type(e).__name__}: {e}")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "pyiqa.json"), "w") as f:
        json.dump({"pyiqa": pyiqa.__version__, "torch": torch.__version__, "runs": runs}, f, indent=1)
    return 0 if runs else 1


if __name__ == "__main__":
    raise SystemExit(main())

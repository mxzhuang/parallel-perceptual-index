#!/usr/bin/env python3
"""Run the measurement experiments E1-E7 and store one --report file per configuration.

  python3 tools/run_experiments.py --results results/run-<host> [options]

Every measurement is taken from the machine-readable file that pi_eval writes with
--report; nothing is parsed from the screen. Each run leaves three files in
<results>/raw/<experiment>/:

  <name>.report     written by pi_eval
  <name>.meta.json  what was run: command line, OpenMP environment, experiment labels,
                    exit status, elapsed time
  <name>.log        stdout and stderr of the run

A configuration whose .meta.json reports success is not run again, so an interrupted
session continues where it stopped (use --force to repeat everything).

Repetitions of a single-image configuration are done inside one process: the image is given
R times after --warmup, and the report holds one line (wall time and module times) per
repetition. Throughput runs (E6) are whole-process measurements and are repeated as
separate processes.

Experiments
  E1  cost breakdown     sequential baseline and faithful variant, per-module times
  E2  strong scaling     naive / intra / graph over a thread sweep, both inputs
  E3  ablations          every design switch at the maximum thread count and at half of it
  E4  load balance       per-thread busy time of the M2, M6 and M7 loops
  E5  task trace         start/end/thread of every module task (Gantt chart)
  E6  throughput         image-level, module-level and hybrid parallelism over an image set
  E7  real data          correctness check on dataset images, PI of every set and method
"""
import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
IMAGE_EXT = (".png", ".bmp", ".jpg", ".jpeg")

# Design switches: (group, command-line option, values); the first value is the default.
SWITCHES = [
    ("m2_sched", "--m2-sched", ["static", "dynamic,8", "static,1", "guided"]),
    ("m7_sched", "--m7-sched", ["static", "dynamic,1"]),
    ("m4_level", "--m4-level", ["lines", "bands"]),
    ("m5_gather", "--m5-gather", ["fused", "materialized"]),
    ("m6_ssim", "--m6-ssim", ["strips", "tasks", "inner"]),
    ("rf_sum", "--rf-sum", ["ordered", "reduction"]),
    ("m1", "--m1", ["fused", "plain"]),
]
# Loops instrumented by --balance and the switch that changes their schedule.
BALANCE = [("m2_sched", "--m2-sched", ["static", "dynamic,8", "static,1", "guided"]),
           ("m7_sched", "--m7-sched", ["static", "dynamic,1"]),
           ("m6_ssim", "--m6-ssim", ["strips", "tasks", "inner"])]


def log(msg):
    print(msg, flush=True)


def tag(value):
    return value.replace(",", "-")


def images_in(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMAGE_EXT)) if os.path.isdir(folder) else []


def cpu_topology():
    """(usable logical CPUs, physical cores among them)."""
    logical = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    physical = logical
    try:
        out = subprocess.run(["lscpu", "-p=CPU,CORE,SOCKET"], capture_output=True, text=True, timeout=20).stdout
        usable = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
        cores = set()
        for line in out.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            cpu, core, socket = line.split(",")[:3]
            if usable is None or int(cpu) in usable:
                cores.add((core, socket))
        if cores:
            physical = len(cores)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return logical, min(physical, logical)


def thread_sweep(tmax, physical, quick):
    counts, t = set(), 1
    while t <= tmax:
        counts.add(t)
        t *= 2
    counts.add(tmax)
    if physical < tmax:  # hyper-threading: also measure one thread per physical core
        counts.add(physical)
    counts = sorted(counts)
    if quick and len(counts) > 3:
        counts = [counts[0], counts[len(counts) // 2], counts[-1]]
    return counts


class Runner:
    def __init__(self, a, tmax, physical):
        self.a = a
        self.tmax = tmax
        self.physical = physical
        self.raw = os.path.join(a.results, "raw")
        self.ran = self.reused = self.failed = 0
        self.failures = []

    def env(self, bind):
        env = dict(os.environ)
        for k in ("OMP_NUM_THREADS", "OMP_SCHEDULE", "GOMP_CPU_AFFINITY", "KMP_AFFINITY"):
            env.pop(k, None)
        omp = {}
        if not self.a.no_pin:
            omp = {"OMP_PROC_BIND": bind, "OMP_PLACES": "cores"}
        omp["OMP_DYNAMIC"] = "false"
        env.update(omp)
        return env, omp

    def run(self, exp, name, binary, args, images, labels, bind="close", warmup=True, timeout=None, extra_files=()):
        """Run one configuration; returns True if its report is available."""
        out = os.path.join(self.raw, exp)
        os.makedirs(out, exist_ok=True)
        report = os.path.join(out, name + ".report")
        meta_path = os.path.join(out, name + ".meta.json")
        if not self.a.force and os.path.isfile(meta_path) and os.path.isfile(report):
            try:
                if json.load(open(meta_path)).get("returncode") in (0, 2):
                    self.reused += 1
                    return True
            except ValueError:
                pass
        cmd = [os.path.join(ROOT, binary), "--models", os.path.join(ROOT, "models"), "--pirm", "--quiet",
               "--report", report, *args]
        if warmup:
            cmd.append("--warmup")
        cmd += images
        env, omp = self.env(bind)
        t0 = time.time()
        with open(os.path.join(out, name + ".log"), "w") as lf:
            lf.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
            lf.flush()
            try:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                                    timeout=timeout or self.a.timeout).returncode
            except subprocess.TimeoutExpired:
                rc = -1
                lf.write("\ntimed out\n")
            except OSError as e:
                rc = -2
                lf.write(f"\ncould not start: {e}\n")
        elapsed = time.time() - t0
        meta = {"experiment": exp, "name": name, "binary": binary, "command": cmd, "omp_env": omp,
                "warmup": warmup, "returncode": rc, "elapsed_seconds": round(elapsed, 3),
                "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": [os.path.basename(f) for f in extra_files]}
        meta.update(labels)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
        ok = rc in (0, 2) and os.path.isfile(report)  # 2: some image could not be scored
        if ok:
            self.ran += 1
        else:
            self.failed += 1
            self.failures.append(f"{exp}/{name} (exit {rc})")
        log(f"  {exp} {name:58s} {elapsed:7.1f} s{'' if ok else '   FAILED (exit %d)' % rc}")
        return ok


def load_inputs(a):
    path = os.path.join(a.data, "inputs", "manifest_quick.json" if a.quick else "manifest.json")
    if not os.path.isfile(path):
        sys.exit(f"{path} not found: run tools/fetch_data.sh first")
    inputs = json.load(open(path))["inputs"]
    for e in inputs:
        if not os.path.isfile(e["path"]):
            sys.exit(f"input {e['path']} is missing: run tools/fetch_data.sh again")
    return inputs


def image_sets(a):
    """{(set, method): [paths]} for data/sr/<set>/<method>/ and the originals as method HR."""
    sets = {}
    for folder in sorted(glob.glob(os.path.join(a.data, "sr", "*", "*"))):
        imgs = images_in(folder)
        if imgs:
            sets[(os.path.basename(os.path.dirname(folder)), os.path.basename(folder))] = imgs
    for folder in sorted(glob.glob(os.path.join(a.data, "hr", "*"))):
        imgs = images_in(folder)
        if imgs:
            sets[(os.path.basename(folder), "HR")] = imgs
    return sets


def throughput_set(a, sets, tmax):
    """Image list of E6: DIV2K bicubic outputs when present, else the bundled fallback set."""
    name = "bundled"
    if not a.quick and ("div2k_val", "bicubic_x4") in sets:
        name = "div2k_val"
    imgs = list(sets.get((name, "bicubic_x4"), []))
    if not imgs:
        return name, imgs
    if a.quick:
        return name, imgs[:4]
    want = a.e6_images if a.e6_images > 0 else max(24, 2 * tmax)
    if a.e6_images < 0:
        want = len(imgs)
    if name == "bundled":  # few small images: repeat them so that every worker has work
        imgs = (imgs * (want // len(imgs) + 1))[:want]
    return name, imgs[:want]


def base_labels(inp, mode, threads, **kw):
    d = {"input": inp["id"], "input_width": inp["width"], "input_height": inp["height"], "mode": mode,
         "threads": threads}
    d.update(kw)
    return d


def e1(r, inputs, reps):
    inp = inputs[0]
    r.run("E1", f"{inp['id']}_seq", "pi_eval_seq", ["--mode", "seq"], [inp["path"]] * reps,
          base_labels(inp, "seq", 1, variant="baseline"))
    if r.a.skip_faithful:
        log("  E1 faithful variant skipped (--skip-faithful)")
        return
    # one repetition and no warm-up: the faithful variant takes minutes per image
    r.run("E1", f"{inp['id']}_faithful", "pi_eval_seq", ["--mode", "seq", "--faithful"], [inp["path"]],
          base_labels(inp, "seq", 1, variant="faithful"), warmup=False, timeout=6 * 3600)


def e2(r, inputs, reps, sweep):
    for inp in inputs:
        r.run("E2", f"{inp['id']}_seq", "pi_eval_seq", ["--mode", "seq"], [inp["path"]] * reps,
              base_labels(inp, "seq", 1, baseline=True))
        for mode in ("naive", "intra", "graph"):
            for t in sweep:
                r.run("E2", f"{inp['id']}_{mode}_t{t}", "pi_eval", ["--mode", mode, "--threads", str(t)],
                      [inp["path"]] * reps, base_labels(inp, mode, t))


def e3(r, inputs, reps, tmax):
    inp = inputs[0]
    half = tmax // 2
    for t in [tmax] + ([half] if half >= 2 else []):
        for mode in ("intra", "graph"):
            common = ["--mode", mode, "--threads", str(t)]
            r.run("E3", f"{inp['id']}_{mode}_t{t}_default", "pi_eval", common, [inp["path"]] * reps,
                  base_labels(inp, mode, t, group="default", value="default", is_default=True))
            for group, opt, values in ([] if r.a.quick and mode == "graph" else SWITCHES):
                for v in values[1:]:
                    r.run("E3", f"{inp['id']}_{mode}_t{t}_{group}-{tag(v)}", "pi_eval", common + [opt, v],
                          [inp["path"]] * reps, base_labels(inp, mode, t, group=group, value=v, is_default=False))
        r.run("E3", f"{inp['id']}_graph_t{t}_light-serial", "pi_eval",
              ["--mode", "graph", "--threads", str(t), "--light", "serial"], [inp["path"]] * reps,
              base_labels(inp, "graph", t, group="light", value="serial", is_default=False))


def e4(r, inputs, reps, tmax):
    # no warm-up: --balance accumulates over everything the process evaluates
    inp = inputs[0]
    for group, opt, values in BALANCE:
        for v in values:
            r.run("E4", f"{inp['id']}_intra_t{tmax}_{group}-{tag(v)}", "pi_eval",
                  ["--mode", "intra", "--threads", str(tmax), "--balance", opt, v], [inp["path"]] * reps,
                  base_labels(inp, "intra", tmax, group=group, value=v, is_default=v == values[0]), warmup=False)


def e5(r, inputs, tmax):
    inp = inputs[0]
    for mode in ("intra", "graph"):
        name = f"{inp['id']}_{mode}_t{tmax}"
        trace = os.path.join(r.raw, "E5", name + ".trace")
        os.makedirs(os.path.dirname(trace), exist_ok=True)
        r.run("E5", name, "pi_eval", ["--mode", mode, "--threads", str(tmax), "--trace", trace], [inp["path"]],
              base_labels(inp, mode, tmax), extra_files=[trace])


def e6(r, sets, tmax, reps):
    set_name, imgs = throughput_set(r.a, sets, tmax)
    if not imgs:
        log("  E6 skipped: no image set available (run tools/fetch_data.sh)")
        return
    out = os.path.join(r.raw, "E6")
    os.makedirs(out, exist_ok=True)
    listing = os.path.join(out, "images.txt")
    with open(listing, "w") as f:
        f.write("\n".join(imgs) + "\n")
    hybrid = sorted({tmax // d for d in (2, 4, 8) if 2 <= tmax // d < tmax})
    configs = [("per_image", "intra", tmax)]                        # (a) one image per thread
    configs += [("in_image", "intra", 1), ("in_image", "graph", 1)]  # (b) all threads in one image
    configs += [("hybrid", "intra", k) for k in hybrid]             # (c) K images, tmax / K threads each
    runs = [(s, m, k, p, "image") for (s, m, k) in configs for p in (0, 1)]
    runs += [("per_image", "intra", tmax, 0, "tree"), ("in_image", "intra", 1, 0, "tree")]
    if r.a.quick:  # one run of each kind
        runs = [("per_image", "intra", tmax, 0, "image"), ("per_image", "intra", tmax, 1, "image"),
                ("in_image", "intra", 1, 0, "image"), ("in_image", "graph", 1, 0, "image"),
                ("per_image", "intra", tmax, 0, "tree")]
    log(f"  E6 image set: {set_name}, {len(imgs)} image(s), {len(runs)} configuration(s) x {reps} run(s)")
    for strategy, mode, k, prefetch, order in runs:
        args = ["--mode", mode, "--threads", str(tmax), "--image-workers", str(k), "--rf-order", order, "--list", listing]
        if prefetch:
            args.append("--prefetch")
        for rep in range(reps):
            name = f"{strategy}_{mode}_k{k}_{'prefetch' if prefetch else 'noprefetch'}_rf-{order}_r{rep}"
            # image level spread over the machine, each image's team on neighbouring cores
            r.run("E6", name, "pi_eval", args, [], {"mode": mode, "threads": tmax, "image_workers": k,
                  "strategy": strategy, "prefetch": prefetch, "rf_order": order, "set": set_name,
                  "n_images": len(imgs), "repetition": rep}, bind="spread,close")


def e7(r, sets, tmax):
    out = os.path.join(r.raw, "E7")
    os.makedirs(out, exist_ok=True)
    quick = r.a.quick
    # ---- correctness of every mode on dataset images (not the calibration images) ----
    order = ["Set5", "Set14", "BSD100", "Urban100"]
    names = sorted({s for (s, m) in sets if m == "bicubic_x4" and s != "bundled"},
                   key=lambda s: (order.index(s) if s in order else len(order) + (s == "div2k_val"), s))
    picked = []
    for s in names:
        picked += [(s, p) for p in sets[(s, "bicubic_x4")][:2]]
    picked = picked[:2 if quick else 6] or [("bundled", p) for p in sets.get(("bundled", "bicubic_x4"), [])[:2 if quick else 3]]
    log_path = os.path.join(out, "check_parallel.log")
    meta_path = os.path.join(out, "check_parallel.meta.json")
    done = False
    if not r.a.force and os.path.isfile(meta_path):
        try:
            done = json.load(open(meta_path)).get("returncode") == 0
        except ValueError:
            pass
    if done:
        r.reused += 1
    elif picked:
        # check_parallel.py writes its dumps to ./tests/out and reads ./models: give it a scratch
        # directory. Its dumps are keyed by file name, so the sample gets unique names.
        work = os.path.join(out, "work")
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(os.path.join(work, "sample"))
        sample = []
        for s, p in picked:
            sample.append(os.path.join(work, "sample", f"{s}__{os.path.basename(p)}"))
            shutil.copyfile(p, sample[-1])
        link = os.path.join(work, "models")
        if not os.path.exists(link):
            os.symlink(os.path.join(ROOT, "models"), link)
        threads = ",".join(str(t) for t in sorted({1, min(2, tmax), min(4, tmax)}))
        cmd = [sys.executable, os.path.join(ROOT, "tests", "check_parallel.py"), "--quick", "--pirm",
               "--bin", os.path.join(ROOT, "pi_eval"), "--seq", os.path.join(ROOT, "pi_eval_seq"),
               "--threads", threads, *sample]
        t0 = time.time()
        with open(log_path, "w") as lf:
            lf.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
            lf.flush()
            try:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=work, timeout=r.a.timeout).returncode
            except (subprocess.TimeoutExpired, OSError) as e:
                rc = -1
                lf.write(f"\n{e}\n")
        shutil.rmtree(work, ignore_errors=True)
        json.dump({"experiment": "E7", "name": "check_parallel", "command": cmd, "returncode": rc, "images": [p for _s, p in picked],
                   "elapsed_seconds": round(time.time() - t0, 3)}, open(meta_path, "w"), indent=1)
        log(f"  E7 check_parallel on {len(sample)} dataset image(s): {'ok' if rc == 0 else 'FAILED (see %s)' % log_path}")
        if rc == 0:
            r.ran += 1
        else:
            r.failed += 1
            r.failures.append(f"E7/check_parallel (exit {rc})")
    # ---- PI of every image set and method ----
    for (s, m), imgs in sorted(sets.items()):
        if quick and s != "bundled":
            continue
        if r.a.e7_max_images > 0:
            imgs = imgs[:r.a.e7_max_images]
        listing = os.path.join(out, f"pi_{s}_{m}.txt")
        with open(listing, "w") as f:
            f.write("\n".join(imgs) + "\n")
        k = min(tmax, len(imgs))
        r.run("E7", f"pi_{s}_{m}", "pi_eval", ["--mode", "intra", "--threads", str(tmax), "--image-workers", str(k),
              "--list", listing], [], {"mode": "intra", "threads": tmax, "image_workers": k, "set": s, "method": m,
              "n_images": len(imgs)}, bind="spread,close", warmup=False)


def estimate(r, inputs, reps, e6_reps, sweep, tmax, sets, selected):
    """Rough run-time forecast from one sequential evaluation of the first input."""
    inp = inputs[0]
    scratch = os.path.join(r.raw, "calibration")
    if not r.run("calibration", f"{inp['id']}_seq", "pi_eval_seq", ["--mode", "seq"], [inp["path"]],
                 base_labels(inp, "seq", 1), warmup=False):
        return
    t1 = None
    for line in open(os.path.join(scratch, f"{inp['id']}_seq.report")):
        if line.startswith("image "):
            t1 = float(dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)["wall"])
    if not t1:
        return
    px0 = inp["width"] * inp["height"]
    per_px = t1 / px0

    def par(t_seq, threads):  # assumes 85% of the work scales
        return t_seq * (0.15 + 0.85 / threads)
    est = {}
    n = reps + 1
    est["E1"] = n * t1
    est["E2"] = sum(n * per_px * i["width"] * i["height"] * (1 + sum(3 * (0.15 + 0.85 / t) for t in sweep)) for i in inputs)
    est["E3"] = (2 if tmax >= 4 else 1) * 23 * n * par(t1, tmax)
    est["E4"] = 9 * reps * par(t1, tmax)
    est["E5"] = 4 * par(t1, tmax)
    _name, imgs = throughput_set(r.a, sets, tmax)
    mean_px = px0
    if imgs:
        try:
            from PIL import Image
            with Image.open(imgs[0]) as im:
                mean_px = im.width * im.height
        except Exception:  # noqa: BLE001 - the forecast is optional
            pass
    n_cfg = 8 + 2 * len({tmax // d for d in (2, 4, 8) if 2 <= tmax // d < tmax})
    est["E6"] = n_cfg * e6_reps * (len(imgs) + 1) * par(per_px * mean_px, tmax)
    est["E7"] = 60 + sum(len(v) for (s, m), v in sets.items() if not r.a.quick or s == "bundled") * par(t1 / 4, tmax)
    total = sum(v for k, v in est.items() if k in selected)
    log("Estimated run time (rough; assumes 85% of the work scales with the thread count):")
    for k in ALL:
        if k in selected:
            log(f"  {k}  {est[k] / 60:6.1f} min")
    log(f"  total about {total / 60:.0f} min" + ("" if r.a.skip_faithful or "E1" not in selected
        else ", plus the faithful variant (roughly 100x the sequential baseline, one evaluation)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results directory of this run")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--only", default="", help="comma-separated subset, e.g. E2,E3 (default: all)")
    ap.add_argument("--quick", action="store_true", help="smoke test: bundled images, 1 repetition, few thread counts")
    ap.add_argument("--reps", type=int, default=0, help="repetitions per configuration (default 5; 1 with --quick)")
    ap.add_argument("--e6-reps", type=int, default=0, help="runs per throughput configuration (default 3; 1 with --quick)")
    ap.add_argument("--e6-images", type=int, default=0,
                    help="images in the throughput set (default max(24, 2 x threads); -1 = the whole set)")
    ap.add_argument("--e7-max-images", type=int, default=100, help="images scored per set and method (0 = all)")
    ap.add_argument("--threads-max", type=int, default=0, help="cap on the thread count (default: all usable CPUs)")
    ap.add_argument("--skip-faithful", action="store_true")
    ap.add_argument("--no-pin", action="store_true", help="do not set OMP_PROC_BIND / OMP_PLACES")
    ap.add_argument("--timeout", type=int, default=7200, help="seconds before one run is abandoned")
    ap.add_argument("--force", action="store_true", help="repeat runs that already have a report")
    a = ap.parse_args()

    selected = [e.strip().upper() for e in a.only.split(",") if e.strip()] or ALL
    unknown = [e for e in selected if e not in ALL]
    if unknown:
        sys.exit(f"unknown experiment(s) {', '.join(unknown)}; choose from {', '.join(ALL)}")
    for b in ("pi_eval", "pi_eval_seq"):
        if not os.access(os.path.join(ROOT, b), os.X_OK):
            sys.exit(f"{b} not found: run `make` first")
    a.results = os.path.abspath(a.results)
    a.data = os.path.abspath(a.data)
    if a.quick:
        a.skip_faithful = True
    reps = a.reps or (1 if a.quick else 5)
    e6_reps = a.e6_reps or (1 if a.quick else 3)

    logical, physical = cpu_topology()
    tmax = min(logical, a.threads_max) if a.threads_max > 0 else logical
    physical = min(physical, tmax)
    sweep = thread_sweep(tmax, physical, a.quick)
    inputs = load_inputs(a)
    sets = image_sets(a)
    r = Runner(a, tmax, physical)
    os.makedirs(r.raw, exist_ok=True)
    plan = {"experiments": selected, "quick": a.quick, "repetitions": reps, "e6_repetitions": e6_reps,
            "threads_max": tmax, "logical_cpus": logical, "physical_cores": physical, "thread_sweep": sweep,
            "pinning": None if a.no_pin else {"single image": "OMP_PROC_BIND=close OMP_PLACES=cores",
                                              "image sets (E6, E7)": "OMP_PROC_BIND=spread,close OMP_PLACES=cores"},
            "inputs": inputs, "image_sets": {f"{s}/{m}": len(v) for (s, m), v in sorted(sets.items())}}
    with open(os.path.join(a.results, "experiment_plan.json"), "w") as f:
        json.dump(plan, f, indent=1)
    log(f"Experiments {', '.join(selected)}: threads {sweep} (max {tmax}, {physical} physical core(s)), "
        f"{reps} repetition(s), inputs {', '.join(i['id'] for i in inputs)}")
    estimate(r, inputs, reps, e6_reps, sweep, tmax, sets, selected)

    steps = {"E1": lambda: e1(r, inputs, reps), "E2": lambda: e2(r, inputs, reps, sweep),
             "E3": lambda: e3(r, inputs, reps, tmax), "E4": lambda: e4(r, inputs, reps, tmax),
             "E5": lambda: e5(r, inputs, tmax), "E6": lambda: e6(r, sets, tmax, e6_reps),
             "E7": lambda: e7(r, sets, tmax)}
    status = {}
    for e in selected:
        log(f"== {e}")
        before = r.failed
        t0 = time.time()
        try:
            steps[e]()
            status[e] = "ok" if r.failed == before else f"{r.failed - before} run(s) failed"
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - one broken experiment must not stop the others
            status[e] = f"error: {exc}"
            log(f"  {e} stopped: {exc}")
        status[e] += f" ({(time.time() - t0) / 60:.1f} min)"
    with open(os.path.join(a.results, "experiment_status.json"), "w") as f:
        json.dump({"status": status, "ran": r.ran, "reused": r.reused, "failed": r.failed, "failures": r.failures}, f, indent=1)
    log(f"Runs: {r.ran} new, {r.reused} reused, {r.failed} failed")
    for e in selected:
        log(f"  {e}: {status[e]}")
    return 3 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())

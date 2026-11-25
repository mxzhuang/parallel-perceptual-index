#!/usr/bin/env python3
"""Turn the raw --report files of a results directory into tidy CSV files and summary.txt.

  python3 tools/summarize_results.py results/run-<host>

Reads   <results>/raw/E*/<name>.report + <name>.meta.json, <results>/machine/, run_info.json
Writes  <results>/csv/*.csv (long format, one row per configuration and metric) and
        <results>/summary.txt

Statistics are taken over the repetitions of a configuration: n, median, min, max.
Speedup = median sequential wall time (pi_eval_seq, same input, same machine) divided by the
median wall time of the configuration; efficiency = speedup / threads.

Only the Python standard library is needed.
"""
import csv
import glob
import json
import os
import re
import statistics
import sys

MODULES = [f"m{i}" for i in range(1, 9)]
MODULE_NAMES = {"m1": "spatial pyramid", "m2": "block-DCT statistics", "m3": "patch SVD", "m4": "steerable pyramid",
                "m5": "divisive normalization", "m6": "subband statistics", "m7": "NIQE", "m8": "random forests"}
CONFIG_KEYS = ["mode", "variant", "threads", "image_workers", "threads_per_image", "prefetch", "light", "m1",
               "m2_sched", "m7_sched", "m4_level", "m5_gather", "m6_ssim", "rf_sum", "rf_order"]
COMMON = ["machine", "experiment", "input"] + CONFIG_KEYS
STATS = ["metric", "unit", "n", "median", "min", "max"]


def kv(text):
    return dict(p.split("=", 1) for p in text.split() if "=" in p)


def parse_report(path):
    rep = {"config": {}, "images": [], "total": {}, "load": []}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("config "):
                c = kv(line[7:])
                for k in ("m2_sched", "m7_sched"):  # "static,0" means the default chunk
                    if c.get(k, "").endswith(",0"):
                        c[k] = c[k][:-2]
                rep["config"] = c
            elif line.startswith("image "):
                m = re.match(r"image name=(.*) valid=(\d) (.*)$", line)
                if m:
                    d = {k: float(v) for k, v in kv(m.group(3)).items()}
                    d["name"], d["valid"] = m.group(1), int(m.group(2))
                    rep["images"].append(d)
            elif line.startswith("total "):
                rep["total"] = {k: float(v) for k, v in kv(line[6:]).items()}
            elif line.startswith("load "):
                d = kv(line[5:])
                rep["load"].append({"loop": d["name"], "threads": int(d["threads"]), "max": float(d["max"]),
                                    "mean": float(d["mean"]), "imbalance": float(d["imbalance"]),
                                    "per_thread": [float(x) for x in d["per_thread"].split(",") if x]})
    return rep


def load_runs(results, exp):
    runs = []
    for meta_path in sorted(glob.glob(os.path.join(results, "raw", exp, "*.meta.json"))):
        try:
            meta = json.load(open(meta_path))
        except ValueError:
            continue
        report = meta_path[:-len(".meta.json")] + ".report"
        if meta.get("returncode") not in (0, 2) or not os.path.isfile(report):
            continue
        meta["report"] = parse_report(report)
        meta["_dir"] = os.path.dirname(meta_path)
        runs.append(meta)
    return runs


def stat(values):
    values = [v for v in values if v == v]
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def fmt(x):
    if isinstance(x, float):
        return f"{x:.6g}"
    return "" if x is None else str(x)


def common(machine, exp, run, input_name=None):
    row = {"machine": machine, "experiment": exp, "input": input_name or run.get("input", "")}
    cfg = run["report"]["config"]
    for k in CONFIG_KEYS:
        row[k] = cfg.get(k, "")
    # the run's own label is more specific than the program's (e.g. baseline_slow_core)
    if run.get("variant"):
        row["variant"] = run["variant"]
    return row


def stat_row(base, metric, unit, s, **extra):
    row = dict(base)
    row.update({"metric": metric, "unit": unit, "n": s["n"], "median": s["median"], "min": s["min"], "max": s["max"]})
    row.update(extra)
    return row


def image_stats(run):
    """Statistics over the repetitions (image lines) of a single-image run."""
    imgs = [i for i in run["report"]["images"] if i["valid"]]
    out = {"wall": stat([i["wall"] for i in imgs]), "io": stat([i["io"] for i in imgs])}
    for m in MODULES:
        out[m] = stat([i[m] for i in imgs])
    return out if out["wall"] else None


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([fmt(r.get(c, "")) for c in columns])
    return len(rows)


# ---- machine ---------------------------------------------------------------------------------
def read_text(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def machine_info(results):
    info = {}
    run_info = {}
    try:
        run_info = json.load(open(os.path.join(results, "run_info.json")))
    except (OSError, ValueError):
        pass
    mdir = os.path.join(results, "machine")
    lscpu = {}
    for line in read_text(os.path.join(mdir, "lscpu.txt")).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            lscpu.setdefault(k.strip(), v.strip())
    plan = {}
    try:
        plan = json.load(open(os.path.join(results, "experiment_plan.json")))
    except (OSError, ValueError):
        pass
    mem = re.search(r"MemTotal:\s+(\d+)", read_text(os.path.join(mdir, "meminfo.txt")))
    hostname = read_text(os.path.join(mdir, "hostname.txt")).strip()
    info["machine"] = run_info.get("machine_id") or hostname or os.path.basename(os.path.normpath(results))
    info["hostname"] = hostname
    info["cpu_model"] = lscpu.get("Model name", "")
    info["logical_cpus"] = plan.get("logical_cpus", lscpu.get("CPU(s)", ""))
    info["physical_cores"] = plan.get("physical_cores", "")
    info["sockets"] = lscpu.get("Socket(s)", "")
    info["threads_per_core"] = lscpu.get("Thread(s) per core", "")
    info["threads_max"] = plan.get("threads_max", "")
    info["mem_gb"] = round(int(mem.group(1)) / 1048576, 1) if mem else ""
    info["virtualization"] = read_text(os.path.join(mdir, "virt.txt")).strip().splitlines()[0] if read_text(os.path.join(mdir, "virt.txt")).strip() else ""
    info["compiler"] = (read_text(os.path.join(mdir, "gxx.txt")).splitlines() or [""])[0]
    info["kernel"] = read_text(os.path.join(mdir, "uname.txt")).strip()
    info["git_commit"] = run_info.get("git_commit", "")
    info["gpu"] = (read_text(os.path.join(mdir, "gpu.txt")).strip().splitlines() or [""])[0]
    info["omp_pinning"] = "none" if plan.get("pinning") is None and plan else "OMP_PROC_BIND=close OMP_PLACES=cores"
    info["quick"] = int(bool(plan.get("quick", run_info.get("quick", False))))
    info["check"] = run_info.get("check", "")
    return info


MACHINE_COLS = ["machine", "hostname", "cpu_model", "logical_cpus", "physical_cores", "sockets", "threads_per_core",
                "threads_max", "mem_gb", "virtualization", "compiler", "kernel", "git_commit", "gpu", "omp_pinning",
                "quick", "check"]


# ---- experiments -----------------------------------------------------------------------------
def summarize(results, out=None):
    results = os.path.abspath(results)
    out = out or os.path.join(results, "csv")
    os.makedirs(out, exist_ok=True)
    info = machine_info(results)
    machine = info["machine"]
    written = {}
    text = []

    def emit(name, columns, rows):
        if rows:
            written[name] = write_csv(os.path.join(out, name), columns, rows)

    emit("machine.csv", MACHINE_COLS, [info])
    text.append(f"Machine {machine}: {info['cpu_model']}, {info['logical_cpus']} logical CPU(s), "
                f"{info['physical_cores']} physical core(s), {info['mem_gb']} GB, virtualization: {info['virtualization'] or 'unknown'}")
    text.append(f"Compiler: {info['compiler']}; commit {info['git_commit']}; pinning: {info['omp_pinning']}")
    if info["quick"]:
        text.append("QUICK RUN: smoke-test settings (bundled images, few repetitions); the numbers are not measurements to report.")
    if info["check"] and info["check"] != "passed":
        text.append(f"*** accuracy check: {info['check'].upper()} -- see logs/03_check.log ***")
    elif info["check"]:
        text.append("accuracy check: passed")

    # E1 ---------------------------------------------------------------------------------------
    rows = []
    for run in load_runs(results, "E1"):
        st = image_stats(run)
        if not st:
            continue
        base = common(machine, "E1", run)
        total = sum(st[m]["median"] for m in MODULES)
        rows.append(stat_row(base, "wall", "s", st["wall"]))
        rows.append(stat_row(base, "io", "s", st["io"]))
        for m in MODULES:
            rows.append(stat_row(base, m, "s", st[m], share=st[m]["median"] / total if total else ""))
        rows.append(stat_row(base, "modules_sum", "s", {"n": st["wall"]["n"], "median": total, "min": total, "max": total}))
        text.append(f"\nE1 cost breakdown, {base['variant']} sequential, input {base['input']} "
                    f"(median of {st['wall']['n']}): wall {st['wall']['median']:.3f} s")
        for m in MODULES:
            text.append(f"  {m.upper()} {MODULE_NAMES[m]:24s} {st[m]['median']:9.3f} s  {100 * st[m]['median'] / total if total else 0:5.1f}%")
    emit("E1_cost_breakdown.csv", COMMON + STATS + ["share"], rows)

    # E2 ---------------------------------------------------------------------------------------
    rows = []
    runs = load_runs(results, "E2")
    base_stats = {run["input"]: image_stats(run) for run in runs if run.get("baseline")}
    table = {}
    for run in runs:
        st = image_stats(run)
        if not st:
            continue
        base = common(machine, "E2", run)
        rows.append(stat_row(base, "wall", "s", st["wall"]))
        for m in MODULES:
            rows.append(stat_row(base, m, "s", st[m]))
        ref = base_stats.get(run["input"])
        if ref and not run.get("baseline"):
            t = int(base["threads"])
            b = ref["wall"]["median"]
            sp = {"n": st["wall"]["n"], "median": b / st["wall"]["median"], "min": b / st["wall"]["max"], "max": b / st["wall"]["min"]}
            rows.append(stat_row(base, "speedup", "x", sp))
            rows.append(stat_row(base, "efficiency", "fraction", {k: (v if k == "n" else v / t) for k, v in sp.items()}))
            for m in MODULES:
                if st[m]["median"] > 0 and st[m]["min"] > 0 and st[m]["max"] > 0:
                    bm = ref[m]["median"]
                    rows.append(stat_row(base, f"speedup_{m}", "x", {"n": st[m]["n"], "median": bm / st[m]["median"],
                                                                     "min": bm / st[m]["max"], "max": bm / st[m]["min"]}))
            table.setdefault(run["input"], {}).setdefault(base["mode"], {})[t] = (st["wall"]["median"], sp["median"])
    emit("E2_scaling.csv", COMMON + STATS, rows)
    for inp, modes in table.items():
        text.append(f"\nE2 strong scaling, input {inp}: sequential baseline {base_stats[inp]['wall']['median']:.3f} s")
        threads = sorted({t for m in modes.values() for t in m})
        text.append("  threads  " + "".join(f"{t:>9d}" for t in threads))
        for mode in ("naive", "intra", "graph"):
            if mode in modes:
                text.append(f"  {mode:8s} " + "".join(f"{modes[mode][t][1]:8.2f}x" if t in modes[mode] else " " * 9 for t in threads)
                            + "   (speedup)")

    # E3 ---------------------------------------------------------------------------------------
    rows = []
    runs = load_runs(results, "E3")
    stats = [(run, image_stats(run)) for run in runs]
    defaults = {(r["mode"], r["threads"]): s["wall"]["median"] for r, s in stats if s and r.get("is_default")}
    lines = []
    for run, st in stats:
        if not st:
            continue
        base = common(machine, "E3", run)
        extra = {"group": run.get("group", ""), "value": run.get("value", ""), "is_default": int(bool(run.get("is_default")))}
        d = defaults.get((run["mode"], run["threads"]))
        rows.append(stat_row(base, "wall", "s", st["wall"], rel_to_default=st["wall"]["median"] / d if d else "", **extra))
        for m in MODULES:
            rows.append(stat_row(base, m, "s", st[m], **extra))
        rss = run["report"]["total"].get("peak_rss_mb")
        if rss is not None:
            rows.append(stat_row(base, "peak_rss_mb", "MB", {"n": 1, "median": rss, "min": rss, "max": rss}, **extra))
        lines.append((run["threads"], run["mode"], extra["group"], extra["value"], st["wall"]["median"],
                      st["wall"]["median"] / d if d else float("nan"), rss or 0))
    emit("E3_ablations.csv", COMMON + ["group", "value", "is_default"] + STATS + ["rel_to_default"], rows)
    if lines:
        text.append("\nE3 ablations (median wall time; ratio to the default configuration of the same mode and thread count)")
        for t, mode, group, value, wall, rel, rss in sorted(lines, key=lambda x: (-x[0], x[1], x[2] != "default", x[2], x[3])):
            text.append(f"  t={t:<3d} {mode:6s} {group:10s} {value:13s} {wall:9.3f} s  x{rel:5.3f}  peak {rss:7.0f} MB")

    # E4 ---------------------------------------------------------------------------------------
    rows, thread_rows = [], []
    loop_of = {"m2_sched": "m2_windows", "m7_sched": "m7_blocks", "m6_ssim": "m6_ssim"}
    for run in load_runs(results, "E4"):
        base = common(machine, "E4", run)
        for ld in run["report"]["load"]:
            if ld["loop"] != loop_of.get(run.get("group")):
                continue  # only the loop whose schedule this run varies
            extra = {"group": run["group"], "value": run["value"], "loop": ld["loop"]}
            rows.append(dict(base, **extra, loop_threads=ld["threads"], busy_max=ld["max"], busy_mean=ld["mean"],
                             imbalance=ld["imbalance"]))
            for i, v in enumerate(ld["per_thread"]):
                thread_rows.append({"machine": machine, "input": base["input"], "mode": base["mode"],
                                    "threads": base["threads"], **extra, "thread": i, "busy_seconds": v})
    emit("E4_load_balance.csv", COMMON + ["group", "value", "loop", "loop_threads", "busy_max", "busy_mean", "imbalance"], rows)
    emit("E4_load_balance_threads.csv", ["machine", "input", "mode", "threads", "group", "value", "loop", "thread", "busy_seconds"], thread_rows)
    if rows:
        text.append("\nE4 load balance (per-thread busy time, max / mean; 1.0 is perfectly balanced)")
        for r in rows:
            text.append(f"  {r['loop']:11s} {r['value']:10s} threads {r['loop_threads']:<3d} max/mean {r['imbalance']:.3f}")

    # E5 ---------------------------------------------------------------------------------------
    rows = []
    for run in load_runs(results, "E5"):
        for name in run.get("files", []):
            path = os.path.join(run["_dir"], name)
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                next(f, None)
                for line in f:
                    p = line.split()
                    if len(p) == 5:
                        rows.append({"machine": machine, "input": run["input"], "mode": run["mode"], "threads": run["threads"],
                                     "image": p[0], "task": p[1], "start": float(p[2]), "end": float(p[3]), "thread": p[4]})
    emit("E5_trace.csv", ["machine", "input", "mode", "threads", "image", "task", "start", "end", "thread"], rows)
    if rows:
        text.append(f"\nE5 task trace: {len(rows)} task interval(s) recorded")

    # E6 ---------------------------------------------------------------------------------------
    rows = []
    groups = {}
    for run in load_runs(results, "E6"):
        key = (run["strategy"], run["mode"], run["image_workers"], run["prefetch"], run["rf_order"])
        groups.setdefault(key, []).append(run)
    lines = []
    for key, grp in sorted(groups.items()):
        run = grp[0]
        base = common(machine, "E6", run, input_name=run["set"])
        extra = {"strategy": run["strategy"], "set": run["set"], "n_images": run["n_images"]}
        for metric, unit, field in (("throughput", "images/s", "throughput"), ("wall", "s", "wall"),
                                    ("peak_rss_mb", "MB", "peak_rss_mb"), ("io", "s", "io")):
            s = stat([g["report"]["total"].get(field, float("nan")) for g in grp])
            if s:
                rows.append(stat_row(base, metric, unit, s, **extra))
        tp = stat([g["report"]["total"].get("throughput", float("nan")) for g in grp])
        rss = stat([g["report"]["total"].get("peak_rss_mb", float("nan")) for g in grp])
        if tp and rss:
            lines.append(f"  {run['strategy']:10s} {run['mode']:6s} workers {run['image_workers']:<3d} prefetch {run['prefetch']} "
                         f"rf-order {run['rf_order']:5s} {tp['median']:8.3f} images/s  peak {rss['max']:7.0f} MB")
    emit("E6_throughput.csv", COMMON + ["strategy", "set", "n_images"] + STATS, rows)
    if lines:
        r0 = next(iter(groups.values()))[0]
        text.append(f"\nE6 throughput on {r0['set']} ({r0['n_images']} image(s), {r0['threads']} thread(s))")
        text += lines

    # E7 ---------------------------------------------------------------------------------------
    set_rows, img_rows = [], []
    for run in load_runs(results, "E7"):
        if "set" not in run:
            continue
        imgs = run["report"]["images"]
        valid = [i for i in imgs if i["valid"]]
        tot = run["report"]["total"]
        set_rows.append({"machine": machine, "set": run["set"], "method": run["method"], "n_images": len(imgs),
                         "n_valid": len(valid), "pi": tot.get("pi", ""),
                         "mean_ma": statistics.fmean(i["ma"] for i in valid) if valid else "",
                         "mean_niqe": statistics.fmean(i["niqe"] for i in valid) if valid else "",
                         "wall": tot.get("wall", ""), "throughput": tot.get("throughput", "")})
        for i in imgs:
            img_rows.append({"machine": machine, "set": run["set"], "method": run["method"], "image": i["name"],
                             "valid": i["valid"], "ma": i["ma"], "niqe": i["niqe"], "pi": i["pi"]})
    emit("E7_dataset_pi.csv", ["machine", "set", "method", "n_images", "n_valid", "pi", "mean_ma", "mean_niqe", "wall", "throughput"], set_rows)
    emit("E7_image_scores.csv", ["machine", "set", "method", "image", "valid", "ma", "niqe", "pi"], img_rows)
    if set_rows:
        text.append("\nE7 Perceptual Index per image set and method (lower is better)")
        for r in set_rows:
            text.append(f"  {r['set']:16s} {r['method']:14s} {r['n_valid']:4d} image(s)  PI {r['pi']:.4f}  Ma {r['mean_ma']:.4f}  NIQE {r['mean_niqe']:.4f}")
    rows = []
    check_log = os.path.join(results, "raw", "E7", "check_parallel.log")
    verdict = ""
    for line in read_text(check_log).splitlines():
        m = re.match(r"(.{44}) +(\S+) +(\S+) +(\S+) +(\S+)  (ok|FAIL.*)$", line)
        if m and m.group(1).strip() != "configuration":
            rows.append({"machine": machine, "configuration": m.group(1).strip(), "feature_rel": m.group(2),
                         "forest_abs": m.group(3), "niqe_abs": m.group(4), "pi_abs": m.group(5), "result": m.group(6)})
        elif "failure(s)" in line or line.startswith("all parallel"):
            verdict = line
    emit("E7_correctness.csv", ["machine", "configuration", "feature_rel", "forest_abs", "niqe_abs", "pi_abs", "result"], rows)
    if rows:
        text.append(f"E7 correctness on dataset images: {len(rows)} configuration(s), {sum(r['result'] != 'ok' for r in rows)} failure(s). {verdict}")

    # E8: PI of pyiqa on the same input (optional) -----------------------------------------------
    rows = []
    try:
        for r in json.load(open(os.path.join(results, "pyiqa", "pyiqa.json")))["runs"]:
            s = stat(r["seconds"])
            if s:
                rows.append({"machine": machine, "input": r["input"], "device": r["device"], "pi": r["pi"], "unit": "s",
                             "metric": "wall", **s})
    except (OSError, ValueError, KeyError):
        pass
    emit("E8_pyiqa.csv", ["machine", "input", "device", "pi"] + STATS, rows)
    for r in rows:
        text.append(f"\nE8 pyiqa PI on {r['input']} ({r['device']}): {r['median']:.3f} s per image, PI {r['pi']:.4f}")

    text.append("\nCSV files: " + ", ".join(f"{k} ({v} rows)" for k, v in written.items()))
    with open(os.path.join(results, "summary.txt"), "w") as f:
        f.write("\n".join(text) + "\n")
    return written


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if len(sys.argv) >= 2 else 1
    for results in sys.argv[1:]:
        if not os.path.isdir(os.path.join(results, "raw")):
            print(f"{results}: no raw/ directory, nothing to summarize", file=sys.stderr)
            return 1
        written = summarize(results)
        print(f"{results}: wrote {len(written)} CSV file(s) to csv/ and summary.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

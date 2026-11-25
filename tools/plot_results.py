#!/usr/bin/env python3
"""Figures (vector PDF) and LaTeX tables (booktabs) from the experiment CSV files.

  python3 tools/plot_results.py RESULTS [RESULTS ...] [--out DIR]

RESULTS is a results directory (results/run-<host>) or an archive produced by run_all.sh
(pi-results-<host>-<date>.tar.gz); archives are unpacked to a temporary directory. If a
directory has raw reports but no csv/ folder, tools/summarize_results.py is run on it first.

With one RESULTS argument the output goes to <RESULTS>/figures and <RESULTS>/tables (for an
archive: next to the archive, in <archive name>/). With several, the machines are merged:
scaling and throughput figures show one series per machine, the remaining figures are
written once per machine, and --out (default results/combined) receives everything.

Figures are sized for one column of a two-column paper (3.4 in wide, 8 pt text).
Requires matplotlib; no other third-party package.
"""
import argparse
import csv
import os
import sys
import tarfile
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize_results  # noqa: E402

COL_W = 3.4
MODULES = [f"m{i}" for i in range(1, 9)]
MODE_ORDER = ["naive", "intra", "graph"]
MODE_STYLE = {"naive": ("#7f7f7f", "s"), "intra": ("#1f77b4", "o"), "graph": ("#d62728", "^")}
MODULE_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3", "#8c8c8c"]
MACHINE_LINES = ["-", "--", ":", "-."]
GREY = "#555555"


def style():
    plt.rcParams.update({
        "font.size": 8, "font.family": "serif", "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "lines.linewidth": 1.1, "lines.markersize": 3.5, "grid.linewidth": 0.4, "grid.color": "#dddddd",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "hatch.linewidth": 0.5,
    })


def tex(s):
    out = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        out = out.replace(a, b)
    return out


class Data:
    """All CSV rows of the given results directories, with numbers converted."""

    NUMERIC = {"threads", "image_workers", "threads_per_image", "prefetch", "n", "median", "min", "max", "share",
               "is_default", "rel_to_default", "loop_threads", "busy_max", "busy_mean", "imbalance", "thread",
               "busy_seconds", "start", "end", "n_images", "n_valid", "pi", "mean_ma", "mean_niqe", "wall",
               "throughput", "logical_cpus", "physical_cores", "threads_max", "mem_gb", "quick"}

    def __init__(self, dirs):
        self.tables = {}
        for d in dirs:
            for name in sorted(os.listdir(os.path.join(d, "csv"))):
                if not name.endswith(".csv"):
                    continue
                with open(os.path.join(d, "csv", name), newline="") as f:
                    for row in csv.DictReader(f):
                        for k in row:
                            if k in self.NUMERIC and row[k] != "":
                                try:
                                    row[k] = float(row[k])
                                except ValueError:
                                    pass
                        self.tables.setdefault(name[:-4], []).append(row)
        self.machines = []
        for r in self.tables.get("machine", []):
            if r["machine"] not in self.machines:
                self.machines.append(r["machine"])

    def rows(self, table, **where):
        return [r for r in self.tables.get(table, []) if all(r.get(k) == v for k, v in where.items())]

    def values(self, table, column, **where):
        seen = []
        for r in self.rows(table, **where):
            if r[column] not in seen:
                seen.append(r[column])
        return seen


class Output:
    def __init__(self, out):
        self.fig_dir = os.path.join(out, "figures")
        self.tab_dir = os.path.join(out, "tables")
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.tab_dir, exist_ok=True)
        self.made = []

    def fig(self, fig, name):
        path = os.path.join(self.fig_dir, name + ".pdf")
        fig.savefig(path)
        plt.close(fig)
        self.made.append(path)

    def table(self, name, header, rows, align=None, caption_note=""):
        if not rows:
            return
        align = align or ("l" + "r" * (len(header) - 1))
        lines = []
        if caption_note:
            lines.append("% " + caption_note)
        lines += [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
        for r in rows:
            lines.append(r"\midrule" if r is None else " & ".join(str(c) for c in r) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        path = os.path.join(self.tab_dir, name + ".tex")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        self.made.append(path)


def suffix(data, machine):
    return f"_{machine}" if len(data.machines) > 1 else ""


def thread_axis(ax, threads):
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(int(t)) for t in threads])
    ax.minorticks_off()
    ax.set_xlabel("Threads")
    ax.grid(True, axis="y")


# ---- E2 --------------------------------------------------------------------------------------
def plot_scaling(data, out):
    for inp in data.values("E2_scaling", "input"):
        for metric, label, name in (("speedup", "Speedup over sequential", "speedup"),
                                    ("efficiency", "Parallel efficiency", "efficiency")):
            fig, ax = plt.subplots(figsize=(COL_W, 2.3))
            all_threads = set()
            for mi, machine in enumerate(data.machines):
                for mode in MODE_ORDER:
                    rows = sorted(data.rows("E2_scaling", machine=machine, input=inp, mode=mode, metric=metric),
                                  key=lambda r: r["threads"])
                    if not rows:
                        continue
                    x = [r["threads"] for r in rows]
                    all_threads.update(x)
                    color, marker = MODE_STYLE[mode]
                    lab = mode if len(data.machines) == 1 else f"{mode}, {machine}"
                    y = [r["median"] for r in rows]
                    ax.errorbar(x, y, yerr=[[max(0.0, v - r["min"]) for v, r in zip(y, rows)],
                                            [max(0.0, r["max"] - v) for v, r in zip(y, rows)]],
                                color=color, marker=marker, linestyle=MACHINE_LINES[mi % 4], label=lab,
                                elinewidth=0.5, capsize=1.2)
            if not all_threads:
                plt.close(fig)
                continue
            threads = sorted(all_threads)
            thread_axis(ax, threads)
            if metric == "speedup":
                ax.plot(threads, threads, color="black", linewidth=0.6, linestyle=(0, (4, 3)), label="ideal")
                ax.set_yscale("log", base=2)
                ax.set_yticks(threads)
                ax.set_yticklabels([str(int(t)) for t in threads])
                ax.minorticks_off()
            else:
                ax.axhline(1.0, color="black", linewidth=0.6, linestyle=(0, (4, 3)))
                top = max([r["max"] for r in data.rows("E2_scaling", input=inp, metric=metric)] + [1.0])
                ax.set_ylim(0, 1.1 * top)
            ax.set_ylabel(label)
            ax.legend(loc="upper left" if metric == "speedup" else "lower left", ncol=1 if len(data.machines) == 1 else 2)
            out.fig(fig, f"fig_{name}_{inp}")


def plot_module_scaling(data, out):
    for machine in data.machines:
        for inp in data.values("E2_scaling", "input", machine=machine):
            for mode in ("intra", "graph"):
                threads = sorted({r["threads"] for r in data.rows("E2_scaling", machine=machine, input=inp, mode=mode, metric="wall")})
                if not threads:
                    continue
                # stacked per-module time, sequential baseline first
                fig, ax = plt.subplots(figsize=(COL_W, 2.2))
                labels = ["seq"] + [str(int(t)) for t in threads]
                bottom = [0.0] * len(labels)
                for k, m in enumerate(MODULES):
                    vals = [r["median"] for r in data.rows("E2_scaling", machine=machine, input=inp, mode="seq", metric=m)][:1]
                    for t in threads:
                        vals += [r["median"] for r in data.rows("E2_scaling", machine=machine, input=inp, mode=mode, metric=m, threads=t)][:1]
                    if len(vals) != len(labels):
                        continue
                    ax.bar(range(len(labels)), vals, bottom=bottom, color=MODULE_COLORS[k], width=0.7, label=m.upper(), linewidth=0)
                    bottom = [b + v for b, v in zip(bottom, vals)]
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_xlabel(f"Threads ({mode})")
                ax.set_ylabel("Time per image [s]")
                ax.grid(True, axis="y")
                ax.set_axisbelow(True)
                ax.legend(ncol=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.6, handlelength=0.8, handletextpad=0.3)
                out.fig(fig, f"fig_module_time_{inp}_{mode}{suffix(data, machine)}")
                # per-module speedup
                fig, ax = plt.subplots(figsize=(COL_W, 2.3))
                for k, m in enumerate(MODULES):
                    rows = sorted(data.rows("E2_scaling", machine=machine, input=inp, mode=mode, metric=f"speedup_{m}"),
                                  key=lambda r: r["threads"])
                    if rows:
                        ax.plot([r["threads"] for r in rows], [r["median"] for r in rows], color=MODULE_COLORS[k],
                                marker="osv^D<>p"[k], label=m.upper())
                thread_axis(ax, threads)
                ax.plot(threads, threads, color="black", linewidth=0.6, linestyle=(0, (4, 3)), label="ideal")
                ax.set_yscale("log", base=2)
                ax.set_yticks(threads)
                ax.set_yticklabels([str(int(t)) for t in threads])
                ax.minorticks_off()
                ax.set_ylabel(f"Module speedup ({mode})")
                ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.6, handletextpad=0.3)
                out.fig(fig, f"fig_module_speedup_{inp}_{mode}{suffix(data, machine)}")


def table_scaling(data, out):
    for machine in data.machines:
        for inp in data.values("E2_scaling", "input", machine=machine):
            base = data.rows("E2_scaling", machine=machine, input=inp, mode="seq", metric="wall")
            rows = []
            for mode in MODE_ORDER:
                walls = sorted(data.rows("E2_scaling", machine=machine, input=inp, mode=mode, metric="wall"), key=lambda r: r["threads"])
                for w in walls:
                    sp = data.rows("E2_scaling", machine=machine, input=inp, mode=mode, metric="speedup", threads=w["threads"])
                    ef = data.rows("E2_scaling", machine=machine, input=inp, mode=mode, metric="efficiency", threads=w["threads"])
                    rows.append([mode, int(w["threads"]), f"{w['median']:.3f}", f"{sp[0]['median']:.2f}" if sp else "--",
                                 f"{100 * ef[0]['median']:.0f}\\%" if ef else "--"])
                if walls:
                    rows.append(None)
            if rows and rows[-1] is None:
                rows.pop()
            note = f"machine {machine}, input {inp}" + (f", sequential baseline {base[0]['median']:.3f} s" if base else "")
            out.table(f"tab_scaling_{inp}{suffix(data, machine)}", ["Version", "Threads", "Time [s]", "Speedup", "Efficiency"],
                      rows, "lrrrr", note)


# ---- E1 --------------------------------------------------------------------------------------
def plot_cost(data, out):
    for machine in data.machines:
        variants = data.values("E1_cost_breakdown", "variant", machine=machine)
        if not variants:
            continue
        fig, ax = plt.subplots(figsize=(COL_W, 0.55 + 0.38 * len(variants)))
        rows_tex = []
        for vi, variant in enumerate(variants):
            left = 0.0
            wall = data.rows("E1_cost_breakdown", machine=machine, variant=variant, metric="wall")
            cells = []
            for k, m in enumerate(MODULES):
                r = data.rows("E1_cost_breakdown", machine=machine, variant=variant, metric=m)
                if not r:
                    continue
                share = 100 * (r[0]["share"] or 0)
                ax.barh(vi, share, left=left, color=MODULE_COLORS[k], height=0.6, linewidth=0, label=m.upper() if vi == 0 else None)
                if share >= 7:
                    ax.text(left + share / 2, vi, f"{share:.0f}", ha="center", va="center", fontsize=6, color="white")
                left += share
                cells.append((r[0]["median"], share))
            name = {"baseline": "optimized baseline", "faithful": "faithful port",
                    "baseline_slow_core": "baseline on a slow core"}.get(variant, variant)
            rows_tex.append((name, wall[0]["median"] if wall else float("nan"), cells))
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(["baseline" if v == "baseline" else v for v in variants])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of sequential module time [%]")
        ax.legend(ncol=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.6, handlelength=0.8, handletextpad=0.3)
        out.fig(fig, f"fig_cost_breakdown{suffix(data, machine)}")
        header = ["Module"] + [f"{tex(n)} [s]" for n, _w, _c in rows_tex for _ in (0,)]
        header = ["Module"]
        for n, _w, _c in rows_tex:
            header += [f"{tex(n)} [s]", r"\%"]
        body = []
        for k, m in enumerate(MODULES):
            line = [f"{m.upper()} {summarize_results.MODULE_NAMES[m]}"]
            for _n, _w, cells in rows_tex:
                line += [f"{cells[k][0]:.3f}", f"{cells[k][1]:.1f}"] if k < len(cells) else ["--", "--"]
            body.append(line)
        body.append(None)
        total = ["Wall time"]
        for _n, w, _c in rows_tex:
            total += [f"{w:.3f}", ""]
        body.append(total)
        inp = data.values("E1_cost_breakdown", "input", machine=machine)
        out.table(f"tab_cost_breakdown{suffix(data, machine)}", header, body, "l" + "rr" * len(rows_tex),
                  f"machine {machine}, input {inp[0] if inp else ''}, one thread")


# ---- E3 --------------------------------------------------------------------------------------
GROUP_LABEL = {"m2_sched": "M2 schedule", "m7_sched": "M7 schedule", "m4_level": "M4 level", "m5_gather": "M5 gather",
               "m6_ssim": "M6 SSIM", "rf_sum": "M8 sum", "m1": "M1 filter", "light": "light modules"}
DEFAULT_VALUE = {"m2_sched": "static", "m7_sched": "static", "m4_level": "lines", "m5_gather": "fused",
                 "m6_ssim": "strips", "rf_sum": "ordered", "m1": "fused", "light": "parallel"}


def plot_ablation(data, out):
    for machine in data.machines:
        walls = data.rows("E3_ablations", machine=machine, metric="wall")
        if not walls:
            continue
        tex_rows = []
        for t in sorted({r["threads"] for r in walls}, reverse=True):
            for mode in ("intra", "graph"):
                sel = [r for r in walls if r["threads"] == t and r["mode"] == mode]
                default = [r for r in sel if r["is_default"] == 1]
                if not sel or not default:
                    continue
                d = default[0]
                labels, vals, lo, hi, colors = [], [], [], [], []
                for group in GROUP_LABEL:
                    members = [r for r in sel if r["group"] == group]
                    if not members:
                        continue
                    entries = [(DEFAULT_VALUE[group], d)] + [(r["value"], r) for r in members]
                    for value, r in entries:
                        labels.append(f"{GROUP_LABEL[group]}: {value}")
                        vals.append(r["median"] / d["median"])
                        lo.append(max(0.0, (r["median"] - r["min"]) / d["median"]))
                        hi.append(max(0.0, (r["max"] - r["median"]) / d["median"]))
                        colors.append("#bbbbbb" if r is d else MODE_STYLE[mode][0])
                        rss = [x for x in data.rows("E3_ablations", machine=machine, metric="peak_rss_mb", mode=mode, threads=t)
                               if (x["is_default"] == 1 if r is d else (x["group"] == group and x["value"] == r["value"]))]
                        tex_rows.append([mode, int(t), GROUP_LABEL[group], tex(value) + (" (default)" if r is d else ""),
                                         f"{r['median']:.3f}", f"{r['median'] / d['median']:.2f}",
                                         f"{rss[0]['median']:.0f}" if rss else "--"])
                fig, ax = plt.subplots(figsize=(COL_W, 0.45 + 0.125 * len(labels)))
                y = range(len(labels))
                ax.barh(y, vals, xerr=[lo, hi], color=colors, height=0.72, linewidth=0, error_kw={"elinewidth": 0.5, "capsize": 1})
                ax.axvline(1.0, color="black", linewidth=0.6, linestyle=(0, (4, 3)))
                ax.set_yticks(y)
                ax.set_yticklabels(labels, fontsize=6)
                ax.invert_yaxis()
                ax.set_xlabel(f"Time relative to default ({mode}, {int(t)} threads)")
                ax.grid(True, axis="x")
                ax.set_axisbelow(True)
                out.fig(fig, f"fig_ablation_{mode}_t{int(t)}{suffix(data, machine)}")
        out.table(f"tab_ablation{suffix(data, machine)}", ["Version", "Threads", "Switch", "Value", "Time [s]", "Rel.", "Peak [MB]"],
                  tex_rows, "lrllrrr", f"machine {machine}; Rel. = time / time of the default configuration")


# ---- E4 --------------------------------------------------------------------------------------
VALUE_ORDER = ["static", "static,1", "dynamic,1", "dynamic,8", "guided", "strips", "tasks", "inner"]


def plot_balance(data, out):
    for machine in data.machines:
        rows = data.rows("E4_load_balance", machine=machine)
        if not rows:
            continue
        loops = [l for l in ("m2_windows", "m6_ssim", "m7_blocks") if any(r["loop"] == l for r in rows)]
        fig, axes = plt.subplots(1, len(loops), figsize=(COL_W, 1.9), sharey=True,
                                 gridspec_kw={"width_ratios": [sum(r["loop"] == l for r in rows) for l in loops]})
        axes = [axes] if len(loops) == 1 else list(axes)
        tex_rows = []
        for ax, loop in zip(axes, loops):
            sel = sorted((r for r in rows if r["loop"] == loop), key=lambda r: VALUE_ORDER.index(r["value"]) if r["value"] in VALUE_ORDER else 99)
            ax.bar(range(len(sel)), [r["imbalance"] for r in sel], color=GREY, width=0.65, linewidth=0)
            ax.axhline(1.0, color="black", linewidth=0.6, linestyle=(0, (4, 3)))
            ax.set_xticks(range(len(sel)))
            ax.set_xticklabels([r["value"] for r in sel], rotation=40, ha="right", fontsize=6)
            ax.set_title(loop.split("_")[0].upper() + " " + loop.split("_")[1], pad=3)
            ax.grid(True, axis="y")
            ax.set_axisbelow(True)
            for r in sel:
                tex_rows.append([loop.split("_")[0].upper() + " " + loop.split("_")[1], tex(r["value"]), int(r["loop_threads"]),
                                 f"{r['busy_max']:.4f}", f"{r['busy_mean']:.4f}", f"{r['imbalance']:.3f}"])
        axes[0].set_ylabel("Busy time, max / mean")
        out.fig(fig, f"fig_load_balance{suffix(data, machine)}")
        out.table(f"tab_load_balance{suffix(data, machine)}", ["Loop", "Schedule", "Threads", "Max [s]", "Mean [s]", "Max/mean"],
                  tex_rows, "llrrrr", f"machine {machine}; per-thread busy time")
        # per-thread busy time of every schedule
        trows = data.rows("E4_load_balance_threads", machine=machine)
        configs = []
        trows = sorted(trows, key=lambda r: (r["loop"], VALUE_ORDER.index(r["value"]) if r["value"] in VALUE_ORDER else 99))
        for r in trows:
            if (r["loop"], r["value"]) not in configs:
                configs.append((r["loop"], r["value"]))
        if configs:
            ncol = 3
            nrow = (len(configs) + ncol - 1) // ncol
            fig, axes = plt.subplots(nrow, ncol, figsize=(COL_W, 0.95 * nrow + 0.3), squeeze=False)
            for ax in axes.flat:
                ax.set_visible(False)
            for ax, (loop, value) in zip(axes.flat, configs):
                ax.set_visible(True)
                sel = sorted((r for r in trows if r["loop"] == loop and r["value"] == value), key=lambda r: r["thread"])
                ax.bar([r["thread"] for r in sel], [r["busy_seconds"] for r in sel], color=GREY, width=0.8, linewidth=0)
                ax.set_title(f"{loop.split('_')[0].upper()} {value}", fontsize=6, pad=2)
                ax.tick_params(labelsize=5, length=2)
                ax.set_xticks([])
            fig.supxlabel("Thread", fontsize=7)
            fig.supylabel("Busy time [s]", fontsize=7)
            fig.tight_layout(pad=0.3)
            out.fig(fig, f"fig_load_balance_threads{suffix(data, machine)}")


# ---- E5 --------------------------------------------------------------------------------------
def task_color(task):
    for k in range(8, 0, -1):
        if f"m{k}" in task.lower():
            return MODULE_COLORS[k - 1]
    return "#bbbbbb"


def plot_gantt(data, out):
    for machine in data.machines:
        for mode in data.values("E5_trace", "mode", machine=machine):
            rows = data.rows("E5_trace", machine=machine, mode=mode)
            most = max(r["threads"] for r in rows)  # a resumed run may hold traces of several thread counts
            rows = [r for r in rows if r["threads"] == most]
            threads = sorted({int(float(r["thread"])) for r in rows})
            if not rows:
                continue
            index = {t: i for i, t in enumerate(threads)}
            fig, ax = plt.subplots(figsize=(COL_W, min(4.5, 0.6 + 0.16 * len(threads) + 0.4)))
            seen = set()
            for r in rows:
                color = task_color(r["task"])
                key = next((f"M{k}" for k in range(8, 0, -1) if f"m{k}" in r["task"].lower()), "other")
                ax.barh(index[int(float(r["thread"]))], r["end"] - r["start"], left=r["start"], height=0.8, color=color,
                        edgecolor="white", linewidth=0.2, label=None if key in seen else key)
                seen.add(key)
            ax.set_yticks(range(len(threads)) if len(threads) <= 16 else range(0, len(threads), max(1, len(threads) // 8)))
            ax.set_ylabel("Thread")
            ax.set_xlabel(f"Time [s] ({mode}, {len(threads)} threads)")
            ax.invert_yaxis()
            handles, labels = ax.get_legend_handles_labels()
            order = sorted(range(len(labels)), key=lambda i: labels[i])
            ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=8, loc="lower center",
                      bbox_to_anchor=(0.5, 1.0), columnspacing=0.6, handlelength=0.8, handletextpad=0.3)
            out.fig(fig, f"fig_gantt_{mode}{suffix(data, machine)}")


# ---- E6 --------------------------------------------------------------------------------------
def strategy_label(r):
    k = int(r["image_workers"])
    if r["strategy"] == "per_image":
        return f"image-level\n{k}x1"
    per = int(r["threads_per_image"]) if r["threads_per_image"] != "" else int(r["threads"] // max(k, 1))
    if r["strategy"] == "in_image":
        return f"{r['mode']}\n1x{per}"
    return f"hybrid\n{k}x{per}"


def plot_throughput(data, out):
    tex_rows = []
    machines = [m for m in data.machines if data.rows("E6_throughput", machine=m)]
    if not machines:
        return
    fig, axes = plt.subplots(len(machines), 1, figsize=(COL_W, 2.1 * len(machines)), squeeze=False)
    for ax, machine in zip(axes[:, 0], machines):
        tp = data.rows("E6_throughput", machine=machine, metric="throughput")
        configs = []
        for r in tp:
            key = (r["strategy"], r["mode"], r["image_workers"])
            if r["rf_order"] == "image" and key not in configs:
                configs.append(key)
        order = {"per_image": 0, "hybrid": 1, "in_image": 2}
        configs.sort(key=lambda c: (order[c[0]], -c[2], c[1]))
        ax2 = ax.twinx()
        labels = []
        for i, (strategy, mode, k) in enumerate(configs):
            for j, prefetch in enumerate((0, 1)):
                sel = [r for r in tp if (r["strategy"], r["mode"], r["image_workers"]) == (strategy, mode, k)
                       and r["prefetch"] == prefetch and r["rf_order"] == "image"]
                if not sel:
                    continue
                r = sel[0]
                ax.bar(i + (j - 0.5) * 0.36, r["median"], width=0.34, color="#1f77b4" if j == 0 else "#9ecae1", linewidth=0,
                       yerr=[[max(0.0, r["median"] - r["min"])], [max(0.0, r["max"] - r["median"])]],
                       error_kw={"elinewidth": 0.5, "capsize": 1}, label=("no prefetch", "prefetch")[j] if i == 0 else None)
                rss = [x for x in data.rows("E6_throughput", machine=machine, metric="peak_rss_mb")
                       if (x["strategy"], x["mode"], x["image_workers"], x["prefetch"], x["rf_order"]) == (strategy, mode, k, prefetch, "image")]
                if rss:
                    ax2.plot(i + (j - 0.5) * 0.36, rss[0]["max"], marker="D", color="#d62728", markersize=3, linestyle="none",
                             label="peak memory" if (i, j) == (0, 0) else None)
                if j == 0:
                    labels.append(strategy_label(r))
        for r in sorted(tp, key=lambda r: (order[r["strategy"]], -r["image_workers"], r["mode"], r["rf_order"], r["prefetch"])):
            rss = [x for x in data.rows("E6_throughput", machine=machine, metric="peak_rss_mb")
                   if all(x[c] == r[c] for c in ("strategy", "mode", "image_workers", "prefetch", "rf_order"))]
            tex_rows.append([tex(machine), strategy_label(r).replace("\n", " "), "yes" if r["prefetch"] else "no", r["rf_order"],
                             f"{r['median']:.2f}", f"{rss[0]['max']:.0f}" if rss else "--"])
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylabel("Images per second")
        ax2.set_ylabel("Peak memory [MB]")
        ax2.spines["right"].set_visible(True)
        ax2.set_ylim(bottom=0)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        if len(machines) > 1:
            ax.set_title(machine, pad=2)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0 if len(machines) == 1 else 1.08))
    fig.tight_layout(pad=0.4)
    out.fig(fig, "fig_throughput")
    out.table("tab_throughput", ["Machine", "Strategy (images x threads)", "Prefetch", "Forest order", "Images/s", "Peak [MB]"],
              tex_rows, "lllrrr", "throughput over the image set; peak memory is the maximum over the runs")


# ---- E7 and machines -------------------------------------------------------------------------
def table_datasets(data, out):
    rows = []
    for r in data.tables.get("E7_dataset_pi", []):
        if len(data.machines) > 1 and r["machine"] != data.machines[0]:
            continue  # scores do not depend on the machine
        rows.append([tex(r["set"]), tex(r["method"]), int(r["n_valid"]), f"{r['mean_ma']:.3f}", f"{r['mean_niqe']:.3f}", f"{r['pi']:.3f}"])
    out.table("tab_dataset_pi", ["Image set", "Method", "Images", "Ma", "NIQE", "PI"], rows, "llrrrr",
              "PI = ((10 - mean Ma) + mean NIQE) / 2, PIRM protocol (Y channel, 4-pixel border removed)")


def table_machines(data, out):
    rows = []
    for r in data.tables.get("machine", []):
        lc = int(r["logical_cpus"]) if r["logical_cpus"] != "" else "--"
        pc = int(r["physical_cores"]) if r["physical_cores"] != "" else "--"
        rows.append([tex(r["machine"]), tex(r["cpu_model"]), pc, lc, r["mem_gb"], tex(r["virtualization"] or "--"),
                     tex(r["compiler"])])
    out.table("tab_machines", ["Machine", "CPU", "Cores", "Threads", "Memory [GB]", "Virtualization", "Compiler"], rows, "llrrrll")


def table_pyiqa(data, out):
    rows = [[tex(r["machine"]), tex(r["input"]), tex(r["device"]), f"{r['median']:.3f}", f"{float(r['pi']):.4f}"]
            for r in data.tables.get("E8_pyiqa", [])]
    out.table("tab_pyiqa", ["Machine", "Input", "Device", "Time [s]", "PI"], rows, "lllrr", "PI computed by the pyiqa package")


# ---- driver ----------------------------------------------------------------------------------
def resolve(path, temp_dirs):
    """Results directory for a directory or an archive argument."""
    if os.path.isfile(path):
        tmp = tempfile.mkdtemp(prefix="pi-results-")
        temp_dirs.append(tmp)
        with tarfile.open(path) as t:
            base = os.path.realpath(tmp)
            for m in t.getmembers():
                if not os.path.realpath(os.path.join(tmp, m.name)).startswith(base + os.sep) or m.issym() or m.islnk():
                    raise SystemExit(f"{path}: unsafe member {m.name}")
            try:
                t.extractall(tmp, filter="data")
            except TypeError:  # Python older than 3.12 without the filter argument
                t.extractall(tmp)
        path = tmp
    for root, dirs, _files in os.walk(path):
        if "csv" in dirs or "raw" in dirs:
            if "csv" not in dirs:
                summarize_results.summarize(root)
            return root
        if root.count(os.sep) - path.count(os.sep) >= 2:
            break
    raise SystemExit(f"{path}: neither csv/ nor raw/ found")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="results directory or pi-results-*.tar.gz archive")
    ap.add_argument("--out", help="output directory (figures/ and tables/ are created inside)")
    a = ap.parse_args()

    import shutil
    temp_dirs = []
    try:
        dirs = [resolve(p, temp_dirs) for p in a.results]
        if a.out:
            out_dir = a.out
        elif len(a.results) > 1:
            out_dir = os.path.join("results", "combined")
        elif os.path.isfile(a.results[0]):
            out_dir = a.results[0].replace(".tar.gz", "").replace(".tgz", "")
        else:
            out_dir = dirs[0]
        style()
        data = Data(dirs)
        if not data.machines:
            raise SystemExit("no machine.csv found in the results")
        if len(set(data.machines)) < len(dirs):
            print("warning: several results share one machine id; their rows are merged", file=sys.stderr)
        out = Output(out_dir)
        steps = [plot_scaling, plot_module_scaling, table_scaling, plot_cost, plot_ablation, plot_balance, plot_gantt,
                 plot_throughput, table_datasets, table_machines, table_pyiqa]
        problems = 0
        for step in steps:
            try:
                step(data, out)
            except Exception as e:  # noqa: BLE001 - one broken figure must not block the others
                problems += 1
                print(f"warning: {step.__name__} failed: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"{len(out.made)} file(s) written to {out.fig_dir} and {out.tab_dir}"
              + (f" ({problems} step(s) failed)" if problems else ""))
        return 1 if problems else 0
    finally:
        for d in temp_dirs:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

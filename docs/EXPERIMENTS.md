# Experiments

One command sets up the environment, builds and checks the program, fetches the data, runs
every experiment, and packs the results:

```bash
bash run_all.sh            # full campaign
bash run_all.sh --quick    # smoke test of a few minutes (bundled images, no downloads)
bash run_all.sh --help
```

Requirements: Linux, `g++` (C++17, OpenMP), `make`, `python3`, `git`, and `curl` or `wget`.
No root access is needed. The script adapts to the number of CPUs and can be run again at any
time: measurements that already have a report are kept, only missing ones are run.

The result is a single archive, `results/pi-results-<hostname>-<YYYYmmdd-HHMMSS>.tar.gz`, with
CSV files, raw reports, logs, the machine description, figures and tables. Datasets and model
files are never included.

A full run takes roughly 45 to 75 minutes on a 20-core machine, plus downloads (about 0.8 GB)
and the faithful variant of E1; a forecast for the machine is printed before the experiments
start.

## Useful options

| Option | Effect |
|---|---|
| `--only E2,E3` | run only the listed experiments (E1 to E8) |
| `--threads-max N` | never use more than N threads |
| `--reps N` | repetitions per configuration (default 5; the median is reported) |
| `--skip-faithful` | leave out the faithful sequential variant in E1 (minutes per image) |
| `--div2k-subset N` | keep and use only the first N DIV2K validation images |
| `--skip-optional-data` | download DIV2K only |
| `--offline` | no downloads; use what is in `data/` and the bundled test images |
| `--with-pyiqa`, `--with-perf` | optional comparisons of E8 |
| `--machine-id NAME` | label of the machine in tables and figures (default: host name) |
| `--force` | repeat measurements that already have a report |

## Stages

1. **Environment.** Checks the tools, creates `.venv` with numpy, scipy, pillow and matplotlib
   (falls back to `pip install --user`), and records the machine in `machine/`.
2. **Models.** `tools/fetch_models.sh`, skipped when `models/*.bin` exist.
3. **Build and check.** `make`, then `make check`. A failed check does not stop the run; it is
   flagged in the summary and by a `CHECK_FAILED.txt` file in the archive. `--quick` runs the
   same steps with the short list of parallel configurations.
4. **Data.** `tools/fetch_data.sh` (see below).
5. **Experiments.** `tools/run_experiments.py` (E1 to E7), then the optional hooks (E8).
6. **Summaries.** `tools/summarize_results.py` writes `csv/` and `summary.txt`.
7. **Figures and tables.** `tools/plot_results.py` writes `figures/*.pdf` and `tables/*.tex`.
8. **Archive.**

## Data

`tools/fetch_data.sh` fills the untracked `data/` directory and can be run on its own.

| Item | Source | Required |
|---|---|---|
| DIV2K validation HR (100 images, about 450 MB) | `data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip` | yes, with fallback |
| Set5, Set14, BSD100, Urban100 (HR) | Hugging Face `eugenesiow/<set>`; fallback: GitHub `jbhuang0604/SelfExSR` (x4 HR crops) | no |
| PIRM 2018 image sets | Dropbox archive linked from GitHub `roimehrez/PIRM2018` | no |
| PIRM 2018 baseline outputs | `pirm2018.org`, linked from `pirm.github.io` | no |
| ESRGAN outputs | one sample image in GitHub `xinntao/ESRGAN`; the complete sets are on Google Drive and are fetched only when `gdown` is installed | no |

Every optional item that cannot be fetched is skipped and listed with its reason in
`data_status.tsv`. Without DIV2K the run still completes: the fixed inputs are replaced by
synthetic images of the same sizes (a bundled test image, mirror-tiled) and the image set by
x4 bicubic versions of the bundled test images. The manifests record which source was used.

Layout: `data/hr/<set>/` holds originals, `data/sr/<set>/<method>/` holds the images that are
scored. Results of other super-resolution methods can be added by hand as
`data/sr/<set>/<method>/*.png`; E7 scores every such directory.

**Fixed inputs.** `tools/make_inputs.py inputs` writes `data/inputs/crop1080x800.png`, the
1080x800 centre crop of DIV2K validation image 0801, and `data/inputs/full2k.png`, the complete
image. With `--quick` the inputs are two bundled test images (`quick_small`, `quick_large`).

**Bicubic x4 outputs.** The Perceptual Index scores super-resolution outputs, so for every
HR set `tools/make_inputs.py bicubic` produces `data/sr/<set>/bicubic_x4/`: the image is cropped
(top-left) to a multiple of 4, reduced by 4 and enlarged by 4 with Pillow's
`Image.resize(..., Image.BICUBIC)` on 8-bit RGB. This is not MATLAB's `imresize`; scores differ
slightly from numbers published for MATLAB-generated bicubic images.

## Measurement protocol

- Every number comes from the file that `pi_eval --report` writes; nothing is read from the
  screen. All runs use the PIRM protocol (`--pirm`: Y channel, 4-pixel border removed).
- Timing runs pass `--warmup`. A single-image configuration is repeated inside one process
  (the image is given R times), and the report has one line per repetition. Exceptions: the
  faithful variant (one evaluation, no warm-up) and the load-balance runs of E4 (no warm-up,
  because the counters accumulate over everything the process evaluates).
- Throughput runs (E6) are repeated as separate processes (default 3).
- Threads are pinned: `OMP_PROC_BIND=close OMP_PLACES=cores` for single-image runs and
  `OMP_PROC_BIND=spread,close OMP_PLACES=cores` for image-set runs, so that images are spread
  over the machine and the threads of one image stay on neighbouring cores. The environment of
  each run is stored in its `.meta.json` file. `--no-pin` turns pinning off.
- Speedup is the median wall time of the sequential baseline (`pi_eval_seq`, built without
  OpenMP) on the same input and machine, divided by the median wall time of the configuration.
  Efficiency is speedup divided by the thread count.
- The sequential binary does not read the OpenMP environment, so it is pinned with `taskset`
  to one CPU of the fastest core. Unpinned, it could migrate during a run.
- Thread counts: powers of two up to the number of usable CPUs, that number itself, and the
  number of physical cores when hyper-threading is detected.
- Hybrid processors (performance and efficiency cores, detected from the maximum frequency
  that cpufreq reports per core) get three adjustments. The places are listed explicitly with
  the fastest cores first, so T threads always run on the T fastest cores. The number of fast
  cores is added to the thread counts. E1 also times the baseline on one of the slowest cores
  (variant `baseline_slow_core`). The ablations (E3) and the load-balance runs (E4) use the
  fast cores alone as their second thread count, so that one set of runs is free of the
  imbalance that unequal cores cause by themselves. On such a machine the speedup at full thread count cannot
  reach the thread count: with f fast cores, s slow cores and a slow core taking r times as
  long as a fast one for the baseline, the capacity is f + s / r fast-core equivalents.

## Experiments

| | What | Runs |
|---|---|---|
| E1 | Cost breakdown | per-module time of the sequential baseline and of the faithful variant on the 1080x800 crop |
| E2 | Strong scaling | modes `naive`, `intra`, `graph` over the thread sweep on both inputs: wall time, module times, speedup, efficiency |
| E3 | Ablations | each design switch against the default, at the maximum thread count and at half of it, for `intra` and `graph`: `--m2-sched`, `--m7-sched`, `--m4-level`, `--m5-gather` (with peak memory), `--m6-ssim`, `--rf-sum`, `--m1`, `--light` |
| E4 | Load balance | `--balance`: per-thread busy time of the M2, M6 and M7 loops under each schedule |
| E5 | Task trace | `--trace` for `intra` and `graph` at the maximum thread count (Gantt chart) |
| E6 | Throughput | images per second and peak memory over an image set: one image per thread, all threads inside one image (`intra`, `graph`), hybrids, each with and without `--prefetch`; `--rf-order image` against `tree` |
| E7 | Real data | `tests/check_parallel.py --quick --pirm` on dataset images, and the PI of every image set and method |
| E8 | Optional | `cuda/run_experiments.sh` when a `cuda/` directory, `nvcc` and a GPU exist; `--with-pyiqa`: PI of the pyiqa package on the same input; `--with-perf`: `tools/profile.sh` |

E6 uses the DIV2K bicubic outputs (by default the first max(24, 2 x threads) images;
`tools/run_experiments.py --e6-images -1` uses all of them), or the bundled fallback set.

`tools/run_experiments.py` can also be called directly, for example
`python3 tools/run_experiments.py --results results/run-$(hostname -s) --only E3 --reps 9`.

## Results layout

```
results/run-<hostname>/            (in the archive: pi-results-<hostname>-<date>/)
  RUN_SUMMARY.txt                  stages that ran, were skipped or failed, with reasons
  summary.txt                      the main numbers in readable form
  CHECK_FAILED.txt                 only present when the accuracy check failed
  run_info.json                    machine id, commit, options, stage status
  experiment_plan.json             thread sweep, repetitions, inputs, image sets
  experiment_status.json           status of E1 to E7, list of failed runs
  data_status.tsv, manifest*.json  what data was available and where it came from
  machine/                         lscpu, memory, kernel, compiler, git, virtualization, GPU, OpenMP environment
  logs/                            one log per stage (03_check.log is the accuracy check)
  raw/E1 ... raw/E7/               <name>.report, <name>.meta.json, <name>.log per run; E5: <name>.trace
  csv/                             tidy tables (below)
  figures/*.pdf, tables/*.tex
  cuda/, pyiqa/, perf/             optional
```

## CSV files

All files are long format with a header row. `machine` is the machine id.

Configuration columns (`CONFIG`), copied from the `config` line of the report:
`mode, variant, threads, image_workers, threads_per_image, prefetch, light, m1, m2_sched,
m7_sched, m4_level, m5_gather, m6_ssim, rf_sum, rf_order`.

Statistics columns (`STATS`): `metric, unit, n, median, min, max`.

| File | Columns | Metrics |
|---|---|---|
| `machine.csv` | `machine, hostname, cpu_model, logical_cpus, physical_cores, sockets, threads_per_core, threads_max, mem_gb, virtualization, compiler, kernel, git_commit, gpu, omp_pinning, quick, check` | |
| `E1_cost_breakdown.csv` | `machine, experiment, input, CONFIG, STATS, share` | `wall`, `io`, `m1`..`m8`, `modules_sum`; `share` = fraction of the summed module time |
| `E2_scaling.csv` | `machine, experiment, input, CONFIG, STATS` | `wall`, `m1`..`m8`, `speedup`, `efficiency`, `speedup_m1`..`speedup_m8`; the baseline has `mode=seq` |
| `E3_ablations.csv` | `machine, experiment, input, CONFIG, group, value, is_default, STATS, rel_to_default` | `wall`, `m1`..`m8`, `peak_rss_mb`; `rel_to_default` = wall / wall of the default (same mode and threads) |
| `E4_load_balance.csv` | `machine, experiment, input, CONFIG, group, value, loop, loop_threads, busy_max, busy_mean, imbalance` | one row per schedule; `imbalance` = max / mean |
| `E4_load_balance_threads.csv` | `machine, input, mode, threads, group, value, loop, thread, busy_seconds` | |
| `E5_trace.csv` | `machine, input, mode, threads, image, task, start, end, thread` | seconds from the start of the batch |
| `E6_throughput.csv` | `machine, experiment, input, CONFIG, strategy, set, n_images, STATS` | `throughput` (images/s), `wall`, `peak_rss_mb`, `io`; `strategy` is `per_image`, `in_image` or `hybrid` |
| `E7_dataset_pi.csv` | `machine, set, method, n_images, n_valid, pi, mean_ma, mean_niqe, wall, throughput` | `pi` = ((10 - mean Ma) + mean NIQE) / 2 |
| `E7_image_scores.csv` | `machine, set, method, image, valid, ma, niqe, pi` | |
| `E7_correctness.csv` | `machine, configuration, feature_rel, forest_abs, niqe_abs, pi_abs, result` | |
| `E8_pyiqa.csv` | `machine, input, device, pi, STATS` | |

A file is written only when its experiment produced data.

## Regenerating figures

```bash
# from a results directory or from an archive
python3 tools/plot_results.py results/run-myhost
python3 tools/plot_results.py pi-results-myhost-20260101-120000.tar.gz

# several machines in one set of figures (one series per machine)
python3 tools/plot_results.py pi-results-a-*.tar.gz pi-results-b-*.tar.gz --out results/combined

# rebuild the CSV files from the raw reports
python3 tools/summarize_results.py results/run-myhost
```

`plot_results.py` needs matplotlib only. Figures are vector PDF, 3.4 in wide with 8 pt text,
meant for one column of a two-column paper; tables are `tabular` environments using booktabs
rules, ready for `\input`.

| Figure | Content |
|---|---|
| `fig_speedup_<input>`, `fig_efficiency_<input>` | speedup and efficiency against threads, with the ideal line |
| `fig_module_time_<input>_<mode>` | stacked per-module time for each thread count |
| `fig_module_speedup_<input>_<mode>` | speedup of each module |
| `fig_cost_breakdown` | share of each module in the sequential time |
| `fig_ablation_<mode>_t<threads>` | time of each switch value relative to the default |
| `fig_load_balance`, `fig_load_balance_threads` | max / mean busy time, and busy time per thread |
| `fig_gantt_<mode>` | module tasks on each thread over time |
| `fig_throughput` | images per second per strategy, with peak memory |

Tables: `tab_machines`, `tab_cost_breakdown`, `tab_scaling_<input>`, `tab_ablation`,
`tab_load_balance`, `tab_throughput`, `tab_dataset_pi`, `tab_pyiqa`.

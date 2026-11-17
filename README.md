# Parallel Perceptual Index

Fast, multi-core C++ implementation of the **Perceptual Index (PI)**, the no-reference metric used to evaluate perceptual image super-resolution ([PIRM 2018](https://github.com/roimehrez/PIRM2018)).

```
PI = ((10 − Ma) + NIQE) / 2      # lower is better
```

- **Accurate.** Matches the official MATLAB code: Ma is bit-identical, and PI differs by less than 1e-4.
- **Fast.** A sequential baseline that removes the redundant work of the official code, and OpenMP versions that use every core, inside an image and across the images of a test set.
- **Reproducible.** The parallel versions return the same bits for any thread count and schedule.
- **Dependency-free.** C++17 and OpenMP only: no BLAS, FFTW or OpenCV. An optional CUDA extension lives in [`cuda/`](cuda/README.md).

## Quick start

```bash
git clone https://github.com/mxzhuang/parallel-perceptual-index.git
cd parallel-perceptual-index
bash tools/fetch_models.sh   # needs Python 3 with numpy and scipy
make
```

```bash
$ ./pi_eval --mode graph --pirm results/*.png
image                                      Ma         NIQE           PI
coffee.png                           8.834929     4.079660     2.622366
coffee_x4bicubic.png                 3.969566     7.504718     6.767576
Perceptual Index over 2 image(s): 4.694971  (mean Ma 6.402247, mean NIQE 5.792189)
```

| Option | Description |
|---|---|
| `--pirm` | PIRM protocol: Y channel, 4-pixel border shave |
| `--mode seq\|naive\|intra\|graph` | Sequential baseline (default), `parallel for` on the hot loops, modules in sequence with parallel loops inside, or modules as dependency-ordered tasks |
| `--threads N` | Total thread budget (default: all cores) |
| `--image-workers K` | Images evaluated at the same time; each gets N / K threads |
| `--prefetch` | Decode upcoming images on a helper thread |
| `--timing`, `--balance`, `--trace FILE` | Per-module times, per-thread load, task timeline |
| `--models DIR` | Model directory (default `models`) |

`./pi_eval --help` lists every option, including the design switches used in the experiments. For a whole test set, `--mode intra --image-workers N` with one thread per image usually gives the highest throughput; `--mode graph` is the fastest way to score one image.

## How it is organized

PI is split into eight modules by data dependency. M4 to M6 hold over 90% of the sequential time.

```
M1 spatial pyramid ──┬─> M2 block-DCT statistics ──────────────┐
                     └─> M3 patch SVD ─────────────────────────┤
M4 steerable pyramid ┬─> M5 divisive normalization ─> M6b ─────┼─> M8 random forests ─> Ma ─┐
                     └─> M6a band / high-pass SSIM ────────────┘                            ├─> PI
M7 NIQE ────────────────────────────────────────────────────────────────────────────────────┘
```

Parallelism exists at three levels, all under one fixed thread budget: inside a module (pixels, windows, bands, trees), across modules (an OpenMP task graph) and across images. [docs/DESIGN.md](docs/DESIGN.md) explains each decision.

## Testing

```bash
make check
```

Checks the sequential baseline against reference outputs of the official code, then every parallel configuration against the baseline, including that all of them agree bit for bit.

## Experiments

```bash
bash run_all.sh            # environment, data, build, checks, all experiments, one archive
bash run_all.sh --quick    # smoke test of a few minutes
```

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the experiments, the result files and how to regenerate the figures. `tools/profile.sh` profiles the sequential versions with `time`, `gprof` and `perf`; `--faithful` repeats the redundant work of the official code so that algorithmic gains can be measured apart from parallel speedup.

## Credits

Based on the official implementations of [Ma et al.](https://github.com/chaoma99/sr-metric) (CVIU 2017) and NIQE ([Mittal et al.](https://github.com/roimehrez/PIRM2018), IEEE SPL 2013). Please cite the original papers. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

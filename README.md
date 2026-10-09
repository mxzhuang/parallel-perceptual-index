# Parallel Perceptual Index

Fast C++ implementation of the **Perceptual Index (PI)**, the no-reference metric used to evaluate perceptual image super-resolution ([PIRM 2018](https://github.com/roimehrez/PIRM2018)).

```
PI = ((10 − Ma) + NIQE) / 2      # lower is better
```

- **Accurate.** Matches the official MATLAB code: Ma is bit-identical, and PI differs by less than 1e-4.
- **Fast.** About 1.7 s per image on a single core. The official code takes minutes.
- **Dependency-free.** Pure C++17, with no BLAS, FFTW or OpenCV.

> Work in progress: multi-core versions are coming.

## Quick start

```bash
git clone https://github.com/mxzhuang/parallel-perceptual-index.git
cd parallel-perceptual-index
bash tools/fetch_models.sh   # needs Python 3 with numpy and scipy
make
```

```bash
$ ./pi_eval --pirm results/*.png
image                                      Ma         NIQE           PI
coffee.png                           8.834929     4.079660     2.622366
coffee_x4bicubic.png                 3.969566     7.504718     6.767576
Perceptual Index over 2 image(s): 4.694971  (mean Ma 6.402247, mean NIQE 5.792189)
```

| Option | Description |
|---|---|
| `--pirm` | PIRM protocol: Y channel, 4-pixel border shave |
| `--timing` | Per-stage run time |
| `--models DIR` | Model directory (default `models`) |

| `--faithful` | Do exactly the work of the official code (slow; for profiling) |

## Testing

```bash
make check   # compares against reference outputs of the official code
```

## Profiling

Per-stage timing (M1–M8) of three sequential versions, pinned to one core:

```bash
# 1. Official MATLAB code (clone chaoma99/sr-metric and roimehrez/PIRM2018 first)
matlab -batch "maxNumCompThreads(1); addpath('reference'); \
  time_official_modules('tests/images/coffee_y.png', 'sr-metric', 'PIRM2018/utils/niqe_release', '', 3)"

# 2. Faithful C++ port: same work as the official code
taskset -c 0 ./pi_eval --faithful --timing tests/images/*_y.png

# 3. Optimized C++ baseline: redundant work removed, same output
taskset -c 0 ./pi_eval --timing tests/images/*_y.png
```

## Credits

Based on the official implementations of [Ma et al.](https://github.com/chaoma99/sr-metric) (CVIU 2017) and NIQE ([Mittal et al.](https://github.com/roimehrez/PIRM2018), IEEE SPL 2013). Please cite the original papers. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

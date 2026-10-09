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

## Testing

```bash
make check   # compares against reference outputs of the official code
```

## Credits

Based on the official implementations of [Ma et al.](https://github.com/chaoma99/sr-metric) (CVIU 2017) and NIQE ([Mittal et al.](https://github.com/roimehrez/PIRM2018), IEEE SPL 2013). Please cite the original papers.

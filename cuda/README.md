# CUDA extension (optional)

`pi_eval_cuda` computes the Perceptual Index with the two heaviest per-pixel stages on a GPU:
M5 (divisive normalization) and the 27 SSIM structure terms of M6. Everything else runs on
the CPU with the OpenMP code of the main program. The main program does not depend on this
directory; delete it and `pi_eval` still builds.

```bash
make -C cuda            # needs nvcc and an NVIDIA GPU; builds ./pi_eval_cuda
./pi_eval_cuda --pirm image.png
bash cuda/run_experiments.sh results/cuda     # timings for the report
```

**Status.** The kernels were checked by running them on a CPU emulation of the CUDA thread
model (real threads per block, real barriers for `__syncthreads()`): on a small image the
SSIM features agree with the CPU modules to 7e-14 and all 18 GGD shapes are identical. Run
`make -C cuda check` on a machine with a GPU to repeat the comparison on real hardware; the
program prints the differences for every image.

## Division of work

| Stage | Where | Why |
|---|---|---|
| M1-M3, M7, M8 | CPU | Small; M8 is branch-heavy tree traversal, which diverges inside a warp |
| M4 steerable pyramid | CPU | Full-image FFTs of arbitrary length; no GPU FFT library is used |
| Upsampling of the coarse bands | CPU | Reuses the validated bicubic resize |
| M5 second moments | GPU | Reduction over every position of a band |
| M5 15x15 eigen-decomposition | CPU | Tiny and serial |
| M5 normalization, guard-band mean | GPU | One thread per position |
| M6 SSIM structure terms | GPU | Tiled separable convolution fused with the per-pixel formula |
| M6 GGD fits | CPU | 18 short reductions followed by a table lookup that is sensitive to rounding |

The pyramid (13 bands, plus the upsampled coarse bands) is uploaded once. What comes back is
per-block partial sums and the twelve normalized bands.

## Kernels

- **Second moments.** Each thread walks a strided subset of the positions and accumulates
  the upper triangle of `y y'` (up to 120 values) privately. The threads of a warp read
  consecutive positions in every step, so the reads coalesce. At the end the block reduces
  one matrix entry at a time with a tree in shared memory, separated by `__syncthreads()`.
  Blocks cannot synchronize with each other, so each block writes its own partial matrix and
  the host adds the partial matrices in block order.
- **Normalization.** One thread per kept position gathers its neighbourhood vector and
  evaluates the quadratic form with the inverse covariance held in constant memory.
- **SSIM.** A block owns a 16x16 tile of the output. It loads the tile and a 10-pixel halo of
  both inputs into shared memory, runs the vertical pass of the five filtered maps into
  shared memory, then the horizontal pass and the SSIM formula in one step, and reduces the
  tile to one partial sum. No filtered map is written to global memory.

All kernels use double precision and are compiled with `--fmad=false`, so every per-pixel
value is rounded as in the CPU build. Sums are combined in a fixed order (threads within a
block by a tree, blocks in index order), so results are reproducible, but they are grouped
differently from the CPU sums. `pi_eval_cuda` therefore reports, for every image, the
largest relative difference of the SSIM features and how many of the 18 GGD shapes differ
from the CPU modules run on the same pyramid.

## What is timed

`pi_eval_cuda` prints every stage separately: CPU stages, upsampling, host-to-device upload,
each kernel group (including the download of its partial sums), the host algebra between the
two M5 kernels, the download of the normalized bands and the GGD fits. The comparison figure
is "M5+M6 with GPU" (all of the above from upsampling to GGD fits) against M5+M6 of the
OpenMP version with the same number of CPU threads. Context creation happens once before
the timed region.

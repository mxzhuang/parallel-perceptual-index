# Design notes

How the Perceptual Index is parallelized, and why each choice was made. The module
numbers (M1 to M8) are those of `src/modules.h`.

## Versions

| `--mode` | What runs | Purpose |
|---|---|---|
| `seq` | Optimized sequential baseline | Reference for every speedup; `pi_eval_seq` is the same code built without OpenMP |
| `naive` | Baseline with `parallel for` on the hottest loops of M4, M5 and M6 | What a straightforward port achieves |
| `intra` | Modules one after another, each parallel inside | Parallelism within modules |
| `graph` | Modules as OpenMP tasks ordered by data dependencies | Parallelism within and across modules |

`--faithful` (with `seq`) repeats the redundant work of the official MATLAB code, so the
gain of the algorithmic fixes can be measured separately from parallel speedup.

## One thread budget

Three levels of parallelism exist: loops inside a module, independent modules, and
independent images. If each level opened its own parallel region, the number of threads
would multiply. Two rules keep the total at `--threads`:

- Every loop of the designed versions goes through `par_for` (`src/parallel.h`). Outside the
  task graph it is a `parallel for` with exactly the budgeted thread count. Inside the task
  graph it is a `taskloop` on the team that is already running, so nested loops add tasks,
  not threads. A loop nested in another `par_for` runs serially.
- At the image level, `--image-workers K` runs K images at once and gives each one
  `threads / K` threads. Only two nesting levels are ever active.

A handful of unequal units (12 bands, 3 scales, 13 inverse FFTs) cannot be balanced across
a large team, so they go through `par_outer`: tasks inside the task graph, a plain loop
elsewhere, which leaves all threads to the loops inside each unit.

## No shared writable state

The designed versions contain no lock in any hot path. Every chunk of a parallel loop
writes either its own range of the output or its own slot, and reads shared data only.
Merging happens after the loop, once per chunk, in a fixed order. The only mutexes guard
events that happen a few times per image (the task trace, error capture) and the image
prefetch buffer.

Exceptions must not leave an OpenMP region, so `par_for` catches the first one and rethrows
it on the calling thread.

## Reproducible reductions

Floating-point addition is not associative, and an OpenMP `reduction` combines partial
results in an order that depends on the thread count. That matters here: several features
end in a table lookup, where a difference in the last bit can select another entry.

The designed versions therefore reduce over a chunk grid that depends on the problem size
only (`reduce_chunks`, `det_sum`): each chunk is summed in index order into its own slot and
the slots are added in chunk order. The result is the same for 1 or 64 threads, any
schedule and any design switch; `make check` compares the outputs of about 35
configurations bit for bit. Compared with the sequential baseline the sums are grouped
differently, so features agree to about 1e-12 (relative) instead of exactly; the forest
outputs and the scores agree exactly on the test images. Where the order can be kept
(the per-tree sums of M8, all loops without a reduction) results are bit-identical to the
baseline.

Builds use `-ffp-contract=off`, so no compiler fuses a multiply and an add and every
platform rounds the same way.

## Cache lines

Matrices are column-major. A range of whole columns is contiguous, so loops hand out
contiguous column ranges, and two threads can touch the same cache line only at the border
between two chunks. Per-chunk accumulators are locals, written to a `Padded` slot (one
cache line each) once at the end of the chunk. Two places need more care:

- **Row pass of the 2-D FFT (M4).** A row is strided and neighbouring rows interleave in
  memory. Each chunk owns a contiguous block of rows and moves eight rows at a time through
  a row-major scratch buffer.
- **Cyclic schedules (M2).** `--m2-sched static,1` gives neighbouring window columns to
  different threads. It balances the load slightly better and pays for it with shared cache
  lines at every column border; the experiments measure both effects.

Scratch memory is allocated once per chunk, never per element, so threads do not contend in
the memory allocator (the baseline of M7 allocated several matrices per block).

## Modules

**M4, steerable pyramid.** A 2-D FFT is a column pass followed by a row pass, with a join
between them. After the forward transform, the high-pass residual and the twelve oriented
bands are thirteen independent inverse transforms. `--m4-level lines` parallelizes the 1-D
transforms inside each 2-D FFT; `--m4-level bands` distributes whole inverse transforms.
The transform always covers the whole image, because tiling would change the metric. FFT
plans are shared and read-only; workspaces are private.

**M5, divisive normalization.** Each band takes two passes over its positions with a tiny
serial 15x15 eigen-decomposition between them. The second-moment matrix is accumulated by
splitting the positions: each chunk sums a private matrix, and the matrices are added in
chunk order. Splitting the 120 matrix entries among threads instead would need no
reduction, but every entry is a sum over all positions, so the data would be swept many
times. `--m5-gather fused` builds each neighbourhood vector on the fly;
`materialized` stores all of them first (about 100 MB per fine band at 1080x800).

**M6, subband statistics.** 27 SSIM structure terms and 18 GGD fits.
`--m6-ssim tasks` runs one SSIM per unit (simple, but 27 unequal units do not divide evenly
among many threads); `inner` parallelizes each filtering pass (balanced, one join per pass);
`strips` cuts every SSIM into 64-column strips and schedules all strips dynamically. A
strip computes its five filtered maps and the formula in buffers that fit in cache and
produces one partial sum, so there is one join per group of SSIMs and no full-size
intermediate map. The twelve terms that compare raw bands with the high-pass residual need
only M4 and overlap with M5 in the task graph. Each GGD fit is an independent unit that
sums sequentially, which keeps the fitted shapes identical to the baseline.

**M2, block-DCT statistics.** The windows are independent; each chunk keeps its window and
DCT in locals and writes its own slice of the statistics. The pooled values are means of
the lowest or highest 10% over a whole scale, with exact sample counts, so threads cannot
pool their own slices: the three statistics are sorted as three independent units.

**M3, patch SVD.** A tall-skinny QR: blocks of 1024 rows are factored independently, the
triangular factors are stacked in block order and reduced once more, and a one-sided Jacobi
iteration on the 25x25 result gives the singular values. The block grid depends on the
matrix size only.

**M1, spatial pyramid.** The three scales form a chain; inside a scale every output column
is independent. Only every second row and column of a filtered image is kept, so the
designed versions filter only those pixels (`--m1 fused`).

**M7, NIQE.** The pixel phase (two 7x7 filters and the normalization) is parallel over
columns, the block phase over blocks, and the two scales are independent of each other. The
small model phase stays serial.

**M8, random forests.** Each tree writes its leaf value to its own slot and the slots are
summed in tree order, which reproduces the sequential output exactly. One prediction visits
only tens of thousands of nodes, so parallelism pays off mainly across a batch:
`--rf-order tree` applies each tree to every image of the batch while its nodes are in
cache.

## Task graph

```
M4 ─> M5 ─> M6b ─> forest 2        critical path
M4 ─> M6a ──────────^              overlaps with M5
M1 ─> M2 ─> forest 1
M1 ─> M3 ─> forest 3
M7
```

Each arrow is a `depend` clause. A task that reads an object is created after the task that
writes it, and no task holds a lock while it waits for another, so the graph cannot
deadlock. The chain M4 to M6 is created first because it bounds the completion time; the
light modules fill cores that the chain leaves idle at its serial points. `--light serial`
runs each light module as a single task instead of splitting it further.

## Across images

Images are independent. `--image-workers K` chooses between one image per thread (no
synchronization inside an image, but K working sets in memory), all threads inside one
image (small memory, a join at every pass) and hybrids. Results go to per-image slots and
are printed in input order.

`--prefetch` decodes upcoming images on a helper thread, a bounded producer-consumer
buffer with a mutex and two condition variables whose predicates are re-checked after every
wake-up. The buffer holds one image more than there are workers, and workers claim images in
increasing order. When the producer is blocked on a full buffer, at least one buffered image
therefore belongs to a worker that is already asking for it, which rules out a deadlock.

## Validity

Images smaller than 96 pixels on a side after preprocessing, images for which a module
fails, and scores that come out as NaN are reported as invalid and excluded from the set
mean; the exit status is 2 if any image could not be scored.

## CUDA extension

See [cuda/README.md](../cuda/README.md). M5 and the SSIM terms of M6 run on the GPU; M4
stays on the CPU because it needs FFTs of arbitrary length and no GPU FFT library is used,
and M8 stays on the CPU because tree traversal diverges inside a warp.

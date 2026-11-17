// Run-time configuration: which version runs (sequential baseline, naive OpenMP, the
// designed parallel versions) and the switches used for the ablation experiments.
#pragma once

#include "variant.h"

// How one image is evaluated.
//   Sequential  the optimized sequential baseline; the reference for every speedup.
//   Naive       the baseline with `parallel for` placed on its hottest loops only (M4-M6).
//   Intra       modules run one after another, each parallel inside.
//   Graph       modules run as OpenMP tasks ordered by their data dependencies; the loops
//               inside a module become taskloops that share the same thread team.
enum class Mode { Sequential, Naive, Intra, Graph };

enum class SchedKind { Static, Dynamic, Guided };
struct Sched {
    SchedKind kind = SchedKind::Static;
    long chunk = 0;  // iterations per chunk; 0 = default of the schedule kind
};

enum class M4Level { Lines, Bands };          // parallelize 1-D FFTs inside a 2-D FFT, or whole bands
enum class M5Gather { Fused, Materialized };  // build neighbourhood vectors on the fly, or store them
enum class M6Ssim { Tasks, Inner, Strips };   // one SSIM per task / loops inside an SSIM / column strips
enum class RfSum { Ordered, Reduction };      // per-tree slots summed in tree order / OpenMP reduction
enum class RfOrder { ImageMajor, TreeMajor }; // forest evaluation order over a batch of images

struct LoadStats;
struct Trace;

struct Config {
    Variant variant = Variant::Baseline;
    Mode mode = Mode::Sequential;
    int threads = 1;             // thread budget of the image being evaluated
    bool tasks = false;          // set by the task graph: loops run as taskloops on the current team
    bool light_parallel = true;  // Graph mode: light modules (M1-M3, M7, M8) also use parallel loops

    bool m1_fused = false;  // filter only the pixels kept by the 2x downsampling
    Sched m2_sched;         // schedule of the loop over DCT window columns
    Sched m7_sched;         // schedule of the loop over NIQE blocks
    M4Level m4_level = M4Level::Lines;
    M5Gather m5_gather = M5Gather::Fused;
    M6Ssim m6_ssim = M6Ssim::Strips;
    RfSum rf_sum = RfSum::Ordered;
    RfOrder rf_order = RfOrder::ImageMajor;

    // optional instrumentation (null = off)
    LoadStats* load_m2 = nullptr;  // per-thread time in the DCT window loop
    LoadStats* load_m6 = nullptr;  // per-thread time in the SSIM work
    LoadStats* load_m7 = nullptr;  // per-thread time in the NIQE block loop
    Trace* trace = nullptr;        // start/end of every module task
    int image_index = 0;

    bool sequential() const { return mode == Mode::Sequential; }
    bool naive() const { return mode == Mode::Naive; }
    // true for the designed parallel versions (Intra and Graph)
    bool designed() const { return mode == Mode::Intra || mode == Mode::Graph; }
};

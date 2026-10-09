// Which sequential version to run.
//
//   Baseline  the optimized sequential baseline (default). Removes redundant work of the
//             official code without changing the output:
//               - GGD shape table built once and searched with binary search (M2, M6)
//               - AGGD shape table built once (M7)
//               - one DCT per 7x7 block instead of five (M2)
//               - separable SSIM window (M6)
//               - unused results skipped: M4 low-pass residual, M5 histogram,
//                 M6 ssim_map / mssim, M7 sharpness map
//   Faithful  does the same work as the official MATLAB code, step by step. Used to measure
//             how much each optimization saves; it is much slower.
#pragma once

enum class Variant { Baseline, Faithful };

// Results that the official code computes but never uses are written here in the faithful
// variant, so that the compiler cannot remove the work.
inline volatile double g_unused_sink = 0;

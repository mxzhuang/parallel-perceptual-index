// Interfaces of the eight modules of the Perceptual Index, split by data dependency.
//
//   M1 spatial pyramid ──┬─> M2 block-DCT statistics ── f1 (18) ─┐
//                        └─> M3 patch SVD ───────────── f3 (75) ─┤
//   M4 steerable pyramid ┬─> M5 divisive normalization ─┐        ├─> M8 forests ─> Ma ─┐
//                        ├─> M6a band / high-pass SSIM ─┼ f2 (45)┘                      ├─> PI
//                        └──────────────> M6b GGD + SSIM┘                               │
//   M7 NIQE ────────────────────────────────────────────────────────────────> NIQE ───┘
//
// Each module mirrors a part of the official MATLAB code (named in its source file). All
// modules take the grey image as uint8 levels stored in doubles (0..255).
#pragma once

#include <string>

#include "config.h"
#include "matrix.h"
#include "timer.h"

struct SpatialPyramid {
    Mat im[3];  // full, half and quarter resolution, values in [0, 1]
};

struct SteerablePyramid {
    Mat hi0;         // high-pass residual
    Mat band[2][6];  // [scale][orientation]
};

struct NormalizedBands {
    Vec subband[12];  // vectorised (column-major), guard band removed, zero mean
    int size_band[12][2];
};

struct NiqeModel {
    Vec mu;   // 36
    Mat cov;  // 36 x 36
    void load(const std::string& path);
};

SpatialPyramid m1_spatial_pyramid(const Mat& gray, const Config& cfg);
// f1: 6 statistics x 3 scales
void m2_block_dct(const SpatialPyramid& sp, double f1[18], const Config& cfg);
// f3: 25 singular values x 3 scales (fewer if a scale has under 25 patches)
Vec m3_patch_svd(const SpatialPyramid& sp, const Config& cfg);
SteerablePyramid m4_steerable_pyramid(const Mat& gray, const Config& cfg);
NormalizedBands m5_divisive_normalization(const SteerablePyramid& pyr, const Config& cfg);
// M6a: structure term between each raw band (resized) and the high-pass residual; needs M4 only
void m6a_band_highpass_ssim(const SteerablePyramid& pyr, double out[12], const Config& cfg);
// M6b: 18 GGD shapes and 15 structure terms between normalized fine-scale orientations
void m6b_normalized_band_stats(const NormalizedBands& nb, double ggd[18], double ssim[15], const Config& cfg);
double m7_niqe(const Mat& gray, const NiqeModel& model, const Config& cfg);

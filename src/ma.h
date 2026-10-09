// Ma et al. (CVIU 2017) no-reference SR quality metric, sequential C++ port of the official
// MATLAB code (sr-metric: feature_all.m, block_dct.m, global_gsm.m, ...).
//
// Module numbering (M1..M8) used throughout the code, timings and README:
//   M1 spatial pyramid   M2 block-DCT statistics (f1)   M3 patch SVD (f3)
//   M4 steerable pyramid M5 divisive normalisation      M6 subband statistics (f2)
//   M8 regression (random_forest.h)
#pragma once

#include "matrix.h"
#include "timer.h"
#include "variant.h"

struct MaFeatures {
    Vec f1;  // 18 = 6 statistics x 3 scales
    Vec f2;  // 45 = 12 + 6 GGD shapes, 12 + 15 SSIM structure terms
    Vec f3;  // 75 = 25 singular values x 3 scales
};

// `gray` holds uint8 grey levels (0..255) of the image, as the official code receives them.
MaFeatures ma_features(const Mat& gray, Variant variant, ModuleTimer* timer);

// Steerable pyramid as used by Ma (2 scales, 6 orientations). Exposed for testing.
struct SteerablePyramid {
    Mat hi0;          // high-pass residual
    Mat band[2][6];   // [scale][orientation]
};
SteerablePyramid build_sf_pyramid(const Mat& im, Variant variant = Variant::Baseline);

// Evaluation of one image: runs modules M1-M8 in the order required by the selected mode.
#pragma once

#include <string>

#include "config.h"
#include "matrix.h"
#include "modules.h"
#include "random_forest.h"
#include "timer.h"

struct MaFeatures {
    Vec f1;  // 18 = 6 statistics x 3 scales
    Vec f2;  // 45 = 12 + 6 GGD shapes, 12 + 15 SSIM structure terms
    Vec f3;  // 75 = 25 singular values x 3 scales
};

struct ImageResult {
    bool valid = true;
    std::string error;  // reason when the image cannot be scored
    MaFeatures feat;
    double s[3] = {0, 0, 0};  // forest outputs
    double ma = 0, niqe = 0, pi = 0;
    ModuleTimer timer;
    double wall = 0;  // seconds from the first to the last module of this image
};

// Smallest side (after preprocessing) for which every module is defined: NIQE needs one
// 96x96 block, and that also guarantees 25 patches at the coarsest scale of M3 and an
// 11x11 SSIM window inside the guard band of M5.
constexpr int kMinImageSide = 96;

// Computes the features and NIQE of one image. With run_forests the three forests and the
// final scores are evaluated as well; otherwise the caller applies the forests later (batch
// evaluation in tree-major order). Never throws: failures are reported in ImageResult.
void evaluate_image(const Mat& gray, const MaModel& ma_model, const NiqeModel& niqe_model, const Config& cfg,
                    bool run_forests, ImageResult& out);

// Ma = linear(s1, s2, s3), PI = ((10 - Ma) + NIQE) / 2.
void fuse_scores(const MaModel& ma_model, ImageResult& r);

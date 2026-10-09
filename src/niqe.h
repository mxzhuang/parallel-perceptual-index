// M7: NIQE (Mittal et al. 2013), sequential C++ port of the official release
// (computequality.m / computefeature.m / estimateaggdparam.m) with 96x96 patches.
#pragma once

#include <string>

#include "matrix.h"
#include "timer.h"
#include "variant.h"

struct NiqeModel {
    Vec mu;   // 36
    Mat cov;  // 36 x 36
    void load(const std::string& path);
};

// `gray`: uint8 grey levels (0..255). Returns the NIQE score.
double niqe_score(const Mat& gray, const NiqeModel& model, Variant variant, ModuleTimer* timer);

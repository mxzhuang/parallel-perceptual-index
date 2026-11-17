#include "modules.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "ggd.h"
#include "matlab_compat.h"
#include "parallel.h"

namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

// ============================================================================
// M1: spatial pyramid (feature_all.m)
// ============================================================================
PI_NOINLINE SpatialPyramid m1_spatial_pyramid(const Mat& gray, const Config& cfg) {
    (void)cfg;
    SpatialPyramid p;
    p.im[0] = gray;
    for (double& v : p.im[0].d) v /= 255.0;  // im2double
    const Mat h = mc::fspecial_gaussian(3, 0.5);
    p.im[1] = mc::downsample2(mc::imfilter(p.im[0], h));
    p.im[2] = mc::downsample2(mc::imfilter(p.im[1], h));
    return p;
}

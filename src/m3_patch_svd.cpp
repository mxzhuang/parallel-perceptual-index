#include "modules.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "ggd.h"
#include "matlab_compat.h"
#include "parallel.h"

namespace {
// ============================================================================
// M3: patch SVD (svd(im2col(im, [5 5], 'distinct')))
// ============================================================================
PI_NOINLINE Vec patch_svd(const Mat& im) { return mc::singular_values_wide(mc::im2col_distinct(im, 5, 5)); }
}  // namespace

Vec m3_patch_svd(const SpatialPyramid& sp, const Config& cfg) {
    (void)cfg;
    Vec f3;
    for (int s = 0; s < 3; ++s) {
        Vec sv = patch_svd(sp.im[s]);
        f3.insert(f3.end(), sv.begin(), sv.end());
    }
    return f3;
}

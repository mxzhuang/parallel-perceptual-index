#include "modules.h"

#include "matlab_compat.h"
#include "parallel.h"

namespace {
// One tap sum of mc::imfilter(a, h) with zero boundary at pixel (r, c): the same taps in
// the same order (kernel columns outer, kernel rows inner), so the value is bit-identical.
inline double filter_at(const Mat& a, const Mat& h, int r, int c) {
    const int R = a.rows, C = a.cols;
    const int cr = (h.rows + 1) / 2 - 1, cc = (h.cols + 1) / 2 - 1;
    double s = 0;
    for (int j = 0; j < h.cols; ++j) {
        const int cj = c + j - cc;
        if (cj < 0 || cj >= C) continue;
        for (int i = 0; i < h.rows; ++i) {
            const int ri = r + i - cr;
            if (ri < 0 || ri >= R) continue;
            s += h(i, j) * a(ri, cj);
        }
    }
    return s;
}

// The loops below hand out contiguous ranges of whole output columns. Columns are
// contiguous in memory, so each chunk writes one private slice of the output and two
// threads can share a cache line only at a chunk border; the input and the kernel are
// read-only. Pixels are independent and the work per column is uniform, hence the static
// block partition.

// gray / 255 (im2double)
Mat to_unit_range(const Mat& gray, const Config& cfg) {
    Mat o(gray.rows, gray.cols);
    par_for(cfg, gray.cols, Sched{}, [&](long clo, long chi) {
        const size_t lo = static_cast<size_t>(clo) * gray.rows, hi = static_cast<size_t>(chi) * gray.rows;
        for (size_t k = lo; k < hi; ++k) o.d[k] = gray.d[k] / 255.0;
    });
    return o;
}

// downsample2(imfilter(a, h)) in one pass: only the pixels that survive the 2x
// downsampling are filtered, a quarter of the work and no full-size temporary.
Mat filter_downsample_fused(const Mat& a, const Mat& h, const Config& cfg) {
    Mat o(a.rows / 2, a.cols / 2);
    par_for(cfg, o.cols, Sched{}, [&](long clo, long chi) {
        for (int c = static_cast<int>(clo); c < chi; ++c)
            for (int r = 0; r < o.rows; ++r) o(r, c) = filter_at(a, h, 2 * r + 1, 2 * c + 1);
    });
    return o;
}

// The same result in two parallel loops: filter every pixel, then keep every second one.
Mat filter_downsample_plain(const Mat& a, const Mat& h, const Config& cfg) {
    Mat f(a.rows, a.cols);
    par_for(cfg, a.cols, Sched{}, [&](long clo, long chi) {
        for (int c = static_cast<int>(clo); c < chi; ++c)
            for (int r = 0; r < a.rows; ++r) f(r, c) = filter_at(a, h, r, c);
    });
    Mat o(a.rows / 2, a.cols / 2);
    par_for(cfg, o.cols, Sched{}, [&](long clo, long chi) {
        for (int c = static_cast<int>(clo); c < chi; ++c)
            for (int r = 0; r < o.rows; ++r) o(r, c) = f(2 * r + 1, 2 * c + 1);
    });
    return o;
}
}  // namespace

// ============================================================================
// M1: spatial pyramid (feature_all.m)
// ============================================================================
PI_NOINLINE SpatialPyramid m1_spatial_pyramid(const Mat& gray, const Config& cfg) {
    SpatialPyramid p;
    const Mat h = mc::fspecial_gaussian(3, 0.5);
    if (!cfg.designed()) {
        p.im[0] = gray;
        for (double& v : p.im[0].d) v /= 255.0;  // im2double
        p.im[1] = mc::downsample2(mc::imfilter(p.im[0], h));
        p.im[2] = mc::downsample2(mc::imfilter(p.im[1], h));
        return p;
    }
    // Each scale is built from the previous one (a dependence carried from scale to scale),
    // so the scales stay in order with a join between them; the parallelism is inside one.
    p.im[0] = to_unit_range(gray, cfg);
    for (int s = 1; s < 3; ++s)
        p.im[s] = cfg.m1_fused ? filter_downsample_fused(p.im[s - 1], h, cfg)
                               : filter_downsample_plain(p.im[s - 1], h, cfg);
    return p;
}

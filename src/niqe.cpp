#include "niqe.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "ggd.h"
#include "matlab_compat.h"

void NiqeModel::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4];
    uint32_t d = 0;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "PINQ", 4) != 0) throw std::runtime_error("bad NIQE model file");
    if (std::fread(&d, 4, 1, f) != 1) throw std::runtime_error("bad NIQE model file");
    mu.resize(d);
    Vec rowmajor(static_cast<size_t>(d) * d);
    if (std::fread(mu.data(), 8, d, f) != d || std::fread(rowmajor.data(), 8, rowmajor.size(), f) != rowmajor.size())
        throw std::runtime_error("NIQE model file truncated");
    std::fclose(f);
    cov = Mat(d, d);
    for (uint32_t r = 0; r < d; ++r)
        for (uint32_t c = 0; c < d; ++c) cov(r, c) = rowmajor[static_cast<size_t>(r) * d + c];
}

namespace {
constexpr int kFeat = 18;

// computefeature.m for one block
// Baseline: AGGD table built once. Faithful: rebuilt on every call, as estimateaggdparam.m does.
void compute_feature(const Mat& block, double* feat, bool faithful) {
    const AggdTable& aggd = aggd_table();
    auto fit = [&](const double* x, size_t n, double& a, double& l, double& r) {
        if (faithful) AggdTable::fit_faithful(x, n, a, l, r);
        else aggd.fit(x, n, a, l, r);
    };
    double alpha, bl, br;
    fit(block.d.data(), block.numel(), alpha, bl, br);
    feat[0] = alpha;
    feat[1] = (bl + br) / 2;
    static const int shifts[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    Vec pair(block.numel());
    for (int s = 0; s < 4; ++s) {
        Mat sh = mc::circshift(block, shifts[s][0], shifts[s][1]);
        for (size_t i = 0; i < pair.size(); ++i) pair[i] = block.d[i] * sh.d[i];
        fit(pair.data(), pair.size(), alpha, bl, br);
        double meanparam = (br - bl) * (std::tgamma(2 / alpha) / std::tgamma(1 / alpha));
        feat[2 + 4 * s + 0] = alpha;
        feat[2 + 4 * s + 1] = meanparam;
        feat[2 + 4 * s + 2] = bl;
        feat[2 + 4 * s + 3] = br;
    }
}
}  // namespace

double niqe_score(const Mat& gray, const NiqeModel& model, Variant variant, ModuleTimer* timer) {
    const bool faithful = variant == Variant::Faithful;
    ModuleTimer::Scope t(timer, 7);
    const int bs = 96;
    const int mb = gray.rows / bs, nbk = gray.cols / bs;
    if (mb == 0 || nbk == 0) return std::nan("");
    Mat im = mc::crop(gray, 0, 0, mb * bs, nbk * bs);

    Mat window = mc::fspecial_gaussian(7, 7.0 / 6.0);
    double total = 0;
    for (int c = 0; c < 7; ++c) {
        double cs = 0;
        for (int r = 0; r < 7; ++r) cs += window(r, c);
        total += cs;
    }
    for (double& v : window.d) v /= total;

    const int nblocks = mb * nbk;
    Mat feat(nblocks, 2 * kFeat);
    for (int scale = 1; scale <= 2; ++scale) {
        Mat sq(im.rows, im.cols);
        for (size_t i = 0; i < im.numel(); ++i) sq.d[i] = im.d[i] * im.d[i];
        const Mat mu = mc::imfilter(im, window, mc::Boundary::Replicate);
        const Mat ex2 = mc::imfilter(sq, window, mc::Boundary::Replicate);
        Mat structdis(im.rows, im.cols), sigma(im.rows, im.cols);
        for (size_t i = 0; i < im.numel(); ++i) {
            sigma.d[i] = std::sqrt(std::fabs(ex2.d[i] - mu.d[i] * mu.d[i]));
            structdis.d[i] = (im.d[i] - mu.d[i]) / (sigma.d[i] + 1);
        }
        // sharpness = blkproc(sigma, [96 96], @computemean): computed by the official code but
        // never used, so only in the faithful variant
        if (faithful && scale == 1) {
            double s = 0;
            for (int j = 0; j < nbk; ++j)
                for (int i = 0; i < mb; ++i) {
                    const Mat blk = mc::crop(sigma, i * bs, j * bs, bs, bs);
                    s += mc::mean(blk.d.data(), blk.numel());
                }
            g_unused_sink = s;
        }
        const int b = bs / scale;
        double f[kFeat];
        for (int j = 0; j < nbk; ++j)
            for (int i = 0; i < mb; ++i) {
                compute_feature(mc::crop(structdis, i * b, j * b, b, b), f, faithful);
                for (int k = 0; k < kFeat; ++k) feat(i + j * mb, (scale - 1) * kFeat + k) = f[k];
            }
        if (scale == 1) im = mc::imresize_scale(im, 0.5);
    }

    // optional: dump the patch features for validation (PI_DEBUG_NIQE=<file>)
    if (const char* dbg = std::getenv("PI_DEBUG_NIQE")) {
        if (FILE* f = std::fopen(dbg, "w")) {
            for (int r = 0; r < nblocks; ++r) {
                for (int c = 0; c < 2 * kFeat; ++c) std::fprintf(f, "%.17g ", feat(r, c));
                std::fprintf(f, "\n");
            }
            std::fclose(f);
        }
    }

    // nanmean / nancov (complete rows)
    const int d = 2 * kFeat;
    Vec mu_dist(d);
    for (int c = 0; c < d; ++c) {
        double s = 0;
        int n = 0;
        for (int r = 0; r < nblocks; ++r)
            if (!std::isnan(feat(r, c))) {
                s += feat(r, c);
                ++n;
            }
        mu_dist[c] = s / n;
    }
    std::vector<int> rows;
    for (int r = 0; r < nblocks; ++r) {
        bool ok = true;
        for (int c = 0; c < d && ok; ++c) ok = !std::isnan(feat(r, c));
        if (ok) rows.push_back(r);
    }
    const int n = static_cast<int>(rows.size());
    Vec colmean(d, 0.0);
    for (int c = 0; c < d; ++c) {
        double s = 0;
        for (int r : rows) s += feat(r, c);
        colmean[c] = s / n;
    }
    Mat cov_dist(d, d);
    for (int j = 0; j < d; ++j)
        for (int i = 0; i < d; ++i) {
            double s = 0;
            for (int r : rows) s += (feat(r, i) - colmean[i]) * (feat(r, j) - colmean[j]);
            cov_dist(i, j) = s / (n - 1);
        }

    Mat avg(d, d);
    for (size_t i = 0; i < avg.numel(); ++i) avg.d[i] = (model.cov.d[i] + cov_dist.d[i]) / 2;
    const Mat inv = mc::pinv_sym(avg);
    Vec diff(d);
    for (int i = 0; i < d; ++i) diff[i] = model.mu[i] - mu_dist[i];
    double q = 0;
    for (int j = 0; j < d; ++j) {
        double t = 0;
        for (int i = 0; i < d; ++i) t += diff[i] * inv(i, j);
        q += t * diff[j];
    }
    return std::sqrt(q);
}

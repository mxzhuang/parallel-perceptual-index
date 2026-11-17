#include "random_forest.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "parallel.h"

namespace {
template <typename T>
void read_into(FILE* f, T* dst, size_t n) {
    if (std::fread(dst, sizeof(T), n, f) != n) throw std::runtime_error("model file truncated");
}

// Leaf value of tree t for feature vector x. Only reads the tree and x.
inline double leaf_value(const RegTree& t, const Vec& x) {
    int k = 0;
    while (t.status[k] != -1) {
        int m = t.var[k] - 1;
        k = (x[m] <= t.value[k]) ? t.left[k] - 1 : t.right[k] - 1;
    }
    return t.value[k];
}

// Images handled per pass of the tree-major order: bounds the [tree][image] slot table
// (2,000 trees x 64 images x 8 bytes = 1 MB) while a tree's nodes are still reused 64 times.
constexpr size_t kBatchImages = 64;
}  // namespace

double RegForest::predict(const Vec& x, const Config& cfg) const {
    const long ntree = static_cast<long>(trees.size());
    if (!cfg.designed()) {
        double sum = 0;
        for (const RegTree& t : trees) sum += leaf_value(t, x);
        return sum / static_cast<double>(trees.size());
    }
    // Trees are independent and of similar depth: one contiguous range of trees per thread
    // (static partition). A whole forest takes about a millisecond, so more chunks than
    // threads would only add scheduling overhead.
    double sum = 0;
    if (cfg.rf_sum == RfSum::Reduction) {
        // What `reduction(+: sum)` does: one partial sum per chunk, merged into the shared
        // total as chunks finish. The grouping follows the thread count and the merge order
        // follows timing, so the last bits of the result may differ from run to run.
        par_for(cfg, ntree, Sched{}, [&](long lo, long hi) {
            double part = 0;
            for (long t = lo; t < hi; ++t) part += leaf_value(trees[t], x);
            #pragma omp atomic
            sum += part;
        });
    } else {
        // One slot per tree, added in tree order after the join: the same additions in the
        // same order as the sequential loop, for any thread count. Ranges are contiguous, so
        // slots written by different threads share a cache line only at chunk borders.
        Vec leaf(trees.size());
        par_for(cfg, ntree, Sched{}, [&](long lo, long hi) {
            for (long t = lo; t < hi; ++t) leaf[t] = leaf_value(trees[t], x);
        });
        for (long t = 0; t < ntree; ++t) sum += leaf[t];
    }
    return sum / static_cast<double>(trees.size());
}

void RegForest::predict_batch(const std::vector<const Vec*>& xs, double* out, const Config& cfg) const {
    if (!cfg.designed() || cfg.rf_order == RfOrder::ImageMajor) {
        for (size_t i = 0; i < xs.size(); ++i) out[i] = predict(*xs[i], cfg);
        return;
    }
    // Tree-major order: a tree is walked for every image of the pass before the next tree is
    // touched, so its nodes are fetched from memory once instead of once per image.
    const long ntree = static_cast<long>(trees.size());
    Vec leaf(trees.size() * std::min(xs.size(), kBatchImages));  // [tree][image]
    for (size_t first = 0; first < xs.size(); first += kBatchImages) {
        const size_t m = std::min(kBatchImages, xs.size() - first);
        // Each chunk writes the table rows of its own contiguous range of trees; the trees
        // and the feature vectors are read-only.
        par_for(cfg, ntree, Sched{}, [&](long lo, long hi) {
            for (long t = lo; t < hi; ++t) {
                double* row = &leaf[static_cast<size_t>(t) * m];
                for (size_t i = 0; i < m; ++i) row[i] = leaf_value(trees[t], *xs[first + i]);
            }
        });
        // Every image accumulates its column in tree order, as image-major order does, so
        // the two orders give identical bits. The table is read row by row (streaming).
        Vec sum(m, 0.0);
        for (long t = 0; t < ntree; ++t) {
            const double* row = &leaf[static_cast<size_t>(t) * m];
            for (size_t i = 0; i < m; ++i) sum[i] += row[i];
        }
        for (size_t i = 0; i < m; ++i) out[first + i] = sum[i] / static_cast<double>(trees.size());
    }
}

void MaModel::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4];
    uint32_t version = 0, nforest = 0;
    read_into(f, magic, 4);
    if (std::memcmp(magic, "PIMA", 4) != 0) throw std::runtime_error("bad Ma model file");
    read_into(f, &version, 1);
    read_into(f, linear, 4);
    read_into(f, &nforest, 1);
    if (nforest != 3) throw std::runtime_error("expected 3 forests");
    for (uint32_t fi = 0; fi < nforest; ++fi) {
        uint32_t ntree = 0;
        read_into(f, &ntree, 1);
        forest[fi].trees.resize(ntree);
        for (RegTree& t : forest[fi].trees) {
            uint32_t n = 0;
            read_into(f, &n, 1);
            t.left.resize(n);
            t.right.resize(n);
            t.status.resize(n);
            t.var.resize(n);
            t.value.resize(n);
            read_into(f, t.left.data(), n);
            read_into(f, t.right.data(), n);
            read_into(f, t.status.data(), n);
            read_into(f, t.var.data(), n);
            read_into(f, t.value.data(), n);
        }
    }
    std::fclose(f);
}

double MaModel::score(const double s[3]) const {
    return linear[0] * 1.0 + linear[1] * s[0] + linear[2] * s[1] + linear[3] * s[2];
}

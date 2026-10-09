#include "random_forest.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
template <typename T>
void read_into(FILE* f, T* dst, size_t n) {
    if (std::fread(dst, sizeof(T), n, f) != n) throw std::runtime_error("model file truncated");
}
}  // namespace

double RegForest::predict(const Vec& x) const {
    double sum = 0;
    for (const RegTree& t : trees) {
        int k = 0;
        while (t.status[k] != -1) {
            int m = t.var[k] - 1;
            k = (x[m] <= t.value[k]) ? t.left[k] - 1 : t.right[k] - 1;
        }
        sum += t.value[k];
    }
    return sum / static_cast<double>(trees.size());
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

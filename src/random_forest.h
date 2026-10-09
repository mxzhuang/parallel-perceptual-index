// M8: regression forests of Ma et al. (Liaw & Wiener randomForest, as compiled in the
// official randomforest-matlab MEX). Prediction = mean of the leaf values over all trees.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "matrix.h"

struct RegTree {
    std::vector<int16_t> left, right;  // 1-based daughter indices
    std::vector<int8_t> status;        // -1 terminal, -3 interior
    std::vector<uint8_t> var;          // 1-based split variable
    std::vector<double> value;         // split threshold (interior) or prediction (terminal)
};

struct RegForest {
    std::vector<RegTree> trees;
    double predict(const Vec& x) const;
};

struct MaModel {
    double linear[4] = {0, 0, 0, 0};
    RegForest forest[3];
    void load(const std::string& path);
    // Ma = [1 s1 s2 s3] * linear
    double score(const double s[3]) const;
};

// Minimal dense matrices stored column-major, matching MATLAB's memory layout so that
// x(:), reshape and block orderings can be ported one-to-one.
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

using cplx = std::complex<double>;

template <typename T>
struct MatT {
    int rows = 0;
    int cols = 0;
    std::vector<T> d;

    MatT() = default;
    MatT(int r, int c, T v = T()) : rows(r), cols(c), d(static_cast<size_t>(r) * c, v) {}

    T& operator()(int r, int c) { return d[static_cast<size_t>(c) * rows + r]; }
    const T& operator()(int r, int c) const { return d[static_cast<size_t>(c) * rows + r]; }
    size_t numel() const { return d.size(); }
    bool empty() const { return d.empty(); }
};

using Mat = MatT<double>;
using CMat = MatT<cplx>;
using Vec = std::vector<double>;

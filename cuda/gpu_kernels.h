// GPU side of the optional CUDA extension: device arrays and the kernels of M5 (divisive
// normalization) and M6 (SSIM structure terms). The interface uses plain C++ types only, so
// the host program needs no CUDA headers.
//
// All arrays are column-major doubles, like the CPU matrices. Double precision and disabled
// multiply-add contraction keep each per-pixel value equal to the CPU value; only the order
// in which partial sums are combined differs (fixed, so results are reproducible).
#pragma once

#include <string>

namespace gpu {

struct DeviceInfo {
    std::string name;
    double memory_mb = 0;
    int capability_major = 0, capability_minor = 0;
};

// Neighbourhood of one band in M5: the 3x3 neighbours of a coefficient, the co-located
// coefficient of the upsampled parent band (handle -1 if the band has none) and of the five
// other orientations at the same scale.
struct Neighbourhood {
    int band = -1;
    int parent = -1;
    int others[5] = {-1, -1, -1, -1, -1};
    int length() const { return 9 + (parent >= 0 ? 1 : 0) + 5; }
};

class Context {
public:
    Context();  // selects device 0 and creates the CUDA context
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    DeviceInfo info() const;

    // ---- device arrays (returned handles stay valid until the context is destroyed) ----
    int upload(const double* data, int rows, int cols);
    int allocate(int rows, int cols);
    void download(int handle, double* out) const;
    int rows(int handle) const;
    int cols(int handle) const;

    // ---- generic reductions --------------------------------------------------------
    // Sum of all elements. Each thread block reduces its elements in shared memory; the
    // block sums are added on the host in block order.
    double sum(int handle);
    void add_scalar(int handle, double value);

    // ---- M5 ---------------------------------------------------------------------------
    // Sum over all interior positions of y y', y being the neighbourhood vector of the
    // position. `out` receives the full symmetric matrix, column-major, length() squared.
    void second_moments(const Neighbourhood& nb, double* out);
    // dst(r, c) = (x - mu) / sqrt(y' P y / divisor) for the interior positions that remain
    // after removing `guard` positions from every border. P is length() x length(),
    // column-major. dst must be (rows - 2 - 2 guard) x (cols - 2 - 2 guard).
    // Returns the number of positions whose local energy is zero.
    long normalize(const Neighbourhood& nb, const double* P, int divisor, double mu, int guard, int dst);

    // ---- M6 ---------------------------------------------------------------------------
    // Sets the 11-tap separable window (constant memory).
    void set_ssim_window(const double* taps11);
    // Sum over the valid region of (2 s12 + C2) / (s1^2 + s2^2 + C2), where the local
    // moments use the separable window. a and b must have the same size.
    double ssim_structure_sum(int a, int b, double c2);

    // Waits for the device to finish (for timing).
    void synchronize();

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace gpu

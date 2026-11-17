// CUDA kernels of the extension. See gpu_kernels.h for the interface.
//
// Thread layout conventions
//   - 1-D kernels use blocks of kBlock threads. Thread g of the grid handles elements
//     g, g + T, g + 2T, ... (T = total threads), so the threads of a warp read consecutive
//     addresses in every step (coalesced access).
//   - The 2-D SSIM kernel uses kTile x kTile blocks with threadIdx.x along rows: arrays are
//     column-major, so x is the contiguous direction.
//   - Every block reduces its partial results with a tree in shared memory, separated by
//     __syncthreads(), and writes one value (or one vector) per block. Blocks cannot
//     synchronize with each other, so the per-block results are combined on the host, in
//     block order, which also makes the sums reproducible.
#include "gpu_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kBlock = 256;          // threads per block of the 1-D kernels (power of two)
constexpr int kMaxLen = 15;          // longest neighbourhood vector
constexpr int kMaxEntries = kMaxLen * (kMaxLen + 1) / 2;
constexpr int kTile = 16;            // output pixels per block side in the SSIM kernel
constexpr int kTaps = 11;            // SSIM window length
constexpr int kHalo = kTaps - 1;
constexpr int kTileIn = kTile + kHalo;

// Read-only parameters shared by all threads: constant memory is cached and broadcast.
__constant__ double c_window[kTaps];
__constant__ double c_P[kMaxLen * kMaxLen];

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}

// Layout of one M5 band on the device.
struct BandView {
    const double* band;
    const double* parent;     // nullptr if the band has no parent
    const double* others[5];
    int rows, cols;           // size of the band
    int len;                  // neighbourhood length
};

// Neighbourhood vector of interior position (r, c): same element order as the CPU code.
__device__ inline void gather(const BandView& v, int r, int c, double* y) {
    int n = 0;
    for (int ny = -1; ny <= 1; ++ny)
        for (int nx = -1; nx <= 1; ++nx) y[n++] = v.band[(r + 1 - ny) + (size_t)(c + 1 - nx) * v.rows];
    const size_t centre = (r + 1) + (size_t)(c + 1) * v.rows;
    if (v.parent) y[n++] = v.parent[centre];
    for (int k = 0; k < 5; ++k) y[n++] = v.others[k][centre];
}

// In-block tree reduction of s[0..kBlock): the result ends in s[0].
__device__ inline void block_reduce(double* s) {
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if ((int)threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    }
    __syncthreads();
}

// partial[block * entries + e] = sum over the block's positions of y_i y_j (upper triangle,
// e enumerates j >= i column by column). Each thread accumulates its positions privately,
// then the block reduces one entry at a time through shared memory.
__global__ void second_moments_kernel(BandView v, long n, long total_threads, double* partial) {
    __shared__ double s[kBlock];
    const int nblv = v.rows - 2;
    const int entries = v.len * (v.len + 1) / 2;
    double acc[kMaxEntries];
    for (int e = 0; e < entries; ++e) acc[e] = 0.0;
    double y[kMaxLen];
    const long g = (long)blockIdx.x * kBlock + threadIdx.x;
    for (long k = g; k < n; k += total_threads) {
        gather(v, (int)(k % nblv), (int)(k / nblv), y);
        int e = 0;
        for (int j = 0; j < v.len; ++j)
            for (int i = 0; i <= j; ++i) acc[e++] += y[i] * y[j];
    }
    for (int e = 0; e < entries; ++e) {
        s[threadIdx.x] = acc[e];
        block_reduce(s);
        if (threadIdx.x == 0) partial[(size_t)blockIdx.x * entries + e] = s[0];
        __syncthreads();  // s is reused for the next entry
    }
}

// One thread per kept position. The quadratic form is evaluated in the order of the CPU
// code: s += (sum_i y_i P(i, j)) * y_j / divisor.
__global__ void normalize_kernel(BandView v, int divisor, double mu, int guard, int out_rows, long n_out, double* dst, int* zero_count) {
    const long idx = (long)blockIdx.x * kBlock + threadIdx.x;
    if (idx >= n_out) return;
    const int r = (int)(idx % out_rows) + guard, c = (int)(idx / out_rows) + guard;
    double y[kMaxLen];
    gather(v, r, c, y);
    double s = 0.0;
    for (int j = 0; j < v.len; ++j) {
        double t = 0.0;
        for (int i = 0; i < v.len; ++i) t += y[i] * c_P[i + j * v.len];
        s += t * y[j] / divisor;
    }
    const double z = sqrt(s);
    if (z == 0.0) atomicAdd(zero_count, 1);  // rare; the host turns it into an error
    dst[idx] = (v.band[(r + 1) + (size_t)(c + 1) * v.rows] - mu) / z;
}

__global__ void sum_kernel(const double* x, long n, long total_threads, double* partial) {
    __shared__ double s[kBlock];
    double acc = 0.0;
    for (long k = (long)blockIdx.x * kBlock + threadIdx.x; k < n; k += total_threads) acc += x[k];
    s[threadIdx.x] = acc;
    block_reduce(s);
    if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

__global__ void add_scalar_kernel(double* x, long n, double value) {
    const long k = (long)blockIdx.x * kBlock + threadIdx.x;
    if (k < n) x[k] += value;
}

// SSIM structure term on one kTile x kTile tile of the valid output region.
//   1. load the tile and its halo of a and b into shared memory (one global read each);
//   2. vertical pass of the five maps (a, b, a*a, b*b, a*b) into shared memory;
//   3. horizontal pass and the per-pixel formula, fused, so no filtered map is ever
//      written back to global memory;
//   4. tree reduction of the tile's pixels to one partial sum.
// Taps are applied in ascending order, vertical pass first, as on the CPU.
__global__ void ssim_kernel(const double* a, const double* b, int rows, int cols, double c2, double* partial) {
    __shared__ double sa[kTileIn * kTileIn];
    __shared__ double sb[kTileIn * kTileIn];
    __shared__ double v[5][kTile * kTileIn];  // vertical pass: kTile rows x kTileIn columns
    __shared__ double s[kTile * kTile];

    const int out_rows = rows - kHalo, out_cols = cols - kHalo;
    const int r0 = blockIdx.x * kTile, c0 = blockIdx.y * kTile;
    const int tid = threadIdx.x + threadIdx.y * kTile;
    const int nthreads = kTile * kTile;

    for (int i = tid; i < kTileIn * kTileIn; i += nthreads) {
        const int gr = r0 + i % kTileIn, gc = c0 + i / kTileIn;
        const bool inside = gr < rows && gc < cols;
        sa[i] = inside ? a[gr + (size_t)gc * rows] : 0.0;
        sb[i] = inside ? b[gr + (size_t)gc * rows] : 0.0;
    }
    __syncthreads();

    for (int i = tid; i < kTile * kTileIn; i += nthreads) {
        const int vr = i % kTile, vc = i / kTile;
        double m1 = 0.0, m2 = 0.0, s11 = 0.0, s22 = 0.0, s12 = 0.0;
        for (int t = 0; t < kTaps; ++t) {
            const double x = sa[(vr + t) + vc * kTileIn], y = sb[(vr + t) + vc * kTileIn];
            const double w = c_window[t];
            m1 += w * x;
            m2 += w * y;
            s11 += w * (x * x);
            s22 += w * (y * y);
            s12 += w * (x * y);
        }
        v[0][i] = m1;
        v[1][i] = m2;
        v[2][i] = s11;
        v[3][i] = s22;
        v[4][i] = s12;
    }
    __syncthreads();

    double cs = 0.0;
    if (r0 + (int)threadIdx.x < out_rows && c0 + (int)threadIdx.y < out_cols) {
        double f[5];
        for (int m = 0; m < 5; ++m) {
            double acc = 0.0;
            for (int t = 0; t < kTaps; ++t) acc += c_window[t] * v[m][threadIdx.x + (threadIdx.y + t) * kTile];
            f[m] = acc;
        }
        const double sigma1_sq = f[2] - f[0] * f[0];
        const double sigma2_sq = f[3] - f[1] * f[1];
        const double sigma12 = f[4] - f[0] * f[1];
        cs = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2);
    }
    s[tid] = cs;
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) s[tid] += s[tid + stride];
    }
    __syncthreads();
    if (tid == 0) partial[blockIdx.x + (size_t)blockIdx.y * gridDim.x] = s[0];
}

struct Array {
    double* p = nullptr;
    int rows = 0, cols = 0;
    size_t n() const { return (size_t)rows * cols; }
};

// Number of blocks for a 1-D kernel over n elements: one thread per element up to a cap, so
// that the per-block results stay a small download.
int blocks_for(long n, int cap) { return (int)std::max(1L, std::min<long>(cap, (n + kBlock - 1) / kBlock)); }

}  // namespace

namespace gpu {

struct Context::Impl {
    std::vector<Array> arrays;
    double* partial = nullptr;  // per-block results of the reducing kernels
    size_t partial_capacity = 0;
    int* zero_count = nullptr;
    std::vector<double> host_partial;

    const Array& at(int h) const {
        if (h < 0 || h >= (int)arrays.size()) throw std::runtime_error("invalid device array handle");
        return arrays[h];
    }
    void reserve_partial(size_t n) {
        if (n <= partial_capacity) return;
        if (partial) check(cudaFree(partial), "cudaFree");
        check(cudaMalloc(&partial, n * sizeof(double)), "cudaMalloc");
        partial_capacity = n;
    }
    BandView view(const Neighbourhood& nb) const {
        const Array& band = at(nb.band);
        BandView v;
        v.band = band.p;
        v.parent = nb.parent >= 0 ? at(nb.parent).p : nullptr;
        for (int k = 0; k < 5; ++k) {
            const Array& o = at(nb.others[k]);
            if (o.rows != band.rows || o.cols != band.cols) throw std::runtime_error("orientation bands differ in size");
            v.others[k] = o.p;
        }
        if (nb.parent >= 0 && (at(nb.parent).rows < band.rows || at(nb.parent).cols < band.cols))
            throw std::runtime_error("parent band smaller than the band");
        if (nb.parent >= 0 && at(nb.parent).rows != band.rows)
            throw std::runtime_error("parent band must have the row count of the band (crop it on the host)");
        v.rows = band.rows;
        v.cols = band.cols;
        v.len = nb.length();
        return v;
    }
};

Context::Context() : impl_(new Impl) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0) throw std::runtime_error("no CUDA device found");
    check(cudaSetDevice(0), "cudaSetDevice");
    check(cudaMalloc(&impl_->zero_count, sizeof(int)), "cudaMalloc");
}

Context::~Context() {
    for (Array& a : impl_->arrays) cudaFree(a.p);
    cudaFree(impl_->partial);
    cudaFree(impl_->zero_count);
    delete impl_;
}

DeviceInfo Context::info() const {
    cudaDeviceProp p;
    check(cudaGetDeviceProperties(&p, 0), "cudaGetDeviceProperties");
    DeviceInfo d;
    d.name = p.name;
    d.memory_mb = p.totalGlobalMem / (1024.0 * 1024.0);
    d.capability_major = p.major;
    d.capability_minor = p.minor;
    return d;
}

int Context::allocate(int rows, int cols) {
    Array a;
    a.rows = rows;
    a.cols = cols;
    check(cudaMalloc(&a.p, a.n() * sizeof(double)), "cudaMalloc");
    impl_->arrays.push_back(a);
    return (int)impl_->arrays.size() - 1;
}

int Context::upload(const double* data, int rows, int cols) {
    const int h = allocate(rows, cols);
    const Array& a = impl_->arrays[h];
    check(cudaMemcpy(a.p, data, a.n() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy (host to device)");
    return h;
}

void Context::download(int handle, double* out) const {
    const Array& a = impl_->at(handle);
    check(cudaMemcpy(out, a.p, a.n() * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy (device to host)");
}

int Context::rows(int handle) const { return impl_->at(handle).rows; }
int Context::cols(int handle) const { return impl_->at(handle).cols; }

double Context::sum(int handle) {
    const Array& a = impl_->at(handle);
    const long n = (long)a.n();
    const int blocks = blocks_for(n, 1024);
    impl_->reserve_partial(blocks);
    sum_kernel<<<blocks, kBlock>>>(a.p, n, (long)blocks * kBlock, impl_->partial);
    check(cudaGetLastError(), "sum_kernel");
    impl_->host_partial.resize(blocks);
    check(cudaMemcpy(impl_->host_partial.data(), impl_->partial, blocks * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
    double total = 0;
    for (int b = 0; b < blocks; ++b) total += impl_->host_partial[b];
    return total;
}

void Context::add_scalar(int handle, double value) {
    const Array& a = impl_->at(handle);
    const long n = (long)a.n();
    add_scalar_kernel<<<(unsigned)((n + kBlock - 1) / kBlock), kBlock>>>(a.p, n, value);
    check(cudaGetLastError(), "add_scalar_kernel");
}

void Context::second_moments(const Neighbourhood& nb, double* out) {
    const BandView v = impl_->view(nb);
    const long n = (long)(v.rows - 2) * (v.cols - 2);
    if (n <= 0) throw std::runtime_error("band too small");
    const int entries = v.len * (v.len + 1) / 2;
    const int blocks = blocks_for(n, 512);
    impl_->reserve_partial((size_t)blocks * entries);
    second_moments_kernel<<<blocks, kBlock>>>(v, n, (long)blocks * kBlock, impl_->partial);
    check(cudaGetLastError(), "second_moments_kernel");
    impl_->host_partial.resize((size_t)blocks * entries);
    check(cudaMemcpy(impl_->host_partial.data(), impl_->partial, impl_->host_partial.size() * sizeof(double), cudaMemcpyDeviceToHost),
          "cudaMemcpy");
    int e = 0;
    for (int j = 0; j < v.len; ++j)
        for (int i = 0; i <= j; ++i, ++e) {
            double total = 0;
            for (int b = 0; b < blocks; ++b) total += impl_->host_partial[(size_t)b * entries + e];
            out[i + j * v.len] = out[j + i * v.len] = total;
        }
}

long Context::normalize(const Neighbourhood& nb, const double* P, int divisor, double mu, int guard, int dst) {
    const BandView v = impl_->view(nb);
    const Array& d = impl_->at(dst);
    const int out_rows = v.rows - 2 - 2 * guard, out_cols = v.cols - 2 - 2 * guard;
    if (out_rows <= 0 || out_cols <= 0 || d.rows != out_rows || d.cols != out_cols)
        throw std::runtime_error("normalize: destination size mismatch");
    check(cudaMemcpyToSymbol(c_P, P, (size_t)v.len * v.len * sizeof(double)), "cudaMemcpyToSymbol");
    check(cudaMemset(impl_->zero_count, 0, sizeof(int)), "cudaMemset");
    const long n_out = (long)out_rows * out_cols;
    normalize_kernel<<<(unsigned)((n_out + kBlock - 1) / kBlock), kBlock>>>(v, divisor, mu, guard, out_rows, n_out, d.p, impl_->zero_count);
    check(cudaGetLastError(), "normalize_kernel");
    int zeros = 0;
    check(cudaMemcpy(&zeros, impl_->zero_count, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
    return zeros;
}

void Context::set_ssim_window(const double* taps11) {
    check(cudaMemcpyToSymbol(c_window, taps11, kTaps * sizeof(double)), "cudaMemcpyToSymbol");
}

double Context::ssim_structure_sum(int a, int b, double c2) {
    const Array& x = impl_->at(a);
    const Array& y = impl_->at(b);
    if (x.rows != y.rows || x.cols != y.cols) throw std::runtime_error("ssim: size mismatch");
    const int out_rows = x.rows - kHalo, out_cols = x.cols - kHalo;
    if (out_rows <= 0 || out_cols <= 0) throw std::runtime_error("ssim: image smaller than the window");
    const dim3 grid((out_rows + kTile - 1) / kTile, (out_cols + kTile - 1) / kTile);
    const dim3 block(kTile, kTile);
    const size_t blocks = (size_t)grid.x * grid.y;
    impl_->reserve_partial(blocks);
    ssim_kernel<<<grid, block>>>(x.p, y.p, x.rows, x.cols, c2, impl_->partial);
    check(cudaGetLastError(), "ssim_kernel");
    impl_->host_partial.resize(blocks);
    check(cudaMemcpy(impl_->host_partial.data(), impl_->partial, blocks * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
    double total = 0;
    for (size_t i = 0; i < blocks; ++i) total += impl_->host_partial[i];
    return total;
}

void Context::synchronize() { check(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

}  // namespace gpu

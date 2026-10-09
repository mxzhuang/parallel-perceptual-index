// Re-implementations of the MATLAB built-in functions used by the official Ma / NIQE code.
// Each function follows MATLAB's definition (indexing, padding, normalisation), so the
// port can be checked feature by feature against the official implementation.
#pragma once

#include <cstdint>

#include "matrix.h"

namespace mc {

// ---- colour / image ---------------------------------------------------------------
// MATLAB rgb2ycbcr on uint8 input, Y channel only (rounded to uint8).
Mat rgb2ycbcr_y(const uint8_t* rgb, int rows, int cols);
// MATLAB rgb2gray on uint8 input (rounded to uint8).
Mat rgb2gray(const uint8_t* rgb, int rows, int cols);
// Remove `w` pixels from every border.
Mat shave(const Mat& a, int w);
// a(r0:r0+nr-1, c0:c0+nc-1) with 0-based start.
Mat crop(const Mat& a, int r0, int c0, int nr, int nc);

// ---- filtering ----------------------------------------------------------------------
// fspecial('gaussian', n, sigma)
Mat fspecial_gaussian(int n, double sigma);
enum class Boundary { Zero, Replicate };
// imfilter(A, h, boundary): correlation, output the same size as A.
Mat imfilter(const Mat& a, const Mat& h, Boundary b = Boundary::Zero);
// filter2(h, A, 'valid'): correlation, valid part only.
Mat filter2_valid(const Mat& h, const Mat& a);
// Same result as filter2_valid for a separable kernel h = k * k' (k a column vector).
Mat filter2_valid_separable(const Vec& k, const Mat& a);
// 1-D kernel k such that fspecial('gaussian', n, sigma) == k * k' (up to rounding).
Vec gaussian_1d(int n, double sigma);

// A(2:2:end, 2:2:end)
Mat downsample2(const Mat& a);
// im2col(A, [m n], 'distinct'), zero padding to a multiple of the block size.
Mat im2col_distinct(const Mat& a, int m, int n);
// imresize(A, scale) and imresize(A, [rows cols]) with MATLAB's bicubic kernel,
// antialiasing when shrinking, symmetric boundary handling.
Mat imresize_scale(const Mat& a, double scale);
Mat imresize_size(const Mat& a, int out_rows, int out_cols);
// circshift(A, [dr dc])
Mat circshift(const Mat& a, int dr, int dc);

// ---- transforms ---------------------------------------------------------------------
// Orthonormal 2-D DCT-II (MATLAB dct2) for an n x n block.
class Dct2 {
public:
    explicit Dct2(int n);
    // out = C * in * C'   (in/out are n*n column-major arrays)
    void apply(const double* in, double* out) const;
    int n() const { return n_; }

private:
    int n_;
    Vec c_;  // n x n, column-major
};

// Complex FFT of arbitrary length (radix-2 for powers of two, Bluestein otherwise).
void fft_inplace(std::vector<cplx>& x, bool inverse);
CMat fft2(const Mat& a);
CMat ifft2(const CMat& a);  // includes the 1/(m*n) scaling
CMat fftshift(const CMat& a);
CMat ifftshift(const CMat& a);

// ---- small dense linear algebra ---------------------------------------------------
// Eigen-decomposition of a symmetric matrix (cyclic Jacobi). vecs columns are eigenvectors.
void eig_sym(const Mat& a, Vec& vals, Mat& vecs);
// pinv for a symmetric matrix, MATLAB tolerance max(size) * norm(A) * eps.
Mat pinv_sym(const Mat& a);
// Singular values (descending) of a matrix with few rows and many columns (svd(A)).
Vec singular_values_wide(const Mat& a);

// ---- statistics (MATLAB semantics, sequential summation) ---------------------------
double mean(const double* x, size_t n);
double var(const double* x, size_t n);  // normalised by n-1
double stdev(const double* x, size_t n);

// MATLAB colon a:d:b
Vec colon(double a, double d, double b);

}  // namespace mc

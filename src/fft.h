// 2-D FFT of arbitrary size, written for this project (no FFT library is used).
// Lengths that are powers of two use radix-2; other lengths use Bluestein's algorithm, which
// turns a length-n transform into a power-of-two convolution.
#pragma once

#include <vector>

#include "config.h"
#include "matrix.h"

namespace mc {

// Complex FFT of arbitrary length, in place.
void fft_inplace(std::vector<cplx>& x, bool inverse);
// MATLAB fft2 / ifft2 (ifft2 includes the 1/(m*n) scaling): every column, then every row.
CMat fft2(const Mat& a, const Config& cfg);
CMat ifft2(const CMat& a, const Config& cfg);
// real(ifft2(a)), for spectra whose inverse transform is used as a real image. Takes the
// spectrum by value and transforms it in place: pass an rvalue to avoid a copy.
Mat ifft2_real(CMat a, const Config& cfg);
CMat fftshift(const CMat& a);
CMat ifftshift(const CMat& a);
// The same permutations, run in parallel in the designed versions (cfg.designed()).
CMat fftshift(const CMat& a, const Config& cfg);
CMat ifftshift(const CMat& a, const Config& cfg);

}  // namespace mc

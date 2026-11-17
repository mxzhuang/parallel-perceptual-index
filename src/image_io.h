// Image decoding (PNG, JPEG, BMP through the single-header stb_image) and the grey-level
// conversion applied before scoring.
#pragma once

#include <string>

#include "matrix.h"

// Returns the grey image as uint8 levels stored in doubles (0..255).
//   single-channel input : used as it is (for example a pre-converted PIRM Y image)
//   RGB input, pirm      : Y channel of MATLAB's rgb2ycbcr, then `shave` pixels removed
//                          from every border (the PIRM 2018 protocol)
//   RGB input, otherwise : MATLAB's rgb2gray, as quality_predict.m does
// Throws std::runtime_error if the file cannot be decoded or is smaller than the border.
Mat load_gray(const std::string& path, bool pirm, int shave);

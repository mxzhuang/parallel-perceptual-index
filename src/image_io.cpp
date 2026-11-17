#include "image_io.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_JPEG
#include "stb_image.h"

#include "matlab_compat.h"

Mat load_gray(const std::string& path, bool pirm, int shave) {
    int w = 0, h = 0, ch = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 0);
    if (!data) throw std::runtime_error("cannot read " + path);
    Mat g;
    if (ch == 1 || ch == 2) {
        g = Mat(h, w);
        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c) g(r, c) = data[(static_cast<size_t>(r) * w + c) * ch];
        stbi_image_free(data);
        return g;
    }
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    for (size_t p = 0; p < static_cast<size_t>(w) * h; ++p)
        for (int k = 0; k < 3; ++k) rgb[p * 3 + k] = data[p * ch + k];
    stbi_image_free(data);
    if (pirm) {
        if (h <= 2 * shave || w <= 2 * shave) throw std::runtime_error("image smaller than the shaved border: " + path);
        return mc::shave(mc::rgb2ycbcr_y(rgb.data(), h, w), shave);
    }
    return mc::rgb2gray(rgb.data(), h, w);
}

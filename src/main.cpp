// pi_eval: Perceptual Index (PIRM 2018) = ((10 - Ma) + NIQE) / 2, sequential C++ baseline.
//
//   pi_eval [options] image1.png [image2.png ...]
//     --models DIR        directory with ma_model.bin and niqe_params.bin (default: models)
//     --pirm              treat RGB inputs like PIRM's calc_scores: Y channel of rgb2ycbcr,
//                         then shave --shave pixels (default 4). Grey inputs are used as is.
//     --shave N           border to remove (only with --pirm)
//     --conv direct|separable   SSIM window filtering in M6 (default: separable)
//     --dump FILE         write all features in the format of reference/run_reference.m
//     --timing            print per-module times (M1..M8)
//
// Without --pirm an RGB input is converted with rgb2gray, as quality_predict.m does.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_JPEG
#include "stb_image.h"

#include "ma.h"
#include "matlab_compat.h"
#include "niqe.h"
#include "random_forest.h"
#include "timer.h"

namespace {
struct Options {
    std::string models = "models";
    bool pirm = false;
    int shave = 4;
    ConvMode conv = ConvMode::Separable;
    std::string dump;
    bool timing = false;
    std::vector<std::string> images;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--models") o.models = next();
        else if (a == "--pirm") o.pirm = true;
        else if (a == "--shave") o.shave = std::atoi(next().c_str());
        else if (a == "--conv") {
            std::string v = next();
            if (v == "direct") o.conv = ConvMode::Direct;
            else if (v == "separable") o.conv = ConvMode::Separable;
            else throw std::runtime_error("--conv must be direct or separable");
        } else if (a == "--dump") o.dump = next();
        else if (a == "--timing") o.timing = true;
        else if (!a.empty() && a[0] == '-') throw std::runtime_error("unknown option " + a);
        else o.images.push_back(a);
    }
    if (o.images.empty()) throw std::runtime_error("no input images");
    return o;
}

Mat load_gray(const std::string& path, const Options& o) {
    int w = 0, h = 0, ch = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 0);
    if (!data) throw std::runtime_error("cannot read " + path);
    Mat g;
    if (ch == 1 || ch == 2) {
        g = Mat(h, w);
        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c) g(r, c) = data[(static_cast<size_t>(r) * w + c) * ch];
        stbi_image_free(data);
        return g;  // already a single channel (e.g. a PIRM Y image)
    }
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    for (size_t p = 0; p < static_cast<size_t>(w) * h; ++p)
        for (int k = 0; k < 3; ++k) rgb[p * 3 + k] = data[p * ch + k];
    stbi_image_free(data);
    if (o.pirm) return mc::shave(mc::rgb2ycbcr_y(rgb.data(), h, w), o.shave);
    return mc::rgb2gray(rgb.data(), h, w);
}

std::string basename(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}

void dump_line(FILE* f, const std::string& name, double ma, double nq, const double s[3], const MaFeatures& feat) {
    std::fprintf(f, "%s %.17g %.17g %.17g %.17g %.17g", name.c_str(), ma, nq, s[0], s[1], s[2]);
    for (const Vec* v : {&feat.f1, &feat.f2, &feat.f3}) {
        std::fprintf(f, " %zu", v->size());
        for (double x : *v) std::fprintf(f, " %.17g", x);
    }
    std::fprintf(f, "\n");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        Options o = parse(argc, argv);
        MaModel ma_model;
        NiqeModel niqe_model;
        ma_model.load(o.models + "/ma_model.bin");
        niqe_model.load(o.models + "/niqe_params.bin");
        FILE* dump = o.dump.empty() ? nullptr : std::fopen(o.dump.c_str(), "w");
        if (!o.dump.empty() && !dump) throw std::runtime_error("cannot write " + o.dump);

        double sum_ma = 0, sum_nq = 0;
        ModuleTimer total;
        std::printf("%-32s %12s %12s %12s\n", "image", "Ma", "NIQE", "PI");
        for (const std::string& path : o.images) {
            Mat gray = load_gray(path, o);
            ModuleTimer timer;
            MaFeatures feat = ma_features(gray, o.conv, &timer);
            double s[3];
            {
                ModuleTimer::Scope t(&timer, 8);
                s[0] = ma_model.forest[0].predict(feat.f1);
                s[1] = ma_model.forest[1].predict(feat.f2);
                s[2] = ma_model.forest[2].predict(feat.f3);
            }
            const double ma = ma_model.score(s);
            const double nq = niqe_score(gray, niqe_model, &timer);
            sum_ma += ma;
            sum_nq += nq;
            std::printf("%-32s %12.6f %12.6f %12.6f\n", basename(path).c_str(), ma, nq, ((10 - ma) + nq) / 2);
            if (o.timing) {
                std::printf("    time [s]:");
                for (int m = 1; m <= 8; ++m) std::printf("  M%d %.3f", m, timer.seconds[m]);
                std::printf("\n");
            }
            for (int m = 1; m <= 8; ++m) total.seconds[m] += timer.seconds[m];
            if (dump) dump_line(dump, basename(path), ma, nq, s, feat);
        }
        const double n = static_cast<double>(o.images.size());
        std::printf("Perceptual Index over %zu image(s): %.6f  (mean Ma %.6f, mean NIQE %.6f)\n", o.images.size(),
                    ((10 - sum_ma / n) + sum_nq / n) / 2, sum_ma / n, sum_nq / n);
        if (o.timing) {
            double all = 0;
            for (int m = 1; m <= 8; ++m) all += total.seconds[m];
            std::printf("Total time per module [s]:");
            for (int m = 1; m <= 8; ++m) std::printf("  M%d %.3f (%.1f%%)", m, total.seconds[m], 100 * total.seconds[m] / all);
            std::printf("\n");
        }
        if (dump) std::fclose(dump);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

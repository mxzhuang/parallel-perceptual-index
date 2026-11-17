// pi_eval_cuda: Perceptual Index with M5 (divisive normalization) and the SSIM terms of M6
// computed on a GPU. Optional extension of pi_eval; the CPU program does not depend on it.
//
//   pi_eval_cuda [options] image1.png [image2.png ...]
//     --models DIR     directory with ma_model.bin and niqe_params.bin (default: models)
//     --pirm           PIRM protocol for RGB inputs (Y channel, 4-pixel shave)
//     --shave N        border to remove (only with --pirm)
//     --threads N      threads for the stages that stay on the CPU (default: all cores)
//     --repeat R       evaluate every image R times and report the median times (default 1)
//     --no-verify      skip the comparison with the CPU implementation of M5 and M6
//     --report FILE    machine-readable times and differences
//
// Division of work
//   CPU (OpenMP)  M1-M3, M4 (the steerable pyramid needs full-image FFTs and no GPU FFT
//                 library may be used), bicubic upsampling of the coarse bands, the 15x15
//                 eigen-decompositions of M5, the GGD fits of M6, M7 and M8.
//   GPU           M5 second moments and normalization, guard-band means, the 27 SSIM
//                 structure terms of M6.
// The pyramid is uploaded once; apart from small per-block partial sums, only the twelve
// normalized bands come back (for the GGD fits, whose table lookup is sensitive to rounding
// and therefore stays in one implementation).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/ggd.h"
#include "../src/image_io.h"
#include "../src/matlab_compat.h"
#include "../src/modules.h"
#include "../src/parallel.h"
#include "../src/random_forest.h"
#include "gpu_kernels.h"

namespace {
struct Options {
    std::string models = "models";
    bool pirm = false;
    int shave = 4;
    int threads = 0;
    int repeat = 1;
    bool verify = true;
    std::string report;
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
        else if (a == "--threads") o.threads = std::atoi(next().c_str());
        else if (a == "--repeat") o.repeat = std::max(1, std::atoi(next().c_str()));
        else if (a == "--no-verify") o.verify = false;
        else if (a == "--report") o.report = next();
        else if (!a.empty() && a[0] == '-') throw std::runtime_error("unknown option " + a);
        else o.images.push_back(a);
    }
    if (o.images.empty()) throw std::runtime_error("no input images");
    if (o.threads <= 0) o.threads = omp_get_max_threads();
    return o;
}

// Wall time of the stages of one evaluation, in seconds.
struct Stages {
    double light = 0;       // M1-M3 on the CPU
    double m4 = 0;          // steerable pyramid on the CPU
    double resize = 0;      // bicubic upsampling of the coarse bands on the CPU
    double upload = 0;      // host to device transfer of the pyramid
    double m5_moments = 0;  // GPU second moments, including the download of block sums
    double m5_host = 0;     // 15x15 eigen-decompositions and centre means on the CPU
    double m5_normalize = 0;
    double m6_ssim = 0;     // GPU, 27 structure terms
    double download = 0;    // normalized bands back to the host
    double m6_ggd = 0;      // GGD fits on the CPU
    double m7 = 0, m8 = 0;
    double gpu_path() const { return resize + upload + m5_moments + m5_host + m5_normalize + m6_ssim + download + m6_ggd; }
    double total() const { return light + m4 + gpu_path() + m7 + m8; }
};

struct Timer {
    double t0 = wall_seconds();
    double lap() {
        const double t = wall_seconds(), d = t - t0;
        t0 = t;
        return d;
    }
};

// M5 and the SSIM terms of M6 on the device; f2 receives the 45 features of M6.
void gpu_m5_m6(const SteerablePyramid& pyr, const Config& cfg, double f2[45], Stages& st) {
    const int R = pyr.hi0.rows, C = pyr.hi0.cols;
    Timer tm;

    // ---- CPU: coarse bands upsampled to the size of the high-pass residual (M6a) --------
    Mat up[6];
    for (int o = 0; o < 6; ++o) up[o] = mc::imresize_size(pyr.band[1][o], R, C, cfg);
    // parent of fine band o (o < 5): imresize(coarse band, 2). It equals up[o] when the
    // fine band is exactly twice the coarse one; otherwise it is computed and cropped.
    const Mat& fine0 = pyr.band[0][0];
    const bool parent_is_up = fine0.rows == R && fine0.cols == C && 2 * pyr.band[1][0].rows == R && 2 * pyr.band[1][0].cols == C;
    Mat parent[5];
    if (!parent_is_up)
        for (int o = 0; o < 5; ++o) {
            const Mat p = mc::imresize_scale(pyr.band[1][o], 2.0, cfg);
            if (p.rows < fine0.rows || p.cols < fine0.cols) throw std::runtime_error("upsampled parent smaller than the band");
            parent[o] = mc::crop(p, 0, 0, fine0.rows, fine0.cols);
        }
    st.resize += tm.lap();

    // ---- one upload of everything the kernels read -----------------------------------
    gpu::Context ctx;
    int h_band[12], h_up[12], h_parent[5], h_norm[12];
    for (int i = 0; i < 12; ++i) {
        const Mat& b = pyr.band[i / 6][i % 6];
        h_band[i] = ctx.upload(b.d.data(), b.rows, b.cols);
    }
    const int h_hi0 = ctx.upload(pyr.hi0.d.data(), R, C);
    for (int i = 0; i < 6; ++i) {
        // a fine band already has the size of the residual: bicubic resizing with scale 1
        // returns the input unchanged
        h_up[i] = (fine0.rows == R && fine0.cols == C) ? h_band[i] : -1;
        if (h_up[i] < 0) {
            const Mat u = mc::imresize_size(pyr.band[0][i], R, C, cfg);
            h_up[i] = ctx.upload(u.d.data(), R, C);
        }
        h_up[6 + i] = ctx.upload(up[i].d.data(), R, C);
    }
    for (int o = 0; o < 5; ++o) h_parent[o] = parent_is_up ? h_up[6 + o] : ctx.upload(parent[o].d.data(), parent[o].rows, parent[o].cols);
    const Vec window = mc::gaussian_1d(11, 1.5);
    ctx.set_ssim_window(window.data());
    ctx.synchronize();
    st.upload += tm.lap();

    // ---- M5: per band, second moments (GPU), 15x15 algebra (CPU), normalization (GPU) ----
    int size_band[12][2];
    for (int p = 0; p < 12; ++p) {
        const int scale = p / 6, orien = p % 6;
        const Mat& aux = pyr.band[scale][orien];
        gpu::Neighbourhood nb;
        nb.band = h_band[p];
        nb.parent = (scale == 0 && orien < 5) ? h_parent[orien] : -1;  // as the reference code: bands 2..6 only
        for (int k = 0, n = 0; k < 6; ++k)
            if (k != orien) nb.others[n++] = h_band[scale * 6 + k];
        const int len = nb.length(), divisor = len - 5;
        const int nblv = aux.rows - 2, nblh = aux.cols - 2;
        const long nexp = static_cast<long>(nblv) * nblh;

        Mat Cx(len, len);
        ctx.second_moments(nb, Cx.d.data());
        st.m5_moments += tm.lap();

        for (double& v : Cx.d) v /= static_cast<double>(nexp);
        // clip negative eigenvalues, keep the trace, rebuild, invert (as on the CPU)
        Vec lam;
        Mat Q;
        mc::eig_sym(Cx, lam, Q);
        double total = 0, pos = 0;
        for (double l : lam) {
            total += l;
            pos += l * (l > 0);
        }
        const double denom = pos + (pos == 0 ? 1.0 : 0.0);
        for (double& l : lam) l = (l * (l > 0)) * total / denom;
        Mat Cc(len, len);
        for (int j = 0; j < len; ++j)
            for (int i = 0; i < len; ++i) {
                double s = 0;
                for (int k = 0; k < len; ++k) s += Q(i, k) * lam[k] * Q(j, k);
                Cc(i, j) = s;
            }
        const Mat P = mc::pinv_sym(Cc);
        const double mu = det_sum(cfg, nexp, [&](long k) { return aux(static_cast<int>(k % nblv) + 1, static_cast<int>(k / nblv) + 1); }) /
                          static_cast<double>(nexp);
        st.m5_host += tm.lap();

        const int gb = 16 >> scale;
        size_band[p][0] = nblv - 2 * gb;
        size_band[p][1] = nblh - 2 * gb;
        h_norm[p] = ctx.allocate(size_band[p][0], size_band[p][1]);
        if (ctx.normalize(nb, P.d.data(), divisor, mu, gb, h_norm[p]) != 0)
            throw std::runtime_error("zero local energy in divisive normalisation");
        // remove the mean of what is left after the guard band
        const double m2 = ctx.sum(h_norm[p]) / (static_cast<double>(size_band[p][0]) * size_band[p][1]);
        ctx.add_scalar(h_norm[p], -m2);
        ctx.synchronize();
        st.m5_normalize += tm.lap();
    }

    // ---- M6: 27 SSIM structure terms on the GPU -----------------------------------------
    const double c2 = (0.03 * 255) * (0.03 * 255);
    for (int i = 0; i < 12; ++i) f2[18 + i] = ctx.ssim_structure_sum(h_up[i], h_hi0, c2) / (static_cast<double>(R - 10) * (C - 10));
    int k = 0;
    for (int i = 0; i < 6; ++i)
        for (int j = i + 1; j < 6; ++j)
            f2[30 + k++] = ctx.ssim_structure_sum(h_norm[i], h_norm[j], c2) /
                           (static_cast<double>(size_band[i][0] - 10) * (size_band[i][1] - 10));
    ctx.synchronize();
    st.m6_ssim += tm.lap();

    // ---- GGD fits on the CPU from the downloaded normalized bands -----------------------
    Vec sub[12];
    for (int p = 0; p < 12; ++p) {
        sub[p].resize(static_cast<size_t>(size_band[p][0]) * size_band[p][1]);
        ctx.download(h_norm[p], sub[p].data());
    }
    st.download += tm.lap();
    const GgdTable& ggd = ggd_table();
    #pragma omp parallel for num_threads(cfg.threads) schedule(dynamic, 1)
    for (int i = 0; i < 18; ++i) {
        if (i < 12) {
            f2[i] = ggd.fit(sub[i].data(), sub[i].size());
        } else {
            Vec both = sub[i - 12];
            both.insert(both.end(), sub[i - 6].begin(), sub[i - 6].end());
            f2[i] = ggd.fit(both.data(), both.size());
        }
    }
    st.m6_ggd += tm.lap();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::string basename(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        omp_set_dynamic(0);
        MaModel ma_model;
        NiqeModel niqe_model;
        ma_model.load(o.models + "/ma_model.bin");
        niqe_model.load(o.models + "/niqe_params.bin");

        Config cfg;
        cfg.mode = Mode::Intra;
        cfg.threads = o.threads;
        cfg.m1_fused = true;

        gpu::DeviceInfo dev;
        {
            gpu::Context warm;  // creating the first context is slow; keep it out of the timings
            dev = warm.info();
        }
        std::printf("GPU: %s, %.0f MB, compute capability %d.%d; CPU threads: %d\n", dev.name.c_str(), dev.memory_mb, dev.capability_major,
                    dev.capability_minor, o.threads);

        FILE* rep = o.report.empty() ? nullptr : std::fopen(o.report.c_str(), "w");
        if (!o.report.empty() && !rep) throw std::runtime_error("cannot write " + o.report);
        if (rep) std::fprintf(rep, "config gpu=\"%s\" gpu_memory_mb=%.0f threads=%d repeat=%d\n", dev.name.c_str(), dev.memory_mb, o.threads, o.repeat);

        std::printf("%-32s %12s %12s %12s\n", "image", "Ma", "NIQE", "PI");
        for (const std::string& path : o.images) {
            const Mat gray = load_gray(path, o.pirm, o.shave);
            if (gray.rows < 96 || gray.cols < 96) {
                std::printf("%-32s invalid: image smaller than 96 pixels after preprocessing\n", basename(path).c_str());
                continue;
            }
            std::vector<Stages> runs(o.repeat);
            std::vector<double> cpu_m5(o.repeat, 0.0), cpu_m6(o.repeat, 0.0);
            Vec f1(18), f2(45), f3, f2_cpu(45);
            double s[3] = {0, 0, 0}, niqe = 0;
            for (int rep_i = 0; rep_i < o.repeat; ++rep_i) {
                Stages& st = runs[rep_i];
                Timer tm;
                const SpatialPyramid sp = m1_spatial_pyramid(gray, cfg);
                m2_block_dct(sp, f1.data(), cfg);
                f3 = m3_patch_svd(sp, cfg);
                st.light = tm.lap();
                const SteerablePyramid pyr = m4_steerable_pyramid(gray, cfg);
                st.m4 = tm.lap();
                gpu_m5_m6(pyr, cfg, f2.data(), st);
                tm.lap();
                niqe = m7_niqe(gray, niqe_model, cfg);
                st.m7 = tm.lap();
                s[0] = ma_model.forest[0].predict(f1, cfg);
                s[1] = ma_model.forest[1].predict(f2, cfg);
                s[2] = ma_model.forest[2].predict(f3, cfg);
                st.m8 = tm.lap();
                if (o.verify) {  // the same two modules on the CPU, from the same pyramid
                    const NormalizedBands nb = m5_divisive_normalization(pyr, cfg);
                    cpu_m5[rep_i] = tm.lap();
                    m6b_normalized_band_stats(nb, &f2_cpu[0], &f2_cpu[30], cfg);
                    m6a_band_highpass_ssim(pyr, &f2_cpu[18], cfg);
                    cpu_m6[rep_i] = tm.lap();
                }
            }
            const double ma = ma_model.score(s);
            const std::string name = basename(path);
            std::printf("%-32s %12.6f %12.6f %12.6f\n", name.c_str(), ma, niqe, ((10 - ma) + niqe) / 2);

            auto med = [&](double Stages::*field) {
                std::vector<double> v;
                for (const Stages& r : runs) v.push_back(r.*field);
                return median(v);
            };
            std::vector<double> gpu_path, total;
            for (const Stages& r : runs) {
                gpu_path.push_back(r.gpu_path());
                total.push_back(r.total());
            }
            std::printf("    CPU stages [s]: M1-M3 %.3f  M4 %.3f  M7 %.3f  M8 %.3f\n", med(&Stages::light), med(&Stages::m4), med(&Stages::m7),
                        med(&Stages::m8));
            std::printf("    M5+M6 with GPU [s]: %.3f = resize %.3f + upload %.3f + moments %.3f + host algebra %.3f + normalize %.3f"
                        " + SSIM %.3f + download %.3f + GGD fits %.3f\n",
                        median(gpu_path), med(&Stages::resize), med(&Stages::upload), med(&Stages::m5_moments), med(&Stages::m5_host),
                        med(&Stages::m5_normalize), med(&Stages::m6_ssim), med(&Stages::download), med(&Stages::m6_ggd));
            std::printf("    end to end %.3f s\n", median(total));

            double d_ggd = 0, d_ssim = 0;
            int ggd_mismatch = 0;
            if (o.verify) {
                for (int i = 0; i < 45; ++i) {
                    const double d = std::fabs(f2[i] - f2_cpu[i]) / std::max(std::fabs(f2_cpu[i]), 1e-12);
                    if (i < 18) {
                        d_ggd = std::max(d_ggd, d);
                        ggd_mismatch += f2[i] != f2_cpu[i];
                    } else {
                        d_ssim = std::max(d_ssim, d);
                    }
                }
                std::printf("    M5+M6 on the CPU with %d thread(s) [s]: %.3f (M5 %.3f, M6 %.3f)\n", o.threads, median(cpu_m5) + median(cpu_m6),
                            median(cpu_m5), median(cpu_m6));
                std::printf("    GPU vs CPU features: SSIM max relative difference %.2e; GGD shapes differing %d of 18 (max relative %.2e)\n",
                            d_ssim, ggd_mismatch, d_ggd);
            }
            if (rep) {
                std::fprintf(rep,
                             "image name=%s rows=%d cols=%d ma=%.17g niqe=%.17g pi=%.17g light=%.6f m4=%.6f resize=%.6f upload=%.6f "
                             "m5_moments=%.6f m5_host=%.6f m5_normalize=%.6f m6_ssim=%.6f download=%.6f m6_ggd=%.6f m7=%.6f m8=%.6f "
                             "gpu_path=%.6f total=%.6f cpu_m5=%.6f cpu_m6=%.6f ssim_max_rel=%.3e ggd_mismatch=%d ggd_max_rel=%.3e\n",
                             name.c_str(), gray.rows, gray.cols, ma, niqe, ((10 - ma) + niqe) / 2, med(&Stages::light), med(&Stages::m4),
                             med(&Stages::resize), med(&Stages::upload), med(&Stages::m5_moments), med(&Stages::m5_host),
                             med(&Stages::m5_normalize), med(&Stages::m6_ssim), med(&Stages::download), med(&Stages::m6_ggd), med(&Stages::m7),
                             med(&Stages::m8), median(gpu_path), median(total), median(cpu_m5), median(cpu_m6), d_ssim, ggd_mismatch, d_ggd);
            }
        }
        if (rep) std::fclose(rep);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

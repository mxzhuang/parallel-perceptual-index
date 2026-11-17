#include "pipeline.h"

#include <cmath>
#include <exception>
#include <stdexcept>

#include "parallel.h"

namespace {
// Times one module, adds it to the per-module totals and records it in the trace.
struct ModuleScope {
    const Config& cfg;
    ModuleTimer& timer;
    int module;
    const char* name;
    double t0;
    ModuleScope(const Config& c, ModuleTimer& t, int m, const char* n) : cfg(c), timer(t), module(m), name(n), t0(wall_seconds()) {}
    ~ModuleScope() {
        const double t1 = wall_seconds();
        // M6a and M6b may finish on different threads and both add to seconds[6]
        #pragma omp atomic
        timer.seconds[module] += t1 - t0;
        if (cfg.trace) cfg.trace->add(cfg.image_index, name, t0, t1);
    }
};

void check_features(const MaFeatures& f) {
    if (f.f1.size() != 18 || f.f2.size() != 45 || f.f3.size() != 75)
        throw std::runtime_error("unexpected feature count (image too small for every scale)");
    for (const Vec* v : {&f.f1, &f.f2, &f.f3})
        for (double x : *v)
            if (std::isnan(x)) throw std::runtime_error("NaN feature");
}

// Modules one after another, in the order of the sequential baseline.
void run_in_sequence(const Mat& gray, const MaModel& ma_model, const NiqeModel& niqe_model, const Config& cfg,
                     bool run_forests, ImageResult& r) {
    MaFeatures& f = r.feat;
    f.f1.assign(18, 0.0);
    f.f2.assign(45, 0.0);
    SpatialPyramid sp;
    {
        ModuleScope t(cfg, r.timer, 1, "M1");
        sp = m1_spatial_pyramid(gray, cfg);
    }
    {
        ModuleScope t(cfg, r.timer, 2, "M2");
        m2_block_dct(sp, f.f1.data(), cfg);
    }
    {
        ModuleScope t(cfg, r.timer, 3, "M3");
        f.f3 = m3_patch_svd(sp, cfg);
    }
    SteerablePyramid pyr;
    {
        ModuleScope t(cfg, r.timer, 4, "M4");
        pyr = m4_steerable_pyramid(gray, cfg);
    }
    NormalizedBands nb;
    {
        ModuleScope t(cfg, r.timer, 5, "M5");
        nb = m5_divisive_normalization(pyr, cfg);
    }
    {
        ModuleScope t(cfg, r.timer, 6, "M6");
        m6b_normalized_band_stats(nb, &f.f2[0], &f.f2[30], cfg);
        m6a_band_highpass_ssim(pyr, &f.f2[18], cfg);
    }
    check_features(f);
    if (run_forests) {
        ModuleScope t(cfg, r.timer, 8, "M8");
        r.s[0] = ma_model.forest[0].predict(f.f1, cfg);
        r.s[1] = ma_model.forest[1].predict(f.f2, cfg);
        r.s[2] = ma_model.forest[2].predict(f.f3, cfg);
    }
    {
        ModuleScope t(cfg, r.timer, 7, "M7");
        r.niqe = m7_niqe(gray, niqe_model, cfg);
    }
}

#ifdef _OPENMP
// Modules as tasks ordered by their data dependencies. One team of cfg.threads threads runs
// everything: module tasks, and the taskloops that the modules create inside them, so the
// thread budget holds without any per-module thread count.
//
//   M4 -> M5 -> M6b -> forest 2      (critical path)
//   M4 -> M6a ---------^             (overlaps with M5)
//   M1 -> M2 -> forest 1,  M1 -> M3 -> forest 3,  M7 on its own
//
// Each task writes only its own outputs, and a task that reads an object is created after
// the task that writes it and names it in depend(in:), so there is no data race on the
// intermediate results. No task holds a lock while it waits for another task.
void run_task_graph(const Mat& gray, const MaModel& ma_model, const NiqeModel& niqe_model, const Config& cfg_in,
                    bool run_forests, ImageResult& r) {
    Config cfg = cfg_in;
    cfg.tasks = true;
    Config light = cfg;  // M1-M3, M7, M8
    if (!cfg.light_parallel) light.threads = 1;

    MaFeatures& f = r.feat;
    f.f1.assign(18, 0.0);
    f.f2.assign(45, 0.0);
    SpatialPyramid sp;
    SteerablePyramid pyr;
    NormalizedBands nb;
    // tokens that carry the dependencies (depend clauses need list items, not struct members)
    char d_sp = 0, d_pyr = 0, d_nb = 0, d_f1 = 0, d_f3 = 0, d_f2a = 0, d_f2b = 0;
    (void)d_sp, (void)d_pyr, (void)d_nb, (void)d_f1, (void)d_f3, (void)d_f2a, (void)d_f2b;
    detail::ErrorSlot err;
    ModuleTimer& timer = r.timer;
    double niqe = 0;
    double* s = r.s;

    #pragma omp parallel num_threads(cfg.threads) default(shared)
    #pragma omp single
    {
        // the chain M4 -> M5 -> M6 holds over 90% of the work, so it is created first
        #pragma omp task depend(out: d_pyr) default(shared)
        try {
            ModuleScope t(cfg, timer, 4, "M4");
            pyr = m4_steerable_pyramid(gray, cfg);
        } catch (...) { err.capture(); }

        #pragma omp task depend(in: d_pyr) depend(out: d_nb) default(shared)
        try {
            if (!err.first) {
                ModuleScope t(cfg, timer, 5, "M5");
                nb = m5_divisive_normalization(pyr, cfg);
            }
        } catch (...) { err.capture(); }

        #pragma omp task depend(in: d_pyr) depend(out: d_f2a) default(shared)
        try {
            if (!err.first) {
                ModuleScope t(cfg, timer, 6, "M6a");
                m6a_band_highpass_ssim(pyr, &f.f2[18], cfg);
            }
        } catch (...) { err.capture(); }

        #pragma omp task depend(in: d_nb) depend(out: d_f2b) default(shared)
        try {
            if (!err.first) {
                ModuleScope t(cfg, timer, 6, "M6b");
                m6b_normalized_band_stats(nb, &f.f2[0], &f.f2[30], cfg);
            }
        } catch (...) { err.capture(); }

        #pragma omp task depend(out: d_sp) default(shared)
        try {
            ModuleScope t(cfg, timer, 1, "M1");
            sp = m1_spatial_pyramid(gray, light);
        } catch (...) { err.capture(); }

        #pragma omp task depend(in: d_sp) depend(out: d_f1) default(shared)
        try {
            if (!err.first) {
                ModuleScope t(cfg, timer, 2, "M2");
                m2_block_dct(sp, f.f1.data(), light);
            }
        } catch (...) { err.capture(); }

        #pragma omp task depend(in: d_sp) depend(out: d_f3) default(shared)
        try {
            if (!err.first) {
                ModuleScope t(cfg, timer, 3, "M3");
                f.f3 = m3_patch_svd(sp, light);
            }
        } catch (...) { err.capture(); }

        #pragma omp task default(shared)
        try {
            ModuleScope t(cfg, timer, 7, "M7");
            niqe = m7_niqe(gray, niqe_model, light);
        } catch (...) { err.capture(); }

        if (run_forests) {
            // each forest starts as soon as its own feature vector is complete
            #pragma omp task depend(in: d_f1) default(shared)
            try {
                if (!err.first && f.f1.size() == 18) {
                    ModuleScope t(cfg, timer, 8, "M8.1");
                    s[0] = ma_model.forest[0].predict(f.f1, light);
                }
            } catch (...) { err.capture(); }

            #pragma omp task depend(in: d_f3) default(shared)
            try {
                if (!err.first && f.f3.size() == 75) {
                    ModuleScope t(cfg, timer, 8, "M8.3");
                    s[2] = ma_model.forest[2].predict(f.f3, light);
                }
            } catch (...) { err.capture(); }

            #pragma omp task depend(in: d_f2a, d_f2b) default(shared)
            try {
                if (!err.first) {
                    ModuleScope t(cfg, timer, 8, "M8.2");
                    s[1] = ma_model.forest[1].predict(f.f2, light);
                }
            } catch (...) { err.capture(); }
        }
        // the implicit barrier at the end of the parallel region waits for every task
    }
    err.rethrow();
    r.niqe = niqe;
    check_features(f);
}
#endif
}  // namespace

void fuse_scores(const MaModel& ma_model, ImageResult& r) {
    r.ma = ma_model.score(r.s);
    r.pi = ((10 - r.ma) + r.niqe) / 2;
    if (std::isnan(r.pi)) {
        r.valid = false;
        r.error = "score is NaN";
    }
}

void evaluate_image(const Mat& gray, const MaModel& ma_model, const NiqeModel& niqe_model, const Config& cfg,
                    bool run_forests, ImageResult& r) {
    const double t0 = wall_seconds();
    try {
        if (gray.rows < kMinImageSide || gray.cols < kMinImageSide)
            throw std::runtime_error("image smaller than " + std::to_string(kMinImageSide) + " pixels after preprocessing");
#ifdef _OPENMP
        if (cfg.mode == Mode::Graph && cfg.threads > 1)
            run_task_graph(gray, ma_model, niqe_model, cfg, run_forests, r);
        else
#endif
            run_in_sequence(gray, ma_model, niqe_model, cfg, run_forests, r);
        if (run_forests) fuse_scores(ma_model, r);
    } catch (const std::exception& e) {
        r.valid = false;
        r.error = e.what();
    }
    r.wall = wall_seconds() - t0;
}

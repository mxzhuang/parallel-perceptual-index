// pi_eval: Perceptual Index (PIRM 2018) = ((10 - Ma) + NIQE) / 2.
//
//   pi_eval [options] image1.png [image2.png ...]
//
// Input
//   --models DIR         directory with ma_model.bin and niqe_params.bin (default: models)
//   --pirm               PIRM protocol for RGB inputs: Y channel of rgb2ycbcr, then shave
//                        --shave pixels (default 4). Grey inputs are used as they are.
//   --shave N            border to remove (only with --pirm)
//   --list FILE          read image paths from FILE, one per line
//
// Version
//   --mode M             seq    optimized sequential baseline (default)
//                        naive  baseline with `parallel for` on its hottest loops
//                        intra  modules in sequence, each parallel inside
//                        graph  modules as dependency-ordered tasks on one thread team
//   --faithful           repeat the redundant work of the official MATLAB code (seq only)
//   --threads N          total thread budget (default: all cores; 1 for seq)
//   --image-workers K    images evaluated at the same time (default 1); each image gets
//                        max(1, N / K) threads. K = N scores one image per thread.
//   --prefetch           decode upcoming images on a helper thread
//
// Design switches (for the ablation experiments)
//   --light serial|parallel        graph: light modules as single tasks or with parallel loops
//   --m1 plain|fused               M1: filter every pixel, or only those kept by downsampling
//   --m2-sched KIND[,CHUNK]        M2 window loop: static | dynamic | guided
//   --m7-sched KIND[,CHUNK]        M7 block loop:  static | dynamic | guided
//   --m4-level lines|bands         M4: parallel 1-D FFTs inside each 2-D FFT, or whole bands
//   --m5-gather fused|materialized M5: neighbourhood vectors on the fly, or stored
//   --m6-ssim tasks|inner|strips   M6: one SSIM per task, loops inside an SSIM, or strips
//   --rf-sum ordered|reduction     M8: per-tree slots summed in tree order, or a reduction
//   --rf-order image|tree          M8: forests per image, or tree by tree over all images
//
// Output
//   --timing             per-module times (M1..M8) and I/O time
//   --balance            per-thread busy time of the M2 window, M6 SSIM and M7 block loops
//   --trace FILE         start, end and thread of every module task
//   --report FILE        machine-readable results (scores, times, throughput, peak memory)
//   --dump FILE          all features, in the format of reference/run_reference.m
//   --warmup             evaluate the first image once before timing starts
//   --quiet              do not print the per-image table
//   --help               print this text
//
// Without --pirm an RGB input is converted with rgb2gray, as quality_predict.m does.
// Exit status: 0 on success, 1 on a usage or I/O error, 2 if any image could not be scored.

#include <sys/resource.h>

#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "image_io.h"
#include "parallel.h"
#include "pipeline.h"

namespace {
const char* const kUsage =
    R"(Perceptual Index (PIRM 2018) = ((10 - Ma) + NIQE) / 2

  pi_eval [options] image1.png [image2.png ...]

Input
  --models DIR         directory with ma_model.bin and niqe_params.bin (default: models)
  --pirm               PIRM protocol for RGB inputs: Y channel of rgb2ycbcr, then shave
                       --shave pixels (default 4). Grey inputs are used as they are.
  --shave N            border to remove (only with --pirm)
  --list FILE          read image paths from FILE, one per line

Version
  --mode M             seq    optimized sequential baseline (default)
                       naive  baseline with `parallel for` on its hottest loops
                       intra  modules in sequence, each parallel inside
                       graph  modules as dependency-ordered tasks on one thread team
  --faithful           repeat the redundant work of the official MATLAB code (seq only)
  --threads N          total thread budget (default: all cores; 1 for seq)
  --image-workers K    images evaluated at the same time (default 1); each image gets
                       max(1, N / K) threads. K = N scores one image per thread.
  --prefetch           decode upcoming images on a helper thread

Design switches (for the ablation experiments)
  --light serial|parallel        graph: light modules as single tasks or with parallel loops
  --m1 plain|fused               M1: filter every pixel, or only those kept by downsampling
  --m2-sched KIND[,CHUNK]        M2 window loop: static | dynamic | guided
  --m7-sched KIND[,CHUNK]        M7 block loop:  static | dynamic | guided
  --m4-level lines|bands         M4: parallel 1-D FFTs inside each 2-D FFT, or whole bands
  --m5-gather fused|materialized M5: neighbourhood vectors on the fly, or stored
  --m6-ssim tasks|inner|strips   M6: one SSIM per task, loops inside an SSIM, or strips
  --rf-sum ordered|reduction     M8: per-tree slots summed in tree order, or a reduction
  --rf-order image|tree          M8: forests per image, or tree by tree over all images

Output
  --timing             per-module times (M1..M8) and I/O time
  --balance            per-thread busy time of the M2 window, M6 SSIM and M7 block loops
  --trace FILE         start, end and thread of every module task
  --report FILE        machine-readable results (scores, times, throughput, peak memory)
  --dump FILE          all features, in the format of reference/run_reference.m
  --warmup             evaluate the first image once before timing starts
  --quiet              do not print the per-image table

Without --pirm an RGB input is converted with rgb2gray, as quality_predict.m does.
Exit status: 0 on success, 1 on a usage or I/O error, 2 if any image could not be scored.
)";

struct Options {
    std::string models = "models";
    bool pirm = false;
    int shave = 4;
    Config cfg;
    int threads = 0;  // 0 = not given
    int image_workers = 1;
    bool prefetch = false;
    bool m1_given = false;
    std::string dump, trace, report;
    bool timing = false, balance = false, warmup = false, quiet = false;
    std::vector<std::string> images;
};

Sched parse_sched(const std::string& v) {
    Sched s;
    const size_t comma = v.find(',');
    const std::string kind = v.substr(0, comma);
    if (kind == "static") s.kind = SchedKind::Static;
    else if (kind == "dynamic") s.kind = SchedKind::Dynamic;
    else if (kind == "guided") s.kind = SchedKind::Guided;
    else throw std::runtime_error("unknown schedule " + kind);
    if (comma != std::string::npos) s.chunk = std::atol(v.substr(comma + 1).c_str());
    return s;
}

const char* sched_name(Sched s) {
    return s.kind == SchedKind::Static ? "static" : s.kind == SchedKind::Dynamic ? "dynamic" : "guided";
}

const char* mode_name(Mode m) {
    return m == Mode::Sequential ? "seq" : m == Mode::Naive ? "naive" : m == Mode::Intra ? "intra" : "graph";
}

// value of a two-way (or three-way) switch
template <typename T>
T choose(const std::string& opt, const std::string& v, std::initializer_list<std::pair<const char*, T>> table) {
    for (const auto& e : table)
        if (v == e.first) return e.second;
    throw std::runtime_error("bad value '" + v + "' for " + opt);
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            std::fputs(kUsage, stdout);
            std::exit(0);
        }
        else if (a == "--models") o.models = next();
        else if (a == "--pirm") o.pirm = true;
        else if (a == "--shave") o.shave = std::atoi(next().c_str());
        else if (a == "--list") {
            std::ifstream in(next());
            if (!in) throw std::runtime_error("cannot read image list");
            for (std::string line; std::getline(in, line);)
                if (!line.empty()) o.images.push_back(line);
        }
        else if (a == "--mode")
            o.cfg.mode = choose<Mode>(a, next(), {{"seq", Mode::Sequential}, {"naive", Mode::Naive}, {"intra", Mode::Intra}, {"graph", Mode::Graph}});
        else if (a == "--faithful") o.cfg.variant = Variant::Faithful;
        else if (a == "--threads") o.threads = std::atoi(next().c_str());
        else if (a == "--image-workers") o.image_workers = std::atoi(next().c_str());
        else if (a == "--prefetch") o.prefetch = true;
        else if (a == "--light") o.cfg.light_parallel = choose<bool>(a, next(), {{"serial", false}, {"parallel", true}});
        else if (a == "--m1") {
            o.cfg.m1_fused = choose<bool>(a, next(), {{"plain", false}, {"fused", true}});
            o.m1_given = true;
        }
        else if (a == "--m2-sched") o.cfg.m2_sched = parse_sched(next());
        else if (a == "--m7-sched") o.cfg.m7_sched = parse_sched(next());
        else if (a == "--m4-level") o.cfg.m4_level = choose<M4Level>(a, next(), {{"lines", M4Level::Lines}, {"bands", M4Level::Bands}});
        else if (a == "--m5-gather")
            o.cfg.m5_gather = choose<M5Gather>(a, next(), {{"fused", M5Gather::Fused}, {"materialized", M5Gather::Materialized}});
        else if (a == "--m6-ssim")
            o.cfg.m6_ssim = choose<M6Ssim>(a, next(), {{"tasks", M6Ssim::Tasks}, {"inner", M6Ssim::Inner}, {"strips", M6Ssim::Strips}});
        else if (a == "--rf-sum") o.cfg.rf_sum = choose<RfSum>(a, next(), {{"ordered", RfSum::Ordered}, {"reduction", RfSum::Reduction}});
        else if (a == "--rf-order") o.cfg.rf_order = choose<RfOrder>(a, next(), {{"image", RfOrder::ImageMajor}, {"tree", RfOrder::TreeMajor}});
        else if (a == "--dump") o.dump = next();
        else if (a == "--trace") o.trace = next();
        else if (a == "--report") o.report = next();
        else if (a == "--timing") o.timing = true;
        else if (a == "--balance") o.balance = true;
        else if (a == "--warmup") o.warmup = true;
        else if (a == "--quiet") o.quiet = true;
        else if (!a.empty() && a[0] == '-') throw std::runtime_error("unknown option " + a);
        else o.images.push_back(a);
    }
    if (o.images.empty()) throw std::runtime_error("no input images (--help lists the options)");
    if (o.cfg.variant == Variant::Faithful && o.cfg.mode != Mode::Sequential)
        throw std::runtime_error("--faithful is a sequential variant; use it with --mode seq");
#ifndef _OPENMP
    if (o.cfg.mode != Mode::Sequential) throw std::runtime_error("this binary was built without OpenMP; only --mode seq is available");
#endif
    // thread budget
    if (o.cfg.mode == Mode::Sequential && o.image_workers <= 1) {
        o.threads = 1;
    } else if (o.threads <= 0) {
#ifdef _OPENMP
        o.threads = omp_get_max_threads();
#else
        o.threads = 1;
#endif
    }
    if (o.image_workers < 1) o.image_workers = 1;
    if (o.image_workers > o.threads) o.image_workers = o.threads;
    if (o.image_workers > static_cast<int>(o.images.size())) o.image_workers = static_cast<int>(o.images.size());
    o.cfg.threads = std::max(1, o.threads / o.image_workers);
    // fused M1 filtering is a sequential improvement that belongs to the designed versions;
    // the baseline keeps its original filter, so its timings stay comparable over time
    if (!o.m1_given) o.cfg.m1_fused = o.cfg.designed();
    return o;
}

struct Loaded {
    Mat gray;
    std::string error;
    double seconds = 0;
};

Loaded load_one(const std::string& path, const Options& o) {
    Loaded l;
    const double t0 = wall_seconds();
    try {
        l.gray = load_gray(path, o.pirm, o.shave);
    } catch (const std::exception& e) {
        l.error = e.what();
    }
    l.seconds = wall_seconds() - t0;
    return l;
}

// Decodes images ahead of the workers on one helper thread (producer) and hands them out by
// index (consumers), so that decoding overlaps with computation.
//
// The buffer holds at most `capacity` decoded images that nobody has taken yet. Both sides
// sleep on a condition variable and re-check their predicate after every wake-up.
// No deadlock: workers claim indices in increasing order, at most `workers` indices are
// claimed and not yet taken, and capacity > workers. So when the producer is blocked on a
// full buffer, at least one buffered image belongs to a worker that is already asking for
// it; taking it frees a slot.
class Prefetcher {
public:
    Prefetcher(const Options& o, int workers) : o_(o), slots_(o.images.size()), ready_(o.images.size(), 0), capacity_(workers + 1) {
        producer_ = std::thread([this] { produce(); });
    }
    ~Prefetcher() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        space_.notify_all();
        producer_.join();
    }
    Loaded take(size_t i) {
        std::unique_lock<std::mutex> lock(mu_);
        available_.wait(lock, [&] { return ready_[i] != 0; });
        Loaded l = std::move(slots_[i]);
        --buffered_;
        lock.unlock();
        space_.notify_one();
        return l;
    }

private:
    void produce() {
        for (size_t i = 0; i < slots_.size(); ++i) {
            {
                std::unique_lock<std::mutex> lock(mu_);
                space_.wait(lock, [&] { return buffered_ < capacity_ || stop_; });
                if (stop_) return;
            }
            Loaded l = load_one(o_.images[i], o_);  // decoding happens outside the lock
            {
                std::lock_guard<std::mutex> lock(mu_);
                slots_[i] = std::move(l);
                ready_[i] = 1;
                ++buffered_;
            }
            available_.notify_all();  // several workers may wait for different indices
        }
    }

    const Options& o_;
    std::vector<Loaded> slots_;
    std::vector<char> ready_;
    int capacity_;
    int buffered_ = 0;
    bool stop_ = false;
    std::mutex mu_;
    std::condition_variable available_, space_;
    std::thread producer_;
};

std::string basename(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}

void dump_line(FILE* f, const std::string& name, const ImageResult& r) {
    std::fprintf(f, "%s %.17g %.17g %.17g %.17g %.17g", name.c_str(), r.ma, r.niqe, r.s[0], r.s[1], r.s[2]);
    for (const Vec* v : {&r.feat.f1, &r.feat.f2, &r.feat.f3}) {
        std::fprintf(f, " %zu", v->size());
        for (double x : *v) std::fprintf(f, " %.17g", x);
    }
    std::fprintf(f, "\n");
}

double peak_rss_mb() {
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    return u.ru_maxrss / 1024.0;  // Linux reports kilobytes
}

void print_load(FILE* f, const char* name, const LoadStats& st, bool machine) {
    int n = 0;
    for (int t = 0; t < LoadStats::kMaxThreads; ++t)
        if (st.chunks[t].v > 0) n = t + 1;
    if (n == 0) return;
    double mx = 0, sum = 0;
    for (int t = 0; t < n; ++t) {
        mx = std::max(mx, st.seconds[t].v);
        sum += st.seconds[t].v;
    }
    const double mean = sum / n;
    if (machine) {
        std::fprintf(f, "load name=%s threads=%d max=%.6f mean=%.6f imbalance=%.4f per_thread=", name, n, mx, mean, mean > 0 ? mx / mean : 0.0);
        for (int t = 0; t < n; ++t) std::fprintf(f, "%s%.6f", t ? "," : "", st.seconds[t].v);
        std::fprintf(f, "\n");
    } else {
        std::fprintf(f, "  %-10s threads %d, busy time max %.4f s, mean %.4f s, max/mean %.3f\n", name, n, mx, mean, mean > 0 ? mx / mean : 0.0);
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        Options o = parse(argc, argv);
        std::setvbuf(stdout, nullptr, _IOLBF, 0);  // show progress when redirected
#ifdef _OPENMP
        omp_set_dynamic(0);            // the thread budget is exact, never reduced by the runtime
        omp_set_max_active_levels(2);  // image level, then one team per image
#endif
        const double t_models = wall_seconds();
        MaModel ma_model;
        NiqeModel niqe_model;
        ma_model.load(o.models + "/ma_model.bin");
        niqe_model.load(o.models + "/niqe_params.bin");
        const double model_seconds = wall_seconds() - t_models;

        LoadStats load_m2, load_m6, load_m7;
        Trace trace;
        if (o.balance) {
            o.cfg.load_m2 = &load_m2;
            o.cfg.load_m6 = &load_m6;
            o.cfg.load_m7 = &load_m7;
        }

        const size_t n = o.images.size();
        std::vector<ImageResult> results(n);
        std::vector<double> io_seconds(n, 0.0);
        const bool forests_per_image = o.cfg.rf_order == RfOrder::ImageMajor;

        if (o.warmup) {  // starts the thread pool and touches the model once
            Loaded l = load_one(o.images[0], o);
            ImageResult scratch;
            if (l.error.empty()) evaluate_image(l.gray, ma_model, niqe_model, o.cfg, true, scratch);
        }
        if (!o.trace.empty()) o.cfg.trace = &trace;

        const double t_start = wall_seconds();
        {
            std::unique_ptr<Prefetcher> prefetch;
            if (o.prefetch) prefetch.reset(new Prefetcher(o, o.image_workers));
            // Image level: one worker per image in flight. dynamic,1 hands out images in
            // increasing order, which the prefetcher relies on, and balances images of
            // different sizes. Each result goes to its own slot, so workers share no state.
            #pragma omp parallel for num_threads(o.image_workers) schedule(dynamic, 1) if (o.image_workers > 1)
            for (long i = 0; i < static_cast<long>(n); ++i) {
                Loaded l = prefetch ? prefetch->take(i) : load_one(o.images[i], o);
                io_seconds[i] = l.seconds;
                ImageResult& r = results[i];
                if (!l.error.empty()) {
                    r.valid = false;
                    r.error = l.error;
                    continue;
                }
                Config cfg = o.cfg;
                cfg.image_index = static_cast<int>(i);
                evaluate_image(l.gray, ma_model, niqe_model, cfg, forests_per_image, r);
            }
        }
        double batch_forest_seconds = 0;
        if (!forests_per_image) {
            // tree-major order: each tree is applied to every image while its nodes are in cache
            const double t0 = wall_seconds();
            Config cfg = o.cfg;
            cfg.threads = o.threads;
            std::vector<size_t> idx;
            for (size_t i = 0; i < n; ++i)
                if (results[i].valid) idx.push_back(i);
            std::vector<double> out(idx.size());
            for (int k = 0; k < 3; ++k) {
                std::vector<const Vec*> xs;
                for (size_t i : idx) xs.push_back(k == 0 ? &results[i].feat.f1 : k == 1 ? &results[i].feat.f2 : &results[i].feat.f3);
                ma_model.forest[k].predict_batch(xs, out.data(), cfg);
                for (size_t j = 0; j < idx.size(); ++j) results[idx[j]].s[k] = out[j];
            }
            for (size_t i : idx) fuse_scores(ma_model, results[i]);
            batch_forest_seconds = wall_seconds() - t0;
        }
        const double wall = wall_seconds() - t_start;

        // ---- output ----------------------------------------------------------------
        FILE* dump = o.dump.empty() ? nullptr : std::fopen(o.dump.c_str(), "w");
        if (!o.dump.empty() && !dump) throw std::runtime_error("cannot write " + o.dump);
        ModuleTimer total;
        double sum_ma = 0, sum_nq = 0, io_total = 0;
        size_t n_valid = 0;
        if (!o.quiet) std::printf("%-32s %12s %12s %12s\n", "image", "Ma", "NIQE", "PI");
        for (size_t i = 0; i < n; ++i) {
            const ImageResult& r = results[i];
            const std::string name = basename(o.images[i]);
            io_total += io_seconds[i];
            for (int m = 1; m <= 8; ++m) total.seconds[m] += r.timer.seconds[m];
            if (!r.valid) {
                std::printf("%-32s invalid: %s\n", name.c_str(), r.error.c_str());
                continue;
            }
            ++n_valid;
            sum_ma += r.ma;
            sum_nq += r.niqe;
            if (!o.quiet) std::printf("%-32s %12.6f %12.6f %12.6f\n", name.c_str(), r.ma, r.niqe, r.pi);
            if (o.timing && !o.quiet) {
                std::printf("    time [s]:");
                for (int m = 1; m <= 8; ++m) std::printf("  M%d %.3f", m, r.timer.seconds[m]);
                std::printf("\n");
            }
            if (dump) dump_line(dump, name, r);
        }
        if (dump) std::fclose(dump);
        total.seconds[8] += batch_forest_seconds;
        const double nv = static_cast<double>(n_valid);
        const double mean_ma = n_valid ? sum_ma / nv : std::nan(""), mean_nq = n_valid ? sum_nq / nv : std::nan("");
        const double set_pi = ((10 - mean_ma) + mean_nq) / 2;
        std::printf("Perceptual Index over %zu image(s): %.6f  (mean Ma %.6f, mean NIQE %.6f)\n", n_valid, set_pi, mean_ma, mean_nq);
        if (n_valid != n) std::printf("%zu image(s) could not be scored\n", n - n_valid);
        if (o.timing) {
            double all = 0;
            for (int m = 1; m <= 8; ++m) all += total.seconds[m];
            std::printf("Total time per module [s]:");
            for (int m = 1; m <= 8; ++m) std::printf("  M%d %.3f (%.1f%%)", m, total.seconds[m], 100 * total.seconds[m] / all);
            std::printf("  I/O %.3f\n", io_total + model_seconds);
            std::printf("Mode %s, %d thread(s), %d image worker(s): wall %.3f s, %.3f images/s, peak memory %.0f MB\n", mode_name(o.cfg.mode),
                        o.threads, o.image_workers, wall, n / wall, peak_rss_mb());
        }
        if (o.balance) {
            std::printf("Load balance (per-thread busy time):\n");
            print_load(stdout, "M2 windows", load_m2, false);
            print_load(stdout, "M6 SSIM", load_m6, false);
            print_load(stdout, "M7 blocks", load_m7, false);
        }
        if (!o.trace.empty()) {
            FILE* f = std::fopen(o.trace.c_str(), "w");
            if (!f) throw std::runtime_error("cannot write " + o.trace);
            std::fprintf(f, "image task start end thread\n");
            for (const Trace::Event& e : trace.events)
                std::fprintf(f, "%d %s %.6f %.6f %d\n", e.image, e.name.c_str(), e.t0 - t_start, e.t1 - t_start, e.thread);
            std::fclose(f);
        }
        if (!o.report.empty()) {
            FILE* f = std::fopen(o.report.c_str(), "w");
            if (!f) throw std::runtime_error("cannot write " + o.report);
            const Config& c = o.cfg;
            std::fprintf(f,
                         "config mode=%s variant=%s threads=%d image_workers=%d threads_per_image=%d prefetch=%d light=%s m1=%s "
                         "m2_sched=%s,%ld m7_sched=%s,%ld m4_level=%s m5_gather=%s m6_ssim=%s rf_sum=%s rf_order=%s\n",
                         mode_name(c.mode), c.variant == Variant::Faithful ? "faithful" : "baseline", o.threads, o.image_workers, c.threads,
                         o.prefetch ? 1 : 0, c.light_parallel ? "parallel" : "serial", c.m1_fused ? "fused" : "plain", sched_name(c.m2_sched),
                         c.m2_sched.chunk, sched_name(c.m7_sched), c.m7_sched.chunk, c.m4_level == M4Level::Lines ? "lines" : "bands",
                         c.m5_gather == M5Gather::Fused ? "fused" : "materialized",
                         c.m6_ssim == M6Ssim::Tasks ? "tasks" : c.m6_ssim == M6Ssim::Inner ? "inner" : "strips",
                         c.rf_sum == RfSum::Ordered ? "ordered" : "reduction", c.rf_order == RfOrder::ImageMajor ? "image" : "tree");
            for (size_t i = 0; i < n; ++i) {
                const ImageResult& r = results[i];
                std::fprintf(f, "image name=%s valid=%d ma=%.17g niqe=%.17g pi=%.17g wall=%.6f io=%.6f", basename(o.images[i]).c_str(),
                             r.valid ? 1 : 0, r.ma, r.niqe, r.pi, r.wall, io_seconds[i]);
                for (int m = 1; m <= 8; ++m) std::fprintf(f, " m%d=%.6f", m, r.timer.seconds[m]);
                std::fprintf(f, "\n");
            }
            std::fprintf(f, "total images=%zu valid=%zu pi=%.17g wall=%.6f throughput=%.6f peak_rss_mb=%.1f model_load=%.6f io=%.6f", n, n_valid,
                         set_pi, wall, n / wall, peak_rss_mb(), model_seconds, io_total);
            for (int m = 1; m <= 8; ++m) std::fprintf(f, " m%d=%.6f", m, total.seconds[m]);
            std::fprintf(f, "\n");
            if (o.balance) {
                print_load(f, "m2_windows", load_m2, true);
                print_load(f, "m6_ssim", load_m6, true);
                print_load(f, "m7_blocks", load_m7, true);
            }
            std::fclose(f);
        }
        return n_valid == n ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

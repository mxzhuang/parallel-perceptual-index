// Loop-level building blocks shared by all modules.
//
// Every parallel loop of the designed versions goes through par_for(), so that one place
// decides how iterations map onto threads:
//   - outside the task graph: `parallel for` with the requested schedule and exactly
//     cfg.threads threads;
//   - inside the task graph (cfg.tasks): `taskloop` on the team that is already running, so
//     nested loops never create more threads than the fixed budget;
//   - one thread, or a loop nested inside another par_for: a plain loop.
//
// Reductions are made deterministic with a chunk grid that depends only on the problem size
// (reduce_chunks). Each chunk is summed in index order into its own slot and the slots are
// added in chunk order, so the result is identical for any thread count and any schedule.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "config.h"

inline int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

inline double wall_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One value per cache line. Used for per-thread or per-chunk accumulators that different
// threads update, so that neighbouring slots never share a line (no false sharing).
template <typename T>
struct alignas(64) Padded {
    T v{};
};

// Per-thread busy time of an instrumented loop, for the load-balance measurements.
struct LoadStats {
    static constexpr int kMaxThreads = 256;
    Padded<double> seconds[kMaxThreads];
    Padded<long> chunks[kMaxThreads];
    void add(double s) {
        const int t = thread_id() % kMaxThreads;
        seconds[t].v += s;
        chunks[t].v += 1;
    }
};

// Start/end time and executing thread of each module task (written to --trace).
struct Trace {
    struct Event {
        int image;
        std::string name;
        double t0, t1;
        int thread;
    };
    std::vector<Event> events;
    std::mutex mu;
    void add(int image, const char* name, double t0, double t1) {
        std::lock_guard<std::mutex> lock(mu);  // rare: a handful of events per image
        events.push_back({image, name, t0, t1, thread_id()});
    }
};

namespace detail {
// Depth of par_for nesting on this thread; an inner loop of a `parallel for` runs serially
// instead of opening a nested team that would exceed the thread budget.
inline int& loop_depth() {
    static thread_local int depth = 0;
    return depth;
}

// First exception thrown by any chunk; rethrown on the calling thread after the loop,
// because an exception must not leave an OpenMP region.
struct ErrorSlot {
    std::exception_ptr first;
    std::mutex mu;
    void capture() {
        std::lock_guard<std::mutex> lock(mu);
        if (!first) first = std::current_exception();
    }
    void rethrow() {
        if (first) std::rethrow_exception(first);
    }
};
}  // namespace detail

// Default chunk size (iterations) for a schedule: block partition for static, about eight
// chunks per thread for dynamic and guided.
inline long default_chunk(long n, int threads, Sched s) {
    if (s.chunk > 0) return s.chunk;
    const long t = std::max(1, threads);
    if (s.kind == SchedKind::Static) return std::max(1L, (n + t - 1) / t);
    return std::max(1L, n / (8 * t));
}

// Calls body(lo, hi) for a partition of [0, n) into contiguous chunks. The body must be
// safe to run concurrently on disjoint chunks.
template <class Body>
void par_for(const Config& cfg, long n, Sched sched, Body&& body, LoadStats* stats = nullptr) {
    (void)cfg;
    (void)sched;
    if (n <= 0) return;
    auto run = [&](long lo, long hi) {
        if (!stats) {
            body(lo, hi);
            return;
        }
        const double t0 = wall_seconds();
        body(lo, hi);
        stats->add(wall_seconds() - t0);
    };
#ifdef _OPENMP
    const bool serial = !cfg.designed() || cfg.threads <= 1 || (!cfg.tasks && detail::loop_depth() > 0);
#else
    const bool serial = true;
#endif
    if (serial) {
        run(0, n);
        return;
    }
#ifdef _OPENMP
    const long chunk = default_chunk(n, cfg.threads, sched);
    const long nchunks = (n + chunk - 1) / chunk;
    detail::ErrorSlot err;
    auto guarded = [&](long c) {
        try {
            run(c * chunk, std::min(n, (c + 1) * chunk));
        } catch (...) {
            err.capture();
        }
    };
    if (cfg.tasks) {
        // one task per chunk; the implicit taskgroup of taskloop waits for all of them
        #pragma omp taskloop grainsize(1) default(shared)
        for (long c = 0; c < nchunks; ++c) guarded(c);
    } else {
        // chunks are formed above, so every schedule hands out one chunk at a time
        #pragma omp parallel num_threads(cfg.threads) default(shared)
        {
            ++detail::loop_depth();
            switch (sched.kind) {
                case SchedKind::Static: {
                    #pragma omp for schedule(static, 1) nowait
                    for (long c = 0; c < nchunks; ++c) guarded(c);
                    break;
                }
                case SchedKind::Dynamic: {
                    #pragma omp for schedule(dynamic, 1) nowait
                    for (long c = 0; c < nchunks; ++c) guarded(c);
                    break;
                }
                case SchedKind::Guided: {
                    #pragma omp for schedule(guided, 1) nowait
                    for (long c = 0; c < nchunks; ++c) guarded(c);
                    break;
                }
            }
            --detail::loop_depth();
        }
    }
    err.rethrow();
#endif
}

// A loop over a few coarse, unequal units (bands, scales, forests) whose bodies contain
// par_for loops of their own. Inside the task graph each unit becomes a task and the inner
// loops nest as taskloops. Elsewhere the units run one after another, so that the inner
// loops get all the threads: a handful of unequal units cannot be balanced across a team.
template <class Body>
void par_outer(const Config& cfg, int n, Body&& body) {
#ifdef _OPENMP
    if (cfg.designed() && cfg.tasks && cfg.threads > 1) {
        detail::ErrorSlot err;
        #pragma omp taskloop grainsize(1) default(shared)
        for (int i = 0; i < n; ++i) {
            try {
                body(i);
            } catch (...) {
                err.capture();
            }
        }
        err.rethrow();
        return;
    }
#endif
    (void)cfg;
    for (int i = 0; i < n; ++i) body(i);
}

// ---- deterministic reductions -------------------------------------------------------
// Number of chunks used to reduce n elements. Depends on n only, never on the thread count.
inline long reduce_chunks(long n) { return std::max(1L, std::min(256L, n / 2048)); }

// Bounds of chunk c of nchunks over [0, n).
inline void chunk_bounds(long n, long nchunks, long c, long& lo, long& hi) {
    lo = n * c / nchunks;
    hi = n * (c + 1) / nchunks;
}

// Sum of term(i) for i in [0, n): chunks are summed in index order (in parallel) and the
// chunk sums are added in chunk order.
template <class Term>
double det_sum(const Config& cfg, long n, Term&& term) {
    const long nc = reduce_chunks(n);
    std::vector<Padded<double>> part(nc);
    par_for(cfg, nc, Sched{}, [&](long clo, long chi) {
        for (long c = clo; c < chi; ++c) {
            long lo, hi;
            chunk_bounds(n, nc, c, lo, hi);
            double s = 0;
            for (long i = lo; i < hi; ++i) s += term(i);
            part[c].v = s;
        }
    });
    double total = 0;
    for (long c = 0; c < nc; ++c) total += part[c].v;
    return total;
}

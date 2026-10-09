// Per-module wall-clock timing (M1..M8), used for profiling the sequential baseline.
#pragma once

#include <chrono>

struct ModuleTimer {
    double seconds[9] = {0};  // index 1..8 = M1..M8

    struct Scope {
        ModuleTimer* t;
        int m;
        std::chrono::steady_clock::time_point start;
        Scope(ModuleTimer* timer, int module) : t(timer), m(module), start(std::chrono::steady_clock::now()) {}
        ~Scope() {
            if (t) t->seconds[m] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    };
};

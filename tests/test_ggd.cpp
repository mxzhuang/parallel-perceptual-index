// Checks that the binary-search GGD lookup returns exactly the same shape parameter as the
// official linear scan (min(abs(r - rho)), first index on ties) for many rho values.
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

#include "../src/ggd.h"

int main() {
    const GgdTable& t = ggd_table();
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> logu(std::log(0.5), std::log(1e6));
    size_t n = 0, bad = 0;
    auto check = [&](double rho) {
        ++n;
        double a = t.fit_rho(rho), b = t.fit_rho_bruteforce(rho);
        if (!(a == b)) {
            if (++bad <= 10) std::printf("mismatch rho=%.17g fast=%.17g brute=%.17g\n", rho, a, b);
        }
    };
    for (int i = 0; i < 2000000; ++i) check(std::exp(logu(rng)));
    for (double rho : {0.0, 1.0, 1.2, 3.0, 1e9, -1.0, std::numeric_limits<double>::quiet_NaN()}) check(rho);
    std::printf("GGD lookup: %zu values checked, %zu mismatches\n", n, bad);
    return bad == 0 ? 0 : 1;
}

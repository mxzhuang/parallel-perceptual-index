#include "ggd.h"

#include <cmath>
#include <limits>

#include "matlab_compat.h"

GgdTable::GgdTable() {
    g_ = mc::colon(0.03, 0.001, 10);
    r_.resize(g_.size());
    for (size_t i = 0; i < g_.size(); ++i) {
        double g = g_[i];
        double c = std::tgamma(2 / g);
        r_[i] = std::tgamma(1 / g) * std::tgamma(3 / g) / (c * c);
    }
    for (size_t i = 1; i < r_.size(); ++i)
        if (!(r_[i] < r_[i - 1])) decreasing_ = false;
}

double GgdTable::fit_rho_bruteforce(double rho) const {
    size_t best = 0;
    double bd = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 0; i < r_.size(); ++i) {
        double d = std::fabs(r_[i] - rho);
        if (std::isnan(bd) ? !std::isnan(d) : d < bd) {
            bd = d;
            best = i;
        }
    }
    return g_[best];
}

double GgdTable::fit_rho(double rho) const {
    if (!decreasing_ || std::isnan(rho)) return fit_rho_bruteforce(rho);
    const size_t n = r_.size();
    // first index with r <= rho (r is decreasing)
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (r_[mid] <= rho)
            hi = mid;
        else
            lo = mid + 1;
    }
    if (lo == 0) return g_[0];
    if (lo == n) return g_[n - 1];
    double d_prev = std::fabs(r_[lo - 1] - rho), d_here = std::fabs(r_[lo] - rho);
    return d_here < d_prev ? g_[lo] : g_[lo - 1];  // tie -> lower index, as MATLAB min
}

double GgdTable::fit(const double* x, size_t n) const {
    double mu = mc::mean(x, n);
    double v = mc::var(x, n);
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += std::fabs(x[i] - mu);
    double mean_abs = s / static_cast<double>(n);
    mean_abs *= mean_abs;
    double rho = v / (mean_abs + 0.0000001);
    return fit_rho(rho);
}

AggdTable::AggdTable() {
    gam_ = mc::colon(0.2, 0.001, 10);
    r_.resize(gam_.size());
    for (size_t i = 0; i < gam_.size(); ++i) {
        double g = gam_[i];
        double a = std::tgamma(2 / g);
        r_[i] = (a * a) / (std::tgamma(1 / g) * std::tgamma(3 / g));
    }
}

void AggdTable::fit(const double* x, size_t n, double& alpha, double& betal, double& betar) const {
    double sl = 0, sr = 0, sabs = 0, ssq = 0;
    size_t nl = 0, nr = 0;
    for (size_t i = 0; i < n; ++i) {
        double v = x[i];
        if (v < 0) {
            sl += v * v;
            ++nl;
        } else if (v > 0) {
            sr += v * v;
            ++nr;
        }
    }
    for (size_t i = 0; i < n; ++i) sabs += std::fabs(x[i]);
    for (size_t i = 0; i < n; ++i) ssq += x[i] * x[i];
    const double nan = std::numeric_limits<double>::quiet_NaN();
    double leftstd = nl ? std::sqrt(sl / static_cast<double>(nl)) : nan;
    double rightstd = nr ? std::sqrt(sr / static_cast<double>(nr)) : nan;
    double gammahat = leftstd / rightstd;
    double ma = sabs / static_cast<double>(n);
    double rhat = (ma * ma) / (ssq / static_cast<double>(n));
    double g2 = gammahat * gammahat;
    double rhatnorm = (rhat * (std::pow(gammahat, 3) + 1) * (gammahat + 1)) / ((g2 + 1) * (g2 + 1));
    // [~, pos] = min((r_gam - rhatnorm).^2): first minimum, NaN ignored (all NaN -> index 1)
    size_t best = 0;
    double bd = nan;
    for (size_t i = 0; i < r_.size(); ++i) {
        double d = (r_[i] - rhatnorm) * (r_[i] - rhatnorm);
        if (std::isnan(bd) ? !std::isnan(d) : d < bd) {
            bd = d;
            best = i;
        }
    }
    alpha = gam_[best];
    double f = std::sqrt(std::tgamma(1 / alpha) / std::tgamma(3 / alpha));
    betal = leftstd * f;
    betar = rightstd * f;
}

const GgdTable& ggd_table() {
    static const GgdTable t;
    return t;
}

const AggdTable& aggd_table() {
    static const AggdTable t;
    return t;
}

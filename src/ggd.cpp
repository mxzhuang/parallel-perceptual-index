#include "ggd.h"

#include <cmath>
#include <limits>

#include "matlab_compat.h"

namespace {
// g = 0.03:0.001:10; r = gamma(1./g).*gamma(3./g)./(gamma(2./g).^2)
void build_ggd(Vec& g, Vec& r) {
    g = mc::colon(0.03, 0.001, 10);
    r.resize(g.size());
    for (size_t i = 0; i < g.size(); ++i) {
        double c = std::tgamma(2 / g[i]);
        r[i] = std::tgamma(1 / g[i]) * std::tgamma(3 / g[i]) / (c * c);
    }
}

// [~, idx] = min(abs(r - rho)): first minimum, NaN ignored
size_t argmin_abs(const Vec& r, double rho) {
    size_t best = 0;
    double bd = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 0; i < r.size(); ++i) {
        double d = std::fabs(r[i] - rho);
        if (std::isnan(bd) ? !std::isnan(d) : d < bd) {
            bd = d;
            best = i;
        }
    }
    return best;
}

// rho = var / (mean(abs(x - mean))^2 + 1e-7)
double ggd_rho(const double* x, size_t n) {
    double mu = mc::mean(x, n);
    double v = mc::var(x, n);
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += std::fabs(x[i] - mu);
    double mean_abs = s / static_cast<double>(n);
    mean_abs *= mean_abs;
    return v / (mean_abs + 0.0000001);
}
}  // namespace

GgdTable::GgdTable() {
    build_ggd(g_, r_);
    for (size_t i = 1; i < r_.size(); ++i)
        if (!(r_[i] < r_[i - 1])) decreasing_ = false;
}

double GgdTable::fit_rho_bruteforce(double rho) const { return g_[argmin_abs(r_, rho)]; }

double GgdTable::fit_faithful(const double* x, size_t n) {
    const double rho = ggd_rho(x, n);
    Vec g, r;
    build_ggd(g, r);
    return g[argmin_abs(r, rho)];
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

double GgdTable::fit(const double* x, size_t n) const { return fit_rho(ggd_rho(x, n)); }

// gam = 0.2:0.001:10; r_gam = ((gamma(2./gam)).^2)./(gamma(1./gam).*gamma(3./gam))
void AggdTable::build(Vec& gam, Vec& r) {
    gam = mc::colon(0.2, 0.001, 10);
    r.resize(gam.size());
    for (size_t i = 0; i < gam.size(); ++i) {
        double a = std::tgamma(2 / gam[i]);
        r[i] = (a * a) / (std::tgamma(1 / gam[i]) * std::tgamma(3 / gam[i]));
    }
}

AggdTable::AggdTable() { build(gam_, r_); }

void AggdTable::fit(const double* x, size_t n, double& alpha, double& betal, double& betar) const {
    fit_with(gam_, r_, x, n, alpha, betal, betar);
}

void AggdTable::fit_faithful(const double* x, size_t n, double& alpha, double& betal, double& betar) {
    Vec gam, r;
    build(gam, r);
    fit_with(gam, r, x, n, alpha, betal, betar);
}

void AggdTable::fit_with(const Vec& gam, const Vec& r, const double* x, size_t n, double& alpha, double& betal,
                         double& betar) {
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
    for (size_t i = 0; i < r.size(); ++i) {
        double d = (r[i] - rhatnorm) * (r[i] - rhatnorm);
        if (std::isnan(bd) ? !std::isnan(d) : d < bd) {
            bd = d;
            best = i;
        }
    }
    alpha = gam[best];
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

// Shape-parameter estimators used by Ma (GGD) and NIQE (AGGD).
#pragma once

#include <cstddef>

#include "matrix.h"

// Ma et al. gama_gen_gauss: moment matching against the table
// g = 0.03:0.001:10, r(g) = gamma(1/g) gamma(3/g) / gamma(2/g)^2.
// The official code rebuilds the table and scans it linearly on every call; the table is
// monotonic, so we build it once and binary-search it. The chosen index is identical
// (MATLAB's min returns the first index on ties, and so do we).
class GgdTable {
public:
    GgdTable();
    double fit(const double* x, size_t n) const;           // returns the shape parameter
    double fit_rho(double rho) const;                       // lookup only
    double fit_rho_bruteforce(double rho) const;            // reference implementation
private:
    Vec g_, r_;
    bool decreasing_ = true;
};

// NIQE estimateaggdparam: returns alpha, betal, betar.
// Table gam = 0.2:0.001:10, r(gam) = gamma(2/gam)^2 / (gamma(1/gam) gamma(3/gam)).
class AggdTable {
public:
    AggdTable();
    void fit(const double* x, size_t n, double& alpha, double& betal, double& betar) const;

private:
    Vec gam_, r_;
};

const GgdTable& ggd_table();
const AggdTable& aggd_table();

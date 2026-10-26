/* Reviewed implementation of the R1 full-crossing derivations.
 * Resource/numerical errors abort a job; they are never physical rejections.
 */
#include "core.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>

static long allocations_left = -1;
void taxi_test_allocation_failure(long remaining) { allocations_left = remaining; }

int taxi_model_valid(TaxiModel m, double beta) {
    return isfinite(m.rate) && m.rate > 0 && isfinite(m.L) && m.L > 0 &&
        isfinite(m.c) && m.c >= 0 && isfinite(m.b) && m.b >= 0 &&
        isfinite(beta) && (m.b == 0 || beta >= 0 || -beta < m.rate / m.b);
}
int taxi_reserve(TaxiTrip *t, size_t n) {
    if (n <= t->capacity) return 0;
    if (n > SIZE_MAX / sizeof(double)) return -1;
    size_t cap = t->capacity ? t->capacity : 32;
    while (cap < n) {
        if (cap > SIZE_MAX / 2 / sizeof(double)) { cap = n; break; }
        cap *= 2;
    }
    if (allocations_left == 0) return -1;
    if (allocations_left > 0) allocations_left--;
    double *next = realloc(t->v, cap * sizeof(double));
    if (!next) return -1;
    t->v = next; t->capacity = cap;
    return 0;
}
void taxi_free(TaxiTrip *t) {
    free(t->v); memset(t, 0, sizeof(*t));
}
static int append(TaxiSample *s, double x, TaxiModel m) {
    if (!(x > 0) || !isfinite(x) || s->N >= (UINT64_C(1) << 53)) return -1;
    double next = s->X + x;
    double cost = s->C + 1 + m.b * fmax(x - m.c, 0);
    if (!isfinite(next) || !isfinite(cost) || next <= s->X) return -1;
    s->X = next; s->C = cost; s->N++; s->long_steps += x > m.c;
    return 0;
}
int taxi_evaluate(const double *v, size_t n, TaxiModel m, TaxiSample *out) {
    TaxiSample s = {0};
    for (size_t i = 0; i < n && s.X < m.L; i++)
        if (append(&s, v[i], m)) return -1;
    *out = s;
    return s.X >= m.L ? 0 : 1;
}
int taxi_generate(TaxiTrip *t, size_t prefix, TaxiModel m, TaxiRng *r) {
    if (prefix > t->capacity) return -1;
    TaxiSample s = {0};
    if (prefix && taxi_evaluate(t->v, prefix, m, &s) != 1) return -1;
    while (s.X < m.L) {
        if (s.N == SIZE_MAX || taxi_reserve(t, (size_t)s.N + 1)) return -1;
        double x = taxi_exponential(r, m.rate);
        t->v[s.N] = x;
        if (append(&s, x, m)) return -1;
    }
    t->size = (size_t)s.N; t->sample = s;
    return 0;
}
int taxi_initialize(TaxiTrip *t, TaxiModel m, TaxiRng *r, size_t M, const char *kind) {
    if (M) {
        if (taxi_reserve(t, M)) return -1;
        int result;
        do {
            for (size_t i = 0; i < M; i++) t->v[i] = taxi_exponential(r, m.rate);
            result = taxi_evaluate(t->v, M, m, &t->sample);
            if (result < 0) return -1;
        } while (result);
        t->size = M;
    } else if (taxi_generate(t, 0, m, r)) return -1;
    if (!strcmp(kind, "baseline")) return 0;
    double nd;
    if (!strcmp(kind, "high")) nd = 1;
    else if (!strcmp(kind, "low")) nd = m.c > 0 ? floor(m.L / m.c) + 1 : 2;
    else if (!strcmp(kind, "long_N")) nd = ceil(3 * m.rate * m.L) + 1;
    else return -1;
    if (!isfinite(nd) || nd < 1 || nd >= 0x1p53 || nd > (double)(SIZE_MAX / sizeof(double))) return -1;
    size_t n = (size_t)nd;
    if ((M && n > M) || taxi_reserve(t, n)) return -1;
    double step = n == 1 ? m.L + 10 / m.rate : (m.L / (double)n) * (1 + 1e-8);
    for (size_t i = 0; i < n; i++) t->v[i] = step;
    if (taxi_evaluate(t->v, M ? M : n, m, &t->sample)) return -1;
    t->size = M ? M : (size_t)t->sample.N;
    return 0;
}
double taxi_log_accept(double beta, double oldc, double newc, uint64_t oldn, uint64_t newn, int suffix) {
    return -beta * (newc - oldc) + (suffix ? log((double)oldn) - log((double)newn) : 0);
}
int taxi_suffix_propose(const TaxiTrip *old, TaxiTrip *proposal, TaxiModel m, TaxiRng *r, size_t *cut) {
    if (!old->sample.N) return -1;
    size_t j = (size_t)taxi_bounded(r, old->sample.N);
    if (taxi_reserve(proposal, j)) return -1;
    if (j) memcpy(proposal->v, old->v, j * sizeof(double));
    *cut = j; /* zero-based cut means retaining exactly j increments */
    return taxi_generate(proposal, j, m, r);
}
int taxi_position_cdf(double *cdf, size_t M, double floor, double scale) {
    if (M < 2 || !isfinite(floor) || !(floor > 0 && floor <= 1) ||
        !isfinite(scale) || scale <= 0) return -1;
    double denominator = -expm1(-(double)M / scale);
    double factor = -expm1(-1 / scale) / denominator, sum = 0;
    if (!isfinite(factor)) return -1;
    for (size_t i = 0; i < M; i++) {
        double p = floor / (double)M + (1 - floor) * factor * exp(-(double)i / scale);
        double next = sum + p;
        if (!(next > sum) || !isfinite(next)) return -1;
        cdf[i] = next; sum = next;
    }
    if (fabs(sum - 1) > 1e-10 || (M > 1 && cdf[M-2] >= 1)) return -1;
    cdf[M-1] = 1;
    return 0;
}
static size_t choose(TaxiRng *r, const double *cdf, size_t M) {
    if (!cdf) return (size_t)taxi_bounded(r, M);
    double u = taxi_uniform(r);
    size_t lo = 0, hi = M - 1;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (u < cdf[mid]) hi = mid; else lo = mid + 1;
    }
    return lo;
}
int taxi_fixed_propose(const TaxiTrip *old, TaxiTrip *proposal, TaxiModel m, TaxiRng *r,
                      const double *cdf, size_t M) {
    if (M < 2 || old->size != M || taxi_reserve(proposal, M)) return -1;
    memcpy(proposal->v, old->v, M * sizeof(double));
    size_t a = choose(r, cdf, M), b;
    do { b = choose(r, cdf, M); } while (b == a);
    proposal->v[a] = taxi_exponential(r, m.rate);
    proposal->v[b] = taxi_exponential(r, m.rate);
    proposal->size = M;
    return taxi_evaluate(proposal->v, M, m, &proposal->sample);
}
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
int taxi_prefix(TaxiTrip *scratch, TaxiModel m, TaxiRng *r, int sequential, TaxiPrefix *out) {
    TaxiPrefix p = {0, m.L, 0};
    if (sequential) {
        double position = 0;
        for (;;) {
            double gap = taxi_exponential(r, m.rate);
            double next = position + gap;
            if (!isfinite(gap) || gap <= 0 || !isfinite(next) || next <= position) return -1;
            if (next >= m.L) break; /* integrate, rather than use, the crossing overshoot */
            p.pre_cost += 1 + m.b * fmax(gap - m.c, 0);
            p.K++; position = next;
        }
        p.remainder = m.L - position;
    } else {
        /* Poisson additivity keeps each Knuth product safely above underflow.
         * Conditional on K, ordered uniform points give the independent reference.
         */
        double mu = m.rate * m.L;
        if (!isfinite(mu) || mu > 1e6) return -1;
        uint64_t pieces = (uint64_t)ceil(mu / 20);
        if (pieces < 1) pieces = 1;
        double cutoff = exp(-mu / (double)pieces);
        for (uint64_t a = 0; a < pieces; a++) {
            double product = taxi_uniform(r);
            while (product > cutoff) { p.K++; product *= taxi_uniform(r); }
        }
        if (p.K > SIZE_MAX || taxi_reserve(scratch, (size_t)p.K)) return -1;
        for (uint64_t i = 0; i < p.K; i++) scratch->v[i] = m.L * taxi_uniform(r);
        if (p.K) qsort(scratch->v, (size_t)p.K, sizeof(double), cmp_double);
        double last = 0;
        for (uint64_t i = 0; i < p.K; i++) {
            double position = scratch->v[i], gap = position - last;
            if (!(gap > 0) || position >= m.L) return -1;
            p.pre_cost += 1 + m.b * fmax(gap - m.c, 0);
            last = position;
        }
        p.remainder = m.L - last;
    }
    if (!isfinite(p.pre_cost) || !(p.remainder > 0)) return -1;
    *out = p; return 0;
}
double taxi_log_tail(TaxiModel m, TaxiPrefix p, double z, int strict) {
    if (m.b == 0) return (strict ? p.pre_cost + 1 > z : p.pre_cost + 1 >= z) ? 0 : -INFINITY;
    double base = p.pre_cost + 1 + m.b * fmax(p.remainder - m.c, 0);
    if (z < base || (z == base && !strict)) return 0;
    double residual = (z - base) / m.b + fmax(m.c - p.remainder, 0);
    return -m.rate * residual;
}
double taxi_prefix_mean(TaxiModel m, TaxiPrefix p) {
    double d = fmax(m.c - p.remainder, 0);
    return p.pre_cost + 1 + m.b * (fmax(p.remainder - m.c, 0) + exp(-m.rate * d) / m.rate);
}

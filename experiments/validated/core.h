#ifndef TAXI_VALIDATED_CORE_H
#define TAXI_VALIDATED_CORE_H
#include <stddef.h>
#include "rng.h"

typedef struct { double rate, L, c, b; } TaxiModel;
typedef struct { double C, X; uint64_t N, long_steps; } TaxiSample;
typedef struct { double *v; size_t size, capacity; TaxiSample sample; } TaxiTrip;
typedef struct { double pre_cost, remainder; uint64_t K; } TaxiPrefix;

int taxi_model_valid(TaxiModel m, double beta);
int taxi_reserve(TaxiTrip *t, size_t n);
void taxi_free(TaxiTrip *t);
int taxi_evaluate(const double *v, size_t n, TaxiModel m, TaxiSample *out);
int taxi_generate(TaxiTrip *t, size_t prefix, TaxiModel m, TaxiRng *r);
int taxi_initialize(TaxiTrip *t, TaxiModel m, TaxiRng *r, size_t M, const char *kind);
double taxi_log_accept(double beta, double oldc, double newc, uint64_t oldn, uint64_t newn, int suffix);
int taxi_suffix_propose(const TaxiTrip *old, TaxiTrip *proposal, TaxiModel m, TaxiRng *r, size_t *cut);
int taxi_fixed_propose(const TaxiTrip *old, TaxiTrip *proposal, TaxiModel m, TaxiRng *r,
                      const double *cdf, size_t M);
int taxi_position_cdf(double *cdf, size_t M, double floor, double scale);
int taxi_prefix(TaxiTrip *scratch, TaxiModel m, TaxiRng *r, int sequential, TaxiPrefix *out);
double taxi_log_tail(TaxiModel m, TaxiPrefix p, double threshold, int strict);
double taxi_prefix_mean(TaxiModel m, TaxiPrefix p);

/* Deterministic fault injection used only by compiled unit tests. -1 disables. */
void taxi_test_allocation_failure(long remaining);
#endif

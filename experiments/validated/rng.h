/* PCG XSH-RR 64/32, reference seeding (PCG minimal C, Apache-2.0/CC0).
 * Integer arithmetic is modulo 2^64. No platform C rand() dependency.
 * See docs/SATURDAY_HANDOFF.md for the finite uniform-grid contract.
 */
#ifndef TAXI_VALIDATED_RNG_H
#define TAXI_VALIDATED_RNG_H
#include <stdint.h>
#include <math.h>
#include <assert.h>

typedef struct { uint64_t state, inc, draws; } TaxiRng;

static inline uint32_t taxi_pcg32(TaxiRng *r) {
    uint64_t old = r->state;
    r->state = old * UINT64_C(6364136223846793005) + r->inc;
    uint32_t x = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    r->draws++;
    return (x >> rot) | (x << ((-rot) & 31u));
}
static inline int taxi_seed(TaxiRng *r, uint64_t seed, uint64_t stream) {
    if (stream >= (UINT64_C(1) << 63)) return -1;
    r->state = 0; r->inc = (stream << 1u) | 1u; r->draws = 0;
    (void)taxi_pcg32(r); r->state += seed; (void)taxi_pcg32(r);
    r->draws = 0;
    return 0;
}
static inline double taxi_uniform(TaxiRng *r) {
    uint64_t hi = taxi_pcg32(r), lo = taxi_pcg32(r);
    uint64_t k = (hi << 20u) | (lo >> 12u);
    return ((double)k + 0.5) * 0x1p-52;
}
static inline uint32_t taxi_bounded32(TaxiRng *r, uint32_t bound) {
    assert(bound > 0);
    uint32_t threshold = (uint32_t)(-bound) % bound;
    for (;;) { uint32_t x = taxi_pcg32(r); if (x >= threshold) return x % bound; }
}
static inline uint64_t taxi_bounded(TaxiRng *r, uint64_t bound) {
    assert(bound > 0);
    if (bound <= UINT32_MAX) return taxi_bounded32(r, (uint32_t)bound);
    uint64_t threshold = (uint64_t)(-bound) % bound;
    for (;;) {
        uint64_t x = (uint64_t)taxi_pcg32(r) << 32u;
        x |= taxi_pcg32(r);
        if (x >= threshold) return x % bound;
    }
}
static inline double taxi_exponential(TaxiRng *r, double rate) {
    return -log(taxi_uniform(r)) / rate;
}
#endif

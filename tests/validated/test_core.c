#include "../../experiments/validated/core.h"
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static void valid(const TaxiTrip *t, TaxiModel m) {
    TaxiSample s;
    assert(taxi_evaluate(t->v, t->size, m, &s) == 0);
    assert(s.C == t->sample.C && s.X == t->sample.X && s.N == t->sample.N);
    assert(s.C >= (double)s.N && s.N >= 1 && s.long_steps <= s.N);
    double before = 0;
    for (uint64_t i = 0; i + 1 < s.N; i++) before += t->v[i];
    assert(before < m.L && s.X >= m.L);
}
int main(void) {
    TaxiRng r, copy, independent;
    assert(taxi_seed(&r, 42, 54) == 0);
    uint32_t reference[] = {0xa15c02b7,0x7b47f409,0xba1d3330,0x83d2f293,0xbfa4784b,0xcbed606e};
    for (size_t i = 0; i < 6; i++) assert(taxi_pcg32(&r) == reference[i]);
    assert(taxi_seed(&copy, 42, UINT64_C(1) << 63) != 0);
    assert(0.5 * 0x1p-52 > 0 && (0x1p52 - 0.5) * 0x1p-52 < 1);
    char text[200];
    snprintf(text, sizeof(text), "%" PRIu64 " %" PRIu64 " %" PRIu64, r.state, r.inc, r.draws);
    assert(sscanf(text, "%" SCNu64 " %" SCNu64 " %" SCNu64, &copy.state, &copy.inc, &copy.draws) == 3);
    assert(taxi_seed(&independent, 42, 55) == 0);
    for (int i = 0; i < 10000; i++) {
        (void)taxi_pcg32(&independent); /* interleaving does not mutate the tested stream */
        assert(taxi_pcg32(&r) == taxi_pcg32(&copy));
        double u = taxi_uniform(&r), v = taxi_uniform(&copy);
        assert(u > 0 && u < 1 && u == v);
        assert(taxi_bounded(&r, 7) == taxi_bounded(&copy, 7));
        assert(taxi_bounded(&r, UINT64_C(1) << 40) == taxi_bounded(&copy, UINT64_C(1) << 40));
    }
    TaxiModel m = {1,10,2,2};
    assert(taxi_model_valid(m, -1.0/6) && !taxi_model_valid(m, -.5));
    double v[] = {1.5,.5,3,2.5,4};
    TaxiSample s;
    assert(taxi_evaluate(v, 5, m, &s) == 0 && s.C == 12 && s.X == 11.5 && s.N == 5 && s.long_steps == 3);
    assert(taxi_evaluate(v, 4, m, &s) == 1);
    double bad[] = {0, NAN, -1};
    for (size_t i = 0; i < 3; i++) assert(taxi_evaluate(bad+i, 1, m, &s) == -1);
    TaxiTrip cur = {0}, prop = {0}, failed = {0};
    taxi_test_allocation_failure(0);
    assert(taxi_generate(&failed, 0, m, &r) == -1 && failed.v == NULL);
    taxi_test_allocation_failure(-1);
    assert(taxi_initialize(&cur, m, &r, 0, "long_N") == 0);
    int length_changes = 0, rejects = 0;
    for (int i = 0; i < 20000; i++) {
        size_t cut;
        assert(taxi_suffix_propose(&cur, &prop, m, &r, &cut) == 0);
        assert(cut < cur.sample.N && cut < prop.sample.N);
        for (size_t j = 0; j < cut; j++) assert(cur.v[j] == prop.v[j]);
        valid(&prop, m);
        length_changes += cur.sample.N != prop.sample.N;
        double loga = taxi_log_accept(.5, cur.sample.C, prop.sample.C, cur.sample.N, prop.sample.N, 1);
        TaxiSample before = cur.sample;
        if (log(taxi_uniform(&r)) < loga) { TaxiTrip tmp = cur; cur = prop; prop = tmp; }
        else { rejects++; assert(!memcmp(&before, &cur.sample, sizeof(before))); }
        valid(&cur, m);
    }
    assert(length_changes > 1000 && rejects > 1000);
    TaxiModel big = {1,200,2,2};
    assert(taxi_generate(&prop, 0, big, &r) == 0 && prop.sample.N > 80);
    valid(&prop, big); /* no hidden MAX_STEPS at 80 */
    double cdf[80];
    assert(taxi_position_cdf(cdf, 80, .1, 12) == 0);
    assert(taxi_position_cdf(cdf, 80, 0, 12) != 0);
    assert(taxi_position_cdf(cdf, 80, .1, 12) == 0);
    for (size_t i = 0; i < 80; i++) assert(cdf[i] > (i ? cdf[i-1] : 0));
    taxi_free(&cur);
    assert(taxi_initialize(&cur, m, &r, 80, "baseline") == 0);
    for (int i = 0; i < 1000; i++) {
        int rc = taxi_fixed_propose(&cur, &prop, m, &r, cdf, 80);
        assert(rc >= 0);
        if (!rc) valid(&prop, m);
    }
    for (int method = 0; method < 2; method++) {
        for (int i = 0; i < 1000; i++) {
            TaxiPrefix p;
            assert(taxi_prefix(&prop, m, &r, method, &p) == 0);
            assert(p.remainder > 0 && p.remainder <= m.L);
            assert(taxi_prefix_mean(m,p) >= (double)p.K + 1);
            assert(taxi_log_tail(m,p,60,0) <= 0);
        }
    }
    taxi_free(&cur); taxi_free(&prop); taxi_free(&failed);
    puts("PASS PCG vectors/state/interleaving, open uniforms, bounded indices, full crossing, suffix prefix/holding/growth, allocation failure, positional law, conditional prefixes");
    return 0;
}

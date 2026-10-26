
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>





double cost(double eta, double eta_c, double b)
{
    if (eta <= eta_c)
    {
        return 1.0;
    }

    return 1.0 + b* (eta - eta_c);
}

double generate_u(void)
{
    return (rand() + 1.0) / (RAND_MAX + 2.0);
}

double generate_eta(double lambda)
{
    double U = generate_u();

    double eta = -log1p(-U) / lambda;

    return eta;
}

struct TaxiResult
{
    int N;
    double X;
    double C;
};

struct TaxiParams
{
    double eta_c;
    double b;
    double L;
    double lambda;
};

struct TaxiResult simulate_taxi(struct TaxiParams params)
{
    struct TaxiResult result = {0, 0.0, 0.0};

    while (result.X < params.L)
    {
        double eta = generate_eta(params.lambda);

        result.X += eta;

        result.C += cost(eta, params.eta_c, params.b);

        result.N++;
    }

    return result;
}


int main(void)
{
    struct TaxiParams params = {
        2.0,   // eta_c
        2.0,   // b
        10.0,  // L
        1.0    // lambda
    };

    srand(42);

    const int K = 100000;

    double sum_C = 0.0;
    double sum_N = 0.0;

    FILE *fp = fopen("taxi_samples.csv", "w");

    if (fp == NULL)
    {
        perror("Cannot open CSV");
        return 1;
    }

    fprintf(fp, "C,N,X\n");

    for (int i = 0; i < K; i++)
    {
        struct TaxiResult result = simulate_taxi(params);

        assert(result.X >= params.L);
        assert(result.N >= 1);
        assert(result.C >= result.N);

        sum_C += result.C;
        sum_N += result.N;

        fprintf(fp, "%.6f,%d,%.6f\n", result.C, result.N, result.X);
    }


    if (fclose(fp) == EOF)
    {
        perror("Cannot close CSV");
        return 1;
    }

    struct TaxiParams flat = params;
    flat.eta_c = INFINITY;

    for (int i = 0; i < 1000; i++)
    {
        struct TaxiResult result = simulate_taxi(flat);

        assert(result.C == result.N);
    }

    srand(42);
    struct TaxiResult r1 = simulate_taxi(params);

    srand(42);
    struct TaxiResult r2 = simulate_taxi(params);

    assert(r1.N == r2.N);
    assert(r1.X == r2.X);
    assert(r1.C == r2.C);

    printf("All tests passed!\n");

    double avg_C = sum_C / K;

    double avg_N = sum_N / K;

    printf("Simulations = %d\n", K);
    printf("Average C = %.6f\n", avg_C);
    printf("Average N = %.6f\n", avg_N);

    return 0;
}

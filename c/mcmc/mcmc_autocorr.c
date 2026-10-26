#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define MAX_STEPS 80

struct TaxiState
{
    double eta[MAX_STEPS];
};

struct TaxiParams
{
    double eta_c;
    double b;
    double L;
};

struct TaxiResult
{
    int N;
    double X;
    double C;
};

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

double cost(double eta, double eta_c, double b)
{
    if (eta <= eta_c)
    {
        return 1.0;
    }

    return 1.0 + b * (eta - eta_c);
}

struct TaxiResult evaluate_state(
    struct TaxiState state,
    struct TaxiParams params)
{
    struct TaxiResult result = {0, 0.0, 0.0};

    for (int i = 0; i < MAX_STEPS && result.X < params.L; i++)
    {
        double eta = state.eta[i];

        result.X += eta;

        result.C += cost(eta, params.eta_c, params.b);

        result.N++;
    }

    return result;
}

int theta_allowed(double theta, double b, double lambda)
{

    if (theta > 0)
    {
        return 1;
    }

    if (theta < 0)
    {
        if (-theta > b / lambda)
        {
            return 1;
        }
    }

    return 0;
}


int main(void)
{


    const unsigned int seed = 42;
    const int T = 200000;
    const int burn_in = 20000;

    srand(seed);

    struct TaxiParams params = {2.0, 2.0, 10.0};

    const double lambda = 1.0;
    const double theta = 2.0;


    if (!theta_allowed(theta, params.b, lambda))
    {
        fprintf(stderr, "Invalid theta!\n");
        return 1;
    }

    struct TaxiState current = {0};
    struct TaxiResult result_current;

    do
    {
        for (int i = 0; i < MAX_STEPS; i++)
        {
            current.eta[i] = generate_eta(lambda);
        }

        result_current = evaluate_state(current, params);

    } while (result_current.X < params.L);

    int accepted_count = 0;
    int unfinished_proposals = 0;
    int max_N = result_current.N;

    double sum_C = 0.0;

    FILE *fp = fopen("csv/mcmc_autocorr_trace.csv", "w");

    if (fp == NULL)
    {
        perror("Cannot open trace");
        return 1;
    }

    fprintf(fp, "t,C\n");

    for (int t = 0; t < T; t++)
    {
        struct TaxiState proposed = current;

        int j1 = rand() % MAX_STEPS;
        int j2 = rand() % MAX_STEPS;

        while (j2 == j1)
        {
            j2 = rand() % MAX_STEPS;
        }


        proposed.eta[j1] =
            generate_eta(lambda);
        proposed.eta[j2] =
            generate_eta(lambda);

        struct TaxiResult result_new =
            evaluate_state(proposed, params);

        int accepted = 0;

        if (result_new.X >= params.L)
        {
            double U = generate_u();
            double delta_C =
                result_new.C - result_current.C;

            if (log(U) < -delta_C / theta)
            {
                accepted = 1;
            }
        }
        else
        {
            unfinished_proposals++;
        }


        if (accepted)
        {
            current = proposed;
            result_current = result_new;
            accepted_count++;
        }

        fprintf(fp, "%d,%.12f\n", t, result_current.C);

        if (t >= burn_in)
        {
            sum_C += result_current.C;
        }


        if (result_current.N > max_N)
        {
            max_N = result_current.N;
        }
    }

    if (fclose(fp) == EOF)
    {
        perror("Cannot close trace");
        return 1;
    }

    double acceptance_rate =
        (double)accepted_count / T;

    double unfinished_rate =
        (double)unfinished_proposals / T;


    double mean_C = 0.0;
    mean_C = sum_C / (T - burn_in);

    printf("MAX_STEPS = %d\n", MAX_STEPS);
    printf("theta = %.2f\n", theta);
    printf("Acceptance rate = %.6f\n", acceptance_rate);
    printf("Unfinished proposal rate = %.6f\n", unfinished_rate);
    printf("Mean cost after burn-in = %.6f\n", mean_C);
    printf("Maximum observed N = %d\n", max_N);
    printf("Seed = %u\n", seed);
    printf("T = %d\n", T);
    printf("Burn-in = %d\n", burn_in);
    printf("Measured samples = %d\n", T - burn_in);

    return 0;
}

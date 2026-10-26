#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAX_STEPS 40

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



int main(void)
{
    srand(42);

    struct TaxiParams params = {2.0, 2.0, 10.0};

    const double theta = -6.0;
    const int T = 50000;
    const int burn_in = 5000;

    FILE *fp = fopen("csv/mcmc_theta_m6.csv", "w");

    if (fp == NULL)
    {
        perror("Cannot open CSV");
        return 1;
    }

    fprintf(fp, "t,C,accepted\n");

    // Инициализация цепи
    struct TaxiState current = {0};
    struct TaxiResult result_current;

    // Создаём случайную конфигурацию,
    // пока она не станет завершённой поездкой.
    do
    {
        for (int i = 0; i < MAX_STEPS; i++)
        {
            current.eta[i] = generate_eta(1.0);
        }

        result_current = evaluate_state(current, params);

    } while (result_current.X < params.L);

    int accepted_count = 0;

    // Главный цикл Metropolis
    for (int t = 0; t < T; t++)
    {
        struct TaxiState proposed = current;

        int j1 = rand() % MAX_STEPS;
        int j2 = rand() % MAX_STEPS;

        while (j2 == j1)
        {
            j2 = rand() % MAX_STEPS;
        }

        // Перегенерировать два элемента proposed
        // по экспоненциальному распределению.
        proposed.eta[j1] = generate_eta(1.0);
        proposed.eta[j2] = generate_eta(1.0);

        struct TaxiResult result_new =
            evaluate_state(proposed, params);

        int accepted = 0;

        if (result_new.X >= params.L)
        {
            double delta_C =
                result_new.C - result_current.C;

            // Вычислить accepted по правилу Metropolis
            // через log(U), как в предыдущем задании.
            accepted = log(generate_u()) < -delta_C / theta;
        }

        if (accepted)
        {
            // Обновить current и result_current.
            current = proposed;
            result_current = result_new;

            accepted_count++;
        }

        // Записать в CSV:
        // t, текущую стоимость, accepted.
        //
        // ВАЖНО: запись должна выполняться
        // даже при accepted == 0.
        fprintf(fp, "%d,%.6f,%d\n", t, result_current.C, accepted);
    }

    if (fclose(fp) == EOF)
    {
        perror("Cannot close CSV");
        return 1;
    }

    // TODO 5:
    // Вывести acceptance rate:
    // accepted_count / T
    // Не забудь о делении double!

    double acceptance_rate = (double)accepted_count / T;

    printf("Burn-in: %d iterations\n", burn_in);
    printf("Acceptance rate: %.6f\n", acceptance_rate);

    return 0;
}

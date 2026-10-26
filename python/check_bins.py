
import numpy as np
import matplotlib.pyplot as plt

# Direct Monte Carlo
direct = np.loadtxt(
    "csv/taxi_samples.csv",
    delimiter=",",
    skiprows=1
)[:, 0]

# Biased MCMC at theta = -6
data = np.loadtxt(
    "csv/mcmc_theta_m6.csv",
    delimiter=",",
    skiprows=1
)

C_mcmc = data[5000:, 1]

theta = -6.0

# Reweighting
a = C_mcmc / theta
shift = np.max(a)
weights = np.exp(a - shift)

widths = [0.25, 0.5, 1.0]

for width in widths:

    bins = None
    bins = np.arange(5, 40 + width, width)

    hist_direct, edges = np.histogram(
        direct,
        bins=bins,
        density=True
    )

    hist_mcmc, _ = np.histogram(
        C_mcmc,
        bins=bins,
        weights=weights,
        density=True
    )

    centers = None
    centers = (edges[:-1] + edges[1:]) / 2

    plt.figure(figsize=(10, 6))

    plt.step(
        centers,
        hist_direct,
        where="mid",
        label="Direct Monte Carlo"
    )

    plt.step(
        centers,
        hist_mcmc,
        where="mid",
        label="Reweighted MCMC, theta=-6"
    )

    plt.yscale("log")
    plt.xlim(7, 33)
    plt.ylim(bottom=1e-6)
    plt.xlabel("Cost C")
    plt.ylabel("Probability density")
    plt.title(f"Histogram width = {width}")
    plt.legend()
    plt.grid(alpha=0.25)

    filename = f"img/bins_{width}.png"
    plt.savefig(filename, dpi=160)
    plt.close()

    print("Saved:", filename)

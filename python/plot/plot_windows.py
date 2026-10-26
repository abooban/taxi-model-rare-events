
import numpy as np
import matplotlib.pyplot as plt

runs = [
    ("csv/mcmc_trace.csv", "theta = +2"),
    ("csv/mcmc_theta_p5.csv", "theta = +5"),
    ("csv/mcmc_theta_m6.csv", "theta = -6"),
]

burn_in = 5000
samples = []

# Load all three MCMC datasets
for filename, label in runs:
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    C = data[burn_in:, 1]

    samples.append((label, C))

    print(f"{label}: mean C = {np.mean(C):.3f}")
    print(f"{label}: min = {np.min(C):.3f}, max = {np.max(C):.3f}")

# Common histogram bins
max_cost = max(np.max(C) for label, C in samples)
bins = np.arange(0.0, max_cost + 1.0, 0.5)

plt.figure(figsize=(10, 6))

for label, C in samples:
    plt.hist(
        C,
        bins=bins,
        density=True,
        histtype="step",
        label=label
    )

plt.yscale("log")
plt.xlabel("Cost C")
plt.ylabel("Biased probability density")
plt.title("MCMC histograms for different theta")
plt.legend()
plt.grid(alpha=0.25)

plt.savefig("img/mcmc_windows.png", dpi=160)
plt.show()

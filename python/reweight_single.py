
import numpy as np
import matplotlib.pyplot as plt

# Parameters
theta = 2.0
burn_in = 5000

# Load Direct Monte Carlo data
direct_data = np.loadtxt(
    "taxi_samples.csv",
    delimiter=",",
    skiprows=1
)

C_direct = direct_data[:, 0]

# Load biased MCMC data
mcmc_data = np.loadtxt(
    "mcmc_trace.csv",
    delimiter=",",
    skiprows=1
)

C_all = mcmc_data[:, 1]
C_biased = C_all[burn_in:]

# ----------------------------
# Reweighting
# ----------------------------

log_weights = C_biased / theta

# TODO 1:
# Find the maximum value of log_weights
# using np.max().
m = 0.0
m = np.max(log_weights)

# TODO 2:
# Compute stabilized weights:
# weights = exp(log_weights - m)
weights = None
weights = np.exp(log_weights - m)

# ----------------------------
# Histograms
# ----------------------------

bins = np.arange(5.0, 40.5, 0.5)

hist_direct, edges = np.histogram(
    C_direct,
    bins=bins,
    density=True
)

hist_reweighted, _ = np.histogram(
    C_biased,
    bins=bins,
    weights=weights,
    density=True
)

centers = (edges[:-1] + edges[1:]) / 2

# ----------------------------
# Plot
# ----------------------------

plt.figure(figsize=(10, 6))

plt.step(
    centers,
    hist_direct,
    where="mid",
    label="Direct Monte Carlo"
)

plt.step(
    centers,
    hist_reweighted,
    where="mid",
    label="Reweighted MCMC, theta=2"
)

plt.yscale("log")
plt.xlabel("Cost C")
plt.ylabel("Probability density")
plt.title("Direct Monte Carlo vs Reweighted MCMC")
plt.legend()
plt.grid(alpha=0.25)

plt.savefig("img/reweight_comparison.png", dpi=160)
plt.show()

# ----------------------------
# Weight diagnostics
# ----------------------------

# TODO 3:
# Compute the effective sample size of the weights:
#
# ESS_weights = (sum(weights))**2 / sum(weights**2)
#
# Use np.sum().
ESS_weights = (np.sum(weights))**2 / np.sum(weights**2)
print("Direct samples:", len(C_direct))
print("MCMC samples:", len(C_biased))
print("Weight ESS:", ESS_weights)

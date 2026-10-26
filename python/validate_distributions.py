
import numpy as np
import matplotlib.pyplot as plt

burn_in = 5000
bins = np.arange(5.0, 40.5, 0.5)
centers = (bins[:-1] + bins[1:]) / 2
width = bins[1] - bins[0]

# =====================================
# 1. Split Direct Monte Carlo samples
# =====================================

direct = np.loadtxt(
    "csv/taxi_samples.csv",
    delimiter=",",
    skiprows=1
)[:, 0]

C_anchor = direct[:50000]
C_test = direct[50000:]

# Calculate normalization using only
# the first half of Direct MC data.

Z2 = np.mean(np.exp(-C_anchor / 2.0))
Z5 = np.mean(np.exp(-C_anchor / 5.0))

# Overlap ratio calculated in stage 05
Zm6 = Z5 * 163.34474209426642

print("Z(+2):", Z2)
print("Z(+5):", Z5)
print("Z(-6):", Zm6)

# =====================================
# 2. Validation histogram
# =====================================

direct_counts, _ = np.histogram(
    C_test, bins=bins
)

N_direct = len(C_test)

P_direct = direct_counts / (N_direct * width)

# Approximate binomial standard error
prob = direct_counts / N_direct

error_direct = np.sqrt(
    prob * (1.0 - prob) / N_direct
) / width

# =====================================
# 3. Reconstruct MCMC distributions
# =====================================

def reconstruct(filename, theta, Z):

    data = np.loadtxt(
        filename, delimiter=",", skiprows=1
    )

    C = data[burn_in:, 1]

    a = C / theta

    # TODO 1:
    # Find maximum of a.
    shift = None
    shift = np.max(a)

    # TODO 2:
    # Compute stabilized weights
    # using exp(a - shift).
    weights = None
    weights = np.exp(a - shift)

    counts, _ = np.histogram(C, bins=bins)

    weighted, _ = np.histogram(
        C, bins=bins, weights=weights
    )

    # Logarithmic reconstruction
    log_P = np.full(len(counts), -np.inf)

    valid = weighted > 0

    log_P[valid] = (
        np.log(Z)
        + shift
        + np.log(weighted[valid])
        - np.log(len(C) * width)
    )

    P = np.exp(log_P)

    return counts, P


counts_2, P2 = reconstruct(
    "csv/mcmc_trace.csv", 2.0, Z2
)

counts_5, P5 = reconstruct(
    "csv/mcmc_theta_p5.csv", 5.0, Z5
)

counts_m6, Pm6 = reconstruct(
    "csv/mcmc_theta_m6.csv", -6.0, Zm6
)

# =====================================
# 4. Compare neighboring windows
# =====================================

def compare_overlap(
    counts_a, P_a,
    counts_b, P_b,
    c_min, c_max
):

    # TODO 3:
    # Select bins satisfying all conditions:
    # counts_a >= 50
    # counts_b >= 50
    # centers >= c_min
    # centers <= c_max
    mask = None
    mask = (
        (counts_a >= 50)
        & (counts_b >= 50)
        & (centers >= c_min)
        & (centers <= c_max)
    )

    if not np.any(mask):
        print("Insufficient overlap")
        return

    log_ratio = np.log(P_a[mask] / P_b[mask])

    print("Overlap bins:", np.sum(mask))
    print("Median P_a / P_b:",
          np.exp(np.median(log_ratio)))
    print("Log-ratio spread:",
          np.std(log_ratio))
    print()


print("\nOverlap +2 / +5:")
compare_overlap(
    counts_2, P2,
    counts_5, P5,
    10.0, 15.0
)

print("Overlap +5 / -6:")
compare_overlap(
    counts_5, P5,
    counts_m6, Pm6,
    13.0, 18.0
)

# =====================================
# 5. Visualization
# =====================================

plt.figure(figsize=(11, 6))

valid_direct = direct_counts >= 20

plt.errorbar(
    centers[valid_direct],
    P_direct[valid_direct],
    yerr=error_direct[valid_direct],
    fmt=".",
    capsize=2,
    label="Direct MC: validation sample"
)

for label, P in [
    ("MCMC theta = +2", P2),
    ("MCMC theta = +5", P5),
    ("MCMC theta = -6", Pm6),
]:
    plt.step(
        centers,
        np.where(P > 0, P, np.nan),
        where="mid",
        label=label
    )

plt.yscale("log")
plt.xlim(6, 28)
plt.xlabel("Cost C")
plt.ylabel("Probability density")
plt.title("Independent validation of reweighted MCMC")
plt.legend()
plt.grid(alpha=0.25)

plt.savefig(
    "validation_distributions.png",
    dpi=160
)
plt.show()

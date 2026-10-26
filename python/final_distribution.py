
import numpy as np
import matplotlib.pyplot as plt

burn_in = 5000
bins = np.arange(5.0, 40.5, 0.5)
centers = (bins[:-1] + bins[1:]) / 2
width = bins[1] - bins[0]

# Absolute normalization from Direct MC
Z2 = 0.0019984822443927844

# TODO 1:
# Calculate Z5 and Zm6 from our latest
# normalization ratios.
Z5 = None
Zm6 = None
Z5 = Z2 * 35.23428794473919
Zm6 = Z5 * 163.34474209426642

windows = [
    ("theta = +2", "csv/mcmc_trace.csv", 2.0, Z2),
    ("theta = +5", "csv/mcmc_theta_p5.csv", 5.0, Z5),
    ("theta = -6", "csv/mcmc_theta_m6.csv", -6.0, Zm6),
]

# Direct Monte Carlo
direct = np.loadtxt(
    "csv/taxi_samples.csv", delimiter=",", skiprows=1
)[:, 0]

direct_counts, _ = np.histogram(direct, bins=bins)

P_direct = direct_counts / (len(direct) * width)

# Combined estimator
numerator = direct_counts.astype(float)
denominator = np.full(
    len(centers), len(direct), dtype=float
)

plt.figure(figsize=(11, 6))

plt.step(
    centers,
    np.where(P_direct > 0, P_direct, np.nan),
    where="mid",
    label="Direct Monte Carlo",
    linewidth=1.5
)

for label, filename, theta, Z in windows:
    data = np.loadtxt(
        filename, delimiter=",", skiprows=1
    )
    C = data[burn_in:, 1]

    counts, _ = np.histogram(C, bins=bins)

    # Stable reweighting
    a = C / theta
    shift = np.max(a)
    weights = np.exp(a - shift)

    weighted_counts, _ = np.histogram(
        C, bins=bins, weights=weights
    )

    # TODO 2:
    # Reconstruct P(C) from this window.
    # Use:
    # Z * exp(shift) * weighted_counts
    # divided by len(C) * width
    P_window = None
    P_window = Z * np.exp(shift) * weighted_counts / (len(C) * width)

    plt.step(
        centers,
        np.where(P_window > 0, P_window, np.nan),
        where="mid",
        label=label,
        alpha=0.65
    )

    # TODO 3:
    # Update numerator by adding counts.
    #
    # Update denominator by adding:
    # len(C) * exp(-centers / theta) / Z

    numerator += weighted_counts
    denominator += len(C) * np.exp(-centers / theta) / Z



# Combined probability density
P_combined = numerator / denominator / width

# TODO 4:
# Normalize P_combined so its integral
# over the histogram equals 1.

P_combined /= np.sum(P_combined * width)

plt.step(
    centers,
    np.where(P_combined > 0, P_combined, np.nan),
    where="mid",
    color="black",
    linewidth=2,
    label="Combined estimate"
)

plt.yscale("log")
plt.xlabel("Cost C")
plt.ylabel("Probability density")
plt.title("Taxi Model: Combined Cost Distribution")
plt.legend()
plt.grid(alpha=0.25)

plt.savefig("img/final_distribution.png", dpi=160)
plt.show()

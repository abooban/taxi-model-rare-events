
import numpy as np

burn_in = 5000
bins = np.arange(5.0, 40.5, 0.5)
centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]


def load_window(filename, theta):
    data = np.loadtxt(
        filename, delimiter=",", skiprows=1
    )

    C = data[burn_in:, 1]

    # Stable logarithmic reweighting
    log_weights = C / theta

    # TODO 1:
    # Find max(log_weights).
    # Then compute exp(log_weights - maximum).
    shift = 0.0
    weights = None
    shift = np.max(log_weights)
    weights = np.exp(log_weights - shift)

    counts, _ = np.histogram(C, bins=bins)

    weighted, _ = np.histogram(
        C, bins=bins, weights=weights
    )

    # Logarithm of the unnormalized
    # reconstructed probability density
    log_H = np.full(len(counts), np.nan)

    valid = weighted > 0

    log_H[valid] = (
        np.log(weighted[valid])
        + shift
        - np.log(len(C))
        - np.log(bin_width)
    )

    return counts, log_H


window_2 = load_window("csv/mcmc_trace.csv", 2.0)
window_5 = load_window("csv/mcmc_theta_p5.csv", 5.0)
window_m6 = load_window("csv/mcmc_theta_m6.csv", -6.0)


def compare_windows(window_a, window_b, c_min, c_max):
    counts_a, log_H_a = window_a
    counts_b, log_H_b = window_b

    # TODO 2:
    # Select bins where:
    # - counts_a >= 50
    # - counts_b >= 50
    # - centers >= c_min
    # - centers <= c_max
    #
    # Combine conditions using &.
    mask = None
    mask = (
        (counts_a >= 50)
        & (counts_b >= 50)
        & (centers >= c_min)
        & (centers <= c_max)
    )

    if not np.any(mask):
        print("Not enough overlap!")
        return

    differences = log_H_a[mask] - log_H_b[mask]

    # TODO 3:
    # Compute the median of differences.
    # This estimates log(Z_b / Z_a).
    log_ratio = None
    log_ratio = np.median(differences)

    print("Number of overlap bins:", np.sum(mask))
    print("Log normalization ratio:", log_ratio)
    print("Normalization ratio:", np.exp(log_ratio))
    print()

print("Comparing theta +2 and +5:")
compare_windows(window_2, window_5, 10.0, 15.0)

print("Comparing theta +5 and -6:")
compare_windows(window_5, window_m6, 13.0, 18.0)


import numpy as np

# Direct Monte Carlo samples
data = np.loadtxt(
    "csv/taxi_samples.csv",
    delimiter=",",
    skiprows=1
)

C = data[:, 0]

# Normalization constants from Direct MC

# TODO 1:
# Calculate the average of exp(-C / 2)
Z2 = None
Z2 = np.mean(np.exp(-C / 2))

# TODO 2:
# Calculate the average of exp(-C / 5)
Z5 = None
Z5 = np.mean(np.exp(-C / 5))

# TODO 3:
# Calculate the ratio Z5 / Z2
ratio_direct = None
ratio_direct = Z5 / Z2

ratio_mcmc = 34.580289239075896

print("Z(+2) from Direct MC:", Z2)
print("Z(+5) from Direct MC:", Z5)

print("Direct MC ratio:", ratio_direct)
print("MCMC overlap ratio:", ratio_mcmc)

# TODO 4:
# Calculate the relative difference
# abs(ratio_direct - ratio_mcmc) / ratio_direct

relative_difference = None
relative_difference = abs(ratio_direct - ratio_mcmc) / ratio_direct

print("Relative difference:", relative_difference)

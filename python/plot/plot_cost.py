
import numpy as np
import matplotlib.pyplot as plt

# Read the results from C simulation
data = np.loadtxt(
    "taxi_samples.csv",
    delimiter=",",
    skiprows=1
)

C = data[:, 0]

print(f"Number of trips: {len(C)}")
print(f"Mean cost: {np.mean(C):.6f}")

bins = np.arange(5, 41, 0.5)
plt.hist(C, bins=bins, density=True)

plt.xlabel("Trip cost C")
plt.ylabel("Probability density for K = 100000")
plt.title("Taxi Model: L = 10")


plt.savefig("cost_histogram_K100000.png", dpi=160)

plt.yscale("log")

plt.title("Taxi Model: Logarithmic Probability Density for K = 100000")
plt.savefig("cost_histogram_log_K100000.png", dpi=160)

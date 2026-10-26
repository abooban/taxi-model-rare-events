
import numpy as np
import matplotlib.pyplot as plt

burn_in = 20000
max_lag = 1000

data = np.loadtxt(
    "csv/mcmc_autocorr_trace.csv",
    delimiter=",",
    skiprows=1
)

C = data[burn_in:, 1]
N = len(C)

# Subtract the mean
x = C - np.mean(C)

# Autocorrelation using FFT
fft_size = 1 << (2 * N - 1).bit_length()

spectrum = np.fft.rfft(x, n=fft_size)
power = spectrum * np.conj(spectrum)

acf = np.fft.irfft(power, n=fft_size)
acf = acf[:max_lag + 1]

# Correct for the number of overlapping pairs
acf /= N - np.arange(max_lag + 1)

# Normalize so that rho(0) = 1
acf /= acf[0]

print("Samples after burn-in:", N)
print("Mean C:", np.mean(C))
print("rho(1):", acf[1])
print("rho(10):", acf[10])
print("rho(50):", acf[50])
print("rho(100):", acf[100])

# Approximate integrated autocorrelation time.
# Sum adjacent positive pairs of autocorrelations.
tau_int = 1.0

for k in range(1, max_lag, 2):
    pair = acf[k] + acf[k + 1]

    if pair <= 0:
        break

    tau_int += 2 * pair

ESS = N / tau_int

print("Integrated autocorrelation time:", tau_int)
print("Approximate ESS:", ESS)

# Plot
plt.figure(figsize=(9, 5))
plt.plot(np.arange(max_lag + 1), acf)

plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Lag k")
plt.ylabel("Autocorrelation")
plt.title("MCMC Cost Autocorrelation")

plt.grid(alpha=0.25)
plt.savefig("img/mcmc_autocorrelation.png", dpi=160)
plt.show()

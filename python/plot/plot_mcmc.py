
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(
    "mcmc_trace.csv",
    delimiter=",",
    skiprows=1
)

t = data[:, 0]
C = data[:, 1]
accepted = data[:, 2]

burn_in = 5000

# TODO 1:
# Построй C(t).
# Добавь вертикальную линию на t = burn_in.
# Сохрани как mcmc_cost_trace.png.
plt.figure()
plt.plot(t, C, label="Cost C(t)")
plt.xlabel("Iteration t")
plt.ylabel("Cost C")
plt.savefig("img/mcmc_cost_trace.png")
plt.close()
# TODO 2:
# Возьми C только после burn_in.
# Построй нормированную гистограмму.
# Используй логарифмическую ось Y.
# Сохрани как mcmc_histogram.png.

C_post = C[t >= burn_in]
plt.figure()
plt.hist(C_post, bins=50, density=True, log=True)
plt.xlabel("Cost C")
plt.ylabel("Density (log scale)")
plt.savefig("img/mcmc_histogram.png")
plt.close()

# TODO 3:
# Вычисли acceptance rate через массив accepted.
# Сравни с результатом программы C.
acceptance_rate = np.mean(accepted)
print("Acceptance rate:", acceptance_rate)

# TODO 4:
# Вычисли корреляцию соседних значений
# C(t) и C(t+1) после burn-in.
#
# Подсказка: np.corrcoef()
# Тебе нужны два массива, сдвинутые
# относительно друг друга на один элемент.
C_post = C[t >= burn_in]
correlation = np.corrcoef(C_post[:-1], C_post[1:])[0, 1]
print("Neighbor correlation after burn-in:", correlation)

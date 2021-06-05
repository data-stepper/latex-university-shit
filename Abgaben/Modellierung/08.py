import numpy as np
import matplotlib.pyplot as plt

# This script can be used to simulate the SIR epidemic model

N = 8 * 10 ** 6
alpha = 1.0 / N
beta = 1 / 3
infected_at_start = 10


def simulate_step(susceptible, infected, alpha, beta):
    contacts = susceptible * infected
    new_s = susceptible - alpha * contacts
    new_i = infected - beta * infected + alpha * contacts

    return new_s, new_i


S, I = ([], [])
R = []

n_samples = 10

alpha = np.linspace(0.5 / N, 3 / N, num=n_samples)

s, i = np.array([N - infected_at_start] * n_samples), np.array(
    [infected_at_start] * n_samples
)

for _ in range(100):
    s, i = simulate_step(s, i, alpha, beta)

    S.append(s)
    I.append(i)

    r = N - s - i
    R.append(r)

S = np.stack(S, axis=-1)
I = np.stack(I, axis=-1)
R = np.stack(R, axis=-1)

for s, i, r, a in zip(S, I, R, alpha):
    plt.plot(s, label="susceptible")
    plt.plot(i, label=f"infected alpha = {a*N} / N")
    plt.plot(r, label="recovered")

    plt.legend()
    plt.show()

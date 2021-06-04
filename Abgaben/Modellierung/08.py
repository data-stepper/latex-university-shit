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


S, I = ([N - infected_at_start], [infected_at_start])
R = [0]

s, i = S[-1], I[-1]

for _ in range(100):
    s, i = simulate_step(s, i, alpha, beta)

    S.append(s)
    I.append(i)

    r = N - s - i
    R.append(r)

plt.plot(S, label="susceptible")
plt.plot(I, label="infected")
plt.plot(R, label="recovered")
plt.legend()
plt.show()

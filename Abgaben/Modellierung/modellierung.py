"""Python script für Modellierung (Zettel 2)."""

import numpy as np
import scipy
from scipy.stats import linregress
import matplotlib.pyplot as plt

# Die Eis werte direkt als numpy nd array
eis_values = np.array(
    [
        7.051,
        7.667,
        7.138,
        7.302,
        7.395,
        6.805,
        6.698,
        7.411,
        7.279,
        7.369,
        7.008,
        6.143,
        6.473,
        7.474,
        6.397,
        7.138,
        6.08,
        7.583,
        6.686,
        6.536,
        6.117,
        6.246,
        6.732,
        5.827,
        6.116,
        5.984,
        5.504,
        5.862,
        4.267,
        4.687,
        5.262,
        4.865,
        4.561,
        3.566,
        5.208,
        5.22,
        4.616,
        4.528,
        4.822,
        4.785,
        4.364,
        3.925,
    ],
    dtype=np.float64,  # Maximum precision
)

# Werte für die Zeit (t)
time = np.array([i for i in range(1979, 1979 + eis_values.shape[0])], dtype=np.float64)

# Linearer estimator
slope, intercept, r, p, se = linregress(time, eis_values)

# Quadratischer estimator
z = np.polyfit(time, eis_values, 2)

print(f"Found coefficients for quadratic estimator: {z}")


def est(x):
    """Linear estimation"""

    return intercept + x * slope


def q_est(x):
    """Quadratic estimation"""

    r_val = 0

    for i, c in enumerate(z[::-1]):
        r_val += x ** i * c

    return r_val


lin_est = est(time)
quad_est = q_est(time)

lin_error = eis_values - lin_est
quad_error = eis_values - quad_est

print(slope, intercept, "steigung und offset bei t=0")
plt.plot(time, eis_values, label="Werte")
plt.plot(time, lin_est, label="Linear estimation")
plt.plot(time, quad_est, label="Quadratic estimation")

plt.plot(time, lin_error, label="Linearer Fehler")
plt.plot(time, quad_error, label="Quadratischer Fehler")

plt.legend()

plt.show()

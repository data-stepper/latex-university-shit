import numpy as np

# Matrizen zu potenzieren ist leichter in Python ;)

q = 1.0 / 3.0

# Prob that A wins one point
A = q * (1 - q)
B = q * (1 - q)
E = (1 - q) ** 2 + q ** 2

p = np.array(
    [
        [1, 0, 0, 0, 0],
        [A, E, B, 0, 0],
        [0, A, E, B, 0],
        [0, 0, A, E, B],
        [0, 0, 0, 0, 1],
    ],
    dtype=np.float64,
)

for e in range(1, 500, 200):
    print("\n")
    print(np.linalg.matrix_power(p, e))

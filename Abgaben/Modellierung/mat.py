import numpy as np

# Matrizen zu potenzieren ist leichter in Python ;)

q = 1.0 / 3.0

# Prob that A wins one point
A = q * (1 - q)
B = q * (1 - q)
E = (1 - q) ** 2 + q ** 2

# p = np.array(
#     [
#         [1, 0, 0, 0, 0],
#         [A, E, B, 0, 0],
#         [0, A, E, B, 0],
#         [0, 0, A, E, B],
#         [0, 0, 0, 0, 1],
#     ],
#     dtype=np.float64,
# )

p = np.array(
    [[1, 0, 0, 0], [0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5], [0, 0, 0, 1]],
    dtype=np.float64,
)

print(np.linalg.matrix_power(p, 2))

for e in [2 ** i for i in range(8)]:
    print("\n" + str(e))
    print(np.linalg.matrix_power(p, e))

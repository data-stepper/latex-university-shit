import numpy as np
import array_to_latex as a2l

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

# p = np.array(
#     [[1, 0, 0, 0], [0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5], [0, 0, 0, 1]],
#     dtype=np.float64,
# )

N = 0.25
S = 0.5

p = np.array(
    [
        [S + 0.1, N - 0.1, 0, 0, 0, N],
        [N, S + N, 0, 0, 0, 0],
        [0, N + 0.15, S, N - 0.15, 0, 0],
        [0, 0, N + 0.1, S + 0.1, N - 0.2, 0],
        [0, 0, 0, N, S, N],
        [N, 0, 0, 0, N, S],
    ],
    dtype=np.float64,
)

invar = np.array([0.29, 0.2652, 0.057, 0.0815, 0.1075, 0.1988], dtype=np.float64)

y = print(invar @ p)

p_td = np.linalg.matrix_power(p, 200)

a2l.to_clp(p_td, frmt="{:6.4f}", arraytype="array")

print(p)
print("\n")
print(p_td)
quit()

for e in [2 ** i for i in range(8)]:
    print("\n" + str(e))
    print(np.linalg.matrix_power(p, e))

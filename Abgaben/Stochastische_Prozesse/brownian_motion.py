import numpy as np
import matplotlib.pyplot as plt

num_paths = 2
N = 10000
sigma = 1.0 / np.sqrt(N)

one_dim = np.cumsum(
    np.random.normal(
        0.0,
        sigma,
        size=[
            num_paths,
            N,
        ],
    ),
    axis=-1,
)

for path in one_dim:
    plt.plot(path, label="1d")


plt.legend()
plt.show()

two_dim = np.cumsum(
    np.random.normal(
        0.0,
        sigma,
        size=[
            num_paths,
            2,
            N,
        ],
    ),
    axis=-1,
)

for path in two_dim:
    x, y = path[0], path[1]

    plt.plot(x, y, label="2d")

plt.legend()
plt.show()

# %% Abs value function plot
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

plot_dir = Path("./plots")
plot_dir.mkdir(parents=True, exist_ok=True)


# Define the absolute value function
def abs_val_func(x):
    return abs(x)


# Create the x-axis array
x = np.linspace(-1.5, 1.5, 100)

# Compute the corresponding y-values
y = abs_val_func(x)

# Define the point x0
x0 = 1 / 2

# Calculate the corresponding y-value y0
y0 = abs_val_func(x0)

# Create a new figure and axes
fig, ax = plt.subplots()

# Plot the absolute value function on the new axes
ax.plot(x, y, label="|x|")

# Overlay a scatter plot for the point (1/2, 1/2) on the new axes
ax.scatter(x0, y0, color="red", label="(1/2, 1/2)")

# Add a new gradient arrow using plt.arrow, positioning it to the right of the plot
ax.arrow(
    0.6, 0.4, -0.2, -0.2, head_width=0.05, head_length=0.1, fc="red", ec="red"
)

# Add labels, title, and legend to the new axes
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Absolute Value Function and Gradient Descent")
ax.legend()

fig.savefig(plot_dir / "abs_val_func.pdf")


# %%

import matplotlib.pyplot as plt
import numpy as np

import sympy as sp

def get_numpy_func(expr, symbols):
    """
    Converts a sympy expression to a numpy function.

    Args:
        expr: The sympy expression.
        symbols: A list of sympy symbols.

    Returns:
        A numpy function.
    """
    return sp.lambdify(symbols, expr, 'numpy')

def sympy_to_numpy_func(func, symbols):
    """
    Computes the gradient, Hessian, and the function itself as numpy functions.

    Args:
        func: The sympy expression representing the function.
        symbols: A list of sympy symbols.

    Returns:
        A tuple containing the numpy functions for the function, its gradient, and its Hessian.
    """

    # Compute gradient and Hessian
    grad = [sp.diff(func, var) for var in symbols]
    hessian = [[sp.diff(grad_i, var_j) for var_j in symbols] for grad_i in grad]

    # Convert to numpy functions
    f_np = get_numpy_func(func, symbols)
    grad_np = get_numpy_func(grad, symbols)
    hessian_np = get_numpy_func(hessian, symbols)

    return f_np, grad_np, hessian_np

x, y = sp.symbols('x y')
expr = sp.sqrt(x**2 + 3*y**2)

f_np, grad_np, hessian_np = sympy_to_numpy_func(expr, [x, y])


# def f(x) -> float:
#     return np.sqrt(x[0] ** 2 + 3 * x[1] ** 2)

def f(x) -> float:
    return f_np(x[0], x[1])

def f_grad(x) -> np.ndarray:
    return np.array(grad_np(x[0], x[1]))

def f_hessian(x, ensure_pos_def=True) -> np.ndarray:
    h = np.array(hessian_np(x[0], x[1]))

    if ensure_pos_def:
        w, v = np.linalg.eigh(h)
        h = v @ np.diag(np.maximum(w, 1e-3)) @ v.T

    return h


# grid [-1, 1] x [-1, 1]
x = np.mgrid[-1:1:100j, -1:1:100j]
y = f(x)

fig = plt.figure(figsize=(12, 5))

# 3D subplot
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(x[0], x[1], y, cmap="viridis")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")

# Heatmap subplot
ax2 = fig.add_subplot(122)
im = ax2.imshow(y, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
fig.colorbar(im, ax=ax2)
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# Main title
fig.suptitle("Euclidean Norm: $f(x, y) = \\sqrt{x^2 + 3y^2}$")

# Save and show the figure
fig.savefig(plot_dir / "euclidean_norm.pdf")

# %% Gradient of the function

from scipy.optimize import approx_fprime

x0 = np.array([0.5, 0.75])


# def f_grad(x: np.ndarray) -> np.ndarray:
#     return approx_fprime(x, f)
#
#
# def f_hessian(x: np.ndarray, ensure_pos_def: bool = True) -> np.ndarray:
#     h = approx_fprime(x, f_grad)
#
#     if ensure_pos_def:
#         # Ensure positive definiteness
#         w, v = np.linalg.eigh(h)
#         h = v @ np.diag(np.maximum(w, 1e-3)) @ v.T
#
#     return h
#

print(f_grad(x0))
print(f_hessian(x0))

# %% Start with GD on the same function
from typing import List


def gd(step_size=0.1, max_steps=30) -> List[np.ndarray]:
    x = x0
    steps = [x]
    for _ in range(max_steps):
        grad = f_grad(x)
        x = x - step_size * grad
        steps.append(x)
    return steps


gd_steps = gd()

# heatmap with 'x' steps taken by GD
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(y, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
axs[0].scatter(
    *zip(*gd_steps),
    c="red",
    s=30,
    marker="x",
    label="30 GD steps with size 0.1"
)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].legend()

# and gd with decreasing step size


def gd_with_lr_decay(
    step_size=0.1, target_lr=1e-4, max_steps=30
) -> List[np.ndarray]:
    x = x0
    steps = [x]
    decay = (target_lr / step_size) ** (1 / max_steps)
    for i in range(max_steps):
        grad = f_grad(x)
        x = x - step_size * grad
        step_size *= decay
        steps.append(x)
    return steps


gd_steps_decay_good = gd_with_lr_decay(target_lr=2e-3)
gd_steps_decay_bad = gd_with_lr_decay(target_lr=1e-5)

axs[1].imshow(y, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
axs[1].scatter(
    *zip(*gd_steps_decay_good),
    c="red",
    s=30,
    marker="x",
    label="30 GD steps with good decay"
)
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

axs[1].scatter(
    *zip(*gd_steps_decay_bad),
    c="blue",
    s=30,
    marker="x",
    label="30 GD steps with too strong decay"
)
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

axs[1].legend()

fig.savefig(plot_dir / "gd_steps.pdf")

# %% Newton's method

from scipy.optimize import minimize_scalar


def newton_with_exact_line_search(max_steps=30, tol=1e-6, max_t=0.5) -> List[np.ndarray]:
    x = x0
    steps = [x]
    for _ in range(max_steps):
        grad = f_grad(x)
        if np.linalg.norm(grad) < tol:
            break
        try:
            direction = -np.linalg.solve(f_hessian(x), grad)

            def h(t):
                return f(x + t * direction)

            ls_result = minimize_scalar(h, bounds=(0, max_t), method="bounded")
            step_size = ls_result.x

            x = x + step_size * direction
            steps.append(x)
        except np.linalg.LinAlgError:
            break
    return steps


newton_steps = newton_with_exact_line_search(max_t=0.5)

# heatmap with 'x' steps taken by Newton's method
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(y, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
axs[0].scatter(
    *zip(*newton_steps),
    c="red",
    s=30,
    marker="x",
    label="Newton steps with exact line search for $t \\in [0, 0.5]$"
)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].legend()


def gd_with_exact_line_search(max_steps=30, tol=1e-6, max_t=0.5) -> List[np.ndarray]:
    x = x0
    steps = [x]
    for _ in range(max_steps):
        grad = f_grad(x)
        if np.linalg.norm(grad) < tol:
            break
        try:
            direction = -grad

            def h(t):
                return f(x + t * direction)

            ls_result = minimize_scalar(h, bounds=(0, max_t), method="bounded")
            step_size = ls_result.x

            x = x + step_size * direction
            steps.append(x)
        except np.linalg.LinAlgError:
            break
    return steps


gd_steps_ls = gd_with_exact_line_search()

axs[1].imshow(y, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
axs[1].scatter(
    *zip(*gd_steps_ls),
    c="red",
    s=30,
    marker="x",
    label="GD steps with exact line search for $t \\in [0, 0.5]$"
)
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].legend()

fig.savefig(plot_dir / "exact_line_search_newton_vs_gd.pdf")


# %% Explain why newton direction is better than gradient direction

import matplotlib.pyplot as plt
import numpy as np

z = f(x0)
grad = f_grad(x0)
hessian = f_hessian(x0)
newton_direction = np.linalg.solve(hessian, grad)

# Gradient surface
x_k = x - x0[:, None, None]  # of shape (2, 100, 100)
z_grad = z + np.sum(grad[:, None, None] * x_k, axis=0)

# Newton surface
z_hessian = z_grad.copy()
# Now add the quadratic term
# hessian is of shape (2, 2), x_k is of shape (2, 100, 100)
# compute H @ x_k first
rhs = np.einsum("ij, jmn -> imn", hessian, x_k)
# then x_k^T @ rhs
z_hessian += 0.5 * np.sum(x_k * rhs, axis=0)
m = np.max(y)


def make_scene(ax: plt.Axes, grad_or_hess=True) -> None:
    z = f(x0)
    ax.plot_surface(x[0], x[1], y, cmap="viridis", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if grad_or_hess:
        xyz_grad = np.array([x[0].ravel(), x[1].ravel(), z_grad.ravel()])
        xyz_grad = xyz_grad[:, (xyz_grad[2] > 0) & (xyz_grad[2] < m)]

        ax.plot_trisurf(*xyz_grad, cmap="cividis", alpha=0.5)
    else:
        xyz_hessian = np.array([x[0].ravel(), x[1].ravel(), z_hessian.ravel()])
        xyz_hessian = xyz_hessian[
            :, (xyz_hessian[2] > 0) & (xyz_hessian[2] < m)
        ]

        ax.plot_trisurf(*xyz_hessian, cmap="cividis", alpha=0.5)

    ax.scatter(*x0, z + 0.05, c="red", s=50, marker="x", label="Starting point")
    ax.legend()
    ax.set_zlim(0, np.max(y))


# Create the figure and axes
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

# Call make_scene to plot on each axes
make_scene(ax1, grad_or_hess=True)
make_scene(ax2, grad_or_hess=False)

# Adjust viewing angles
# ax1.view_init(elev=30, azim=20)
# ax2.view_init(elev=20, azim=-30)
for ax in [ax1, ax2]:
    ax.view_init(elev=30, azim=20)

fig.savefig(plot_dir / "3d_newton_and_gradient_surface.pdf")

# clear cache
plt.close(fig)


# %% Visualize using contour plots

# Contour plot original function, z_grad and z_hessian
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Original function
axs[0].contour(x[0], x[1], y, levels=20, cmap="viridis")
axs[0].scatter(*x0, c="red", s=50, marker="x", label="Starting point")
axs[0].legend()
axs[0].set_title("Original function")

# Gradient
axs[1].contour(x[0], x[1], z_grad, levels=20, cmap="cividis")
axs[1].scatter(*x0, c="red", s=50, marker="x", label="Starting point")
axs[1].scatter(0, 0, c="red", s=50, marker="x", label="Minimum")

# Gradient direction
norm_grad = 3 * (grad / np.linalg.norm(grad, ord=2))
axs[1].arrow(
    x0[0],
    x0[1],
    -0.1 * norm_grad[0],
    -0.1 * norm_grad[1],
    head_width=0.05,
    head_length=0.1,
    fc="red",
    ec="red",
    label="Gradient direction",
)

axs[1].legend()
axs[1].set_title("Gradient surface")

# Hessian
axs[2].contour(x[0], x[1], z_hessian, levels=20, cmap="cividis")
axs[2].scatter(*x0, c="red", s=50, marker="x", label="Starting point")
axs[2].scatter(0, 0, c="red", s=50, marker="x", label="Minimum")

# Newton direction
norm_newton = 3 * (newton_direction / np.linalg.norm(newton_direction, ord=2))
axs[2].arrow(
    x0[0],
    x0[1],
    -0.1 * norm_newton[0],
    -0.1 * norm_newton[1],
    head_width=0.05,
    head_length=0.1,
    fc="red",
    ec="red",
    label="Newton direction",
)

axs[2].legend()
axs[2].set_title("Newton surface")

fig.savefig(plot_dir / "contour_newton_and_gradient_surface.pdf")


# %% Quasi-Newton


def bfgs_with_exact_line_search(max_steps=30, tol=1e-6) -> List[np.ndarray]:
    x = x0
    steps = [x]
    h_inv = np.eye(2)
    grad = f_grad(x)

    for _ in range(max_steps):
        if np.linalg.norm(grad) < tol:
            break

        direction = -h_inv @ grad

        def h(t):
            return f(x + t * direction)

        ls_result = minimize_scalar(h, bounds=(0, 1.0), method="bounded")
        step_size = ls_result.x

        next_x = x + step_size * direction
        next_grad = f_grad(next_x)
        s = next_x - x
        y = next_grad - grad

        # Update the inverse Hessian approximation
        V = np.eye(2) - (s[:, None] @ y[None, :]) / (y @ s)
        h_inv = V.T @ h_inv @ V + (s[:, None] @ s[None, :]) / (y @ s)

        x = next_x
        grad = next_grad
        steps.append(x)

    return steps


bfgs_steps = bfgs_with_exact_line_search()

# heatmap with 'x' steps taken by BFGS
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(y, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
axs[0].scatter(
    *zip(*bfgs_steps),
    c="red",
    s=30,
    marker="x",
    label="BFGS steps with exact line search for $t \\in [0, 1]$"
)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].legend()

fig.savefig(plot_dir / "exact_line_search_bfgs.pdf")

# %% Compare convergence rates of Newton, GD and BFGS with exact line search

tol = 1e-20
newton_steps = newton_with_exact_line_search(tol=tol)
gd_steps_ls = gd_with_exact_line_search(tol=tol)
bfgs_steps = bfgs_with_exact_line_search(tol=tol)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

for x_i, name in zip(
    [newton_steps, gd_steps_ls, bfgs_steps],
    ["Newton", "GD", "BFGS"],
):
    x_i = np.array(x_i)
    y_i = np.array([f(x) for x in x_i])
    ax.plot(
        np.arange(x_i.shape[0]),
        y_i,
        label=name,
    )

ax.set_yscale("log")
ax.set_xlabel("Iteration")
ax.set_ylabel("Objective function value")

ax.legend()

fig.savefig(plot_dir / "convergence_rates.pdf")

# %% Sympy
import sympy as sp

def get_numpy_func(expr, symbols):
    """
    Converts a sympy expression to a numpy function.

    Args:
        expr: The sympy expression.
        symbols: A list of sympy symbols.

    Returns:
        A numpy function.
    """
    return sp.lambdify(symbols, expr, 'numpy')

def sympy_to_numpy_func(func, symbols):
    """
    Computes the gradient, Hessian, and the function itself as numpy functions.

    Args:
        func: The sympy expression representing the function.
        symbols: A list of sympy symbols.

    Returns:
        A tuple containing the numpy functions for the function, its gradient, and its Hessian.
    """

    # Compute gradient and Hessian
    grad = [sp.diff(func, var) for var in symbols]
    hessian = [[sp.diff(grad_i, var_j) for var_j in symbols] for grad_i in grad]

    # Convert to numpy functions
    f_np = get_numpy_func(func, symbols)
    grad_np = get_numpy_func(grad, symbols)
    hessian_np = get_numpy_func(hessian, symbols)

    return f_np, grad_np, hessian_np

x, y = sp.symbols('x y')
expr = sp.sqrt(x**2 + 3*y**2)

f_np, grad_np, hessian_np = sympy_to_numpy_func(expr, [x, y])



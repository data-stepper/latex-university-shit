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
x0 = 1/2

# Calculate the corresponding y-value y0
y0 = abs_val_func(x0)

# Create a new figure and axes
fig, ax = plt.subplots()

# Plot the absolute value function on the new axes
ax.plot(x, y, label='|x|')

# Overlay a scatter plot for the point (1/2, 1/2) on the new axes
ax.scatter(x0, y0, color='red', label='(1/2, 1/2)')

# Add a new gradient arrow using plt.arrow, positioning it to the right of the plot
ax.arrow(0.6, 0.4, -0.2, -0.2, head_width=0.05, head_length=0.1, fc='red', ec='red')

# Add labels, title, and legend to the new axes
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Absolute Value Function and Gradient Descent')
ax.legend()

fig.savefig(plot_dir / "abs_val_func.pdf")

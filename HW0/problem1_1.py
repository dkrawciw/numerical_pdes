import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_color_codes(palette="colorblind")

plt.rcParams.update({
	"text.usetex": False,  # keep False to avoid requiring a LaTeX installation
	"mathtext.fontset": "cm",  # Computer Modern (LaTeX-like)
	"font.family": "serif",
	"font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.labelsize": 14,      # increase axis label size
    "axes.titlesize": 16,
    "xtick.labelsize": 14,     # increase tick / bin label size
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

num_points = 30
x_range = (0,2 * np.pi)
x_points = np.linspace(x_range[0],x_range[1], num_points)
# delta_x = (x_range[1] - x_range[0]) / (num_points-1)
delta_x = x_points[1] - x_points[0]

f = lambda x: np.exp(np.sin(x))
f_prime = lambda x: np.cos(x) * np.exp(np.sin(x))

f_vals = f(x_points)
numerical_f_prime = np.zeros(num_points)

for i in range(1, num_points - 1):
    numerical_f_prime[i] = (f_vals[i+1] - f_vals[i-1]) / (2 * delta_x)

numerical_f_prime[0] = (f_vals[1] - f_vals[0]) / delta_x
numerical_f_prime[-1] = (f_vals[-1] - f_vals[-2]) / delta_x

plt.figure(figsize=(8,5))

plt.plot(x_points, numerical_f_prime, "-o", markersize=8, linewidth=4,label="Numerical Solution")
plt.plot(x_points, f_prime(x_points), "--",linewidth=4, label="Analytical Solution")

plt.title("Analytical and Numerical Solutions\nof the Derivative of the Given Function $f(x)$", fontsize=16)
plt.legend()
plt.xlabel(r"$x$")
plt.ylabel(r"$f'(x)$")

plt.tight_layout()
plt.savefig("output/Problem1_1.svg")
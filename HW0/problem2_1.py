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
delta_x = (x_range[1] - x_range[0]) / num_points
x_points = np.linspace(x_range[0],x_range[1], num_points, endpoint=False)

f = lambda x: np.exp(np.sin(x))
f_double_prime = lambda x: (np.square(np.cos(x)) - np.sin(x)) * np.exp(np.sin(x))

f_vals = f(x_points)
f_double_prime_vals = f_double_prime(x_points)

D = np.zeros((num_points, num_points))
D = D + np.eye(num_points) * -2
D = D + np.eye(num_points, k=1)
D = D + np.eye(num_points, k=-1)
D[0,-1] = 1
D[-1,0] = 1
numerical_f_double_prime = 1 / (delta_x ** 2) * D@f_vals

# plt.plot(x_points, f_vals, label="$f(x)$")

plt.plot(x_points, f_double_prime_vals, linewidth=4, label="Analytical Solution")
plt.plot(x_points, numerical_f_double_prime, "--o", markersize=8, linewidth=4, label="Numerical Solution")

plt.title("Analytical and Numerical Solutions of the\nSecond Derivative of the Given Function $f(x)$", fontsize=16)
plt.xlabel(r"$x$")
plt.ylabel(r"$f''(x)$")
plt.legend()

plt.tight_layout()
plt.savefig("output/problem2_1.svg")
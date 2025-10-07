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

list_of_number_of_points = list(range(10,510,20))
x_range = (0,2 * np.pi)

error_2norm = []
error_infnorm = []

for num_points in list_of_number_of_points:
    x_points = np.linspace(x_range[0],x_range[1], num_points, endpoint=False)
    delta_x = x_points[1] - x_points[0]

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
    numerical_f_double_prime = D@f_vals / (delta_x ** 2)

    error_2 = np.linalg.norm(numerical_f_double_prime - f_double_prime_vals, ord=2) / np.linalg.norm(f_double_prime_vals, ord=2)
    error_inf = np.linalg.norm(numerical_f_double_prime - f_double_prime_vals, ord=np.inf) / np.linalg.norm(f_double_prime_vals, ord=np.inf)

    error_2norm.append(error_2)
    error_infnorm.append(error_inf)

plt.figure(figsize=(8,5))

plt.loglog(list_of_number_of_points, error_infnorm, "-o", linewidth=4, markersize=8, label=r"$\infty$-Norm Relative Error")
plt.loglog(list_of_number_of_points, error_2norm, '--o', linewidth=4, markersize=8, label=r"2-Norm Relative Error")

plt.title("Error Convergence of the Numerical Solution of\nthe Second Derivative as the Number of Grid Points Increases")
plt.ylabel(r"$\log \log$ Error")
plt.xlabel("Number of Grid Points")

plt.legend()
plt.tight_layout()
plt.savefig("output/problem2_2.svg")

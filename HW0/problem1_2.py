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

# num_points = 30
x_range = (0,2 * np.pi)
num_of_step_sizes = 30


h = []
error_2norm = []
error_infnorm = []
list_of_number_of_points = list(range(10,510,20))

for num_points in list_of_number_of_points:

    delta_x = (x_range[1] - x_range[0]) / (num_points - 1)

    x_points = np.linspace(x_range[0],x_range[1], num_points)
    h.append(delta_x)
    f = lambda x: np.exp(np.sin(x))
    f_prime = lambda x: np.cos(x) * np.exp(np.sin(x))

    f_vals = f(x_points)
    numerical_f_prime = np.zeros(num_points)

    for i in range(1, num_points - 1):
        numerical_f_prime[i] = (f_vals[i+1] - f_vals[i-1]) / (2 * delta_x)

    numerical_f_prime[0] = (f_vals[1] - f_vals[0]) / delta_x
    numerical_f_prime[-1] = (f_vals[-1] - f_vals[-2]) / delta_x

    curr_error_2norm = np.linalg.norm(numerical_f_prime - f_prime(x_points), ord=2) / np.linalg.norm(f_prime(x_points), ord=2)
    curr_error_infnorm = np.linalg.norm(numerical_f_prime - f_prime(x_points), ord=np.inf) / np.linalg.norm(f_prime(x_points), ord=np.inf)

    error_2norm.append(curr_error_2norm)
    error_infnorm.append(curr_error_infnorm)

plt.figure(figsize=(8,5))

# plt.plot(h, np.ones(num_of_step_sizes), '-r')
plt.loglog(list_of_number_of_points, error_infnorm, "-o", linewidth=4, markersize=8, label=r"$\infty$-Norm Relative Error")
plt.loglog(list_of_number_of_points, error_2norm, '-o', linewidth=4, markersize=8, label=r"2-Norm Relative Error")

plt.legend()
plt.title("Comparing the Relative Errors of the 2-norm and\nthe $\infty$-norm as the Step Size Decreases")
plt.ylabel(r"$\log \log$error")
plt.xlabel("Number of Grid Points")
# increase number of xticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig("output/problem1_2.svg")
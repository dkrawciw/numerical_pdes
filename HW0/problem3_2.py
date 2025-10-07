import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

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

list_of_number_of_points = list(range(10,10000,1000))
n_points = 30

error_2_list = []
error_inf_list = []

h_list = []

for n_points in list_of_number_of_points:
    x_range = (0, 5)
    delta_x = (x_range[1] - x_range[0]) / (n_points-1)
    forcing_fxn = lambda t: np.cos(t)*np.sin(t)

    h_list.append(delta_x)

    phi = np.zeros(n_points)
    phi[0] = np.sin(x_range[0])
    phi[-1] = np.sin(x_range[1])

    x_vals = np.linspace(x_range[0], x_range[1], n_points)

    # Calculating the forcing vector
    f = forcing_fxn(x_vals[1:(n_points-1)])
    f[0] -= phi[0] / (delta_x**2) + phi[0] / (2*delta_x)
    f[-1] -= phi[-1] / (delta_x**2) - phi[-1] / (2*delta_x)

    # Central differences 2nd derivative
    D_xx = np.zeros((n_points - 2, n_points - 2))
    D_xx += np.eye(n_points - 2, k=0) * -2
    D_xx += np.eye(n_points - 2, k=1)
    D_xx += np.eye(n_points - 2, k=-1)
    D_xx /= (delta_x ** 2)

    # Central differences 1st derivative
    D_x = np.zeros((n_points - 2, n_points - 2))
    D_x += np.eye(n_points - 2, k=1)
    D_x += np.eye(n_points - 2, k=-1) * -1
    D_x = (D_x.T @ np.diag(np.sin(x_vals[1:(n_points-1)]))).T
    D_x /= (2 * delta_x)

    A = csr_matrix(D_xx + D_x + np.eye(n_points - 2))

    phi[1:-1] = spsolve(A, f)

    error_2 = np.linalg.norm(np.sin(x_vals) - phi, ord=2) / np.linalg.norm(np.sin(x_vals), ord=2)
    error_inf = np.linalg.norm(np.sin(x_vals) - phi, ord=np.inf) / np.linalg.norm(np.sin(x_vals), ord=np.inf)

    error_2_list.append(error_2)
    error_inf_list.append(error_inf)

plt.figure(figsize=(8,5))

plt.loglog(list_of_number_of_points, error_inf_list, "-o", linewidth=5, markersize=8, label="$\infty$-Norm Error")
plt.loglog(list_of_number_of_points, error_2_list, "-o", linewidth=5, markersize=8, label="2-Norm Error")

plt.xlabel("Number of Grid Points")
plt.ylabel(r"$\log \log$ Error")
plt.title("Error of the Finite Differences Numerical\nSolution from Problem 3")
plt.legend()

plt.tight_layout()
plt.savefig("output/problem3_2.svg")
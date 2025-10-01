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

n_points = 30
x_range = (0, 5)
delta_x = (x_range[1] - x_range[0]) / n_points
forcing_fxn = lambda t: np.cos(t)*np.sin(t)

phi_0 = np.sin(x_range[0])
phi_N = np.sin(x_range[1])

x_vals = np.linspace(x_range[0], x_range[1], n_points)

# Calculating the forcing vector
f = forcing_fxn(x_vals[1:(n_points-1)])
# f[0] = f[0] - phi_0 / (delta_x**2) - phi_0 / (2*delta_x)
f[-1] = f[-1] - phi_N / (delta_x**2) - phi_N / (2*delta_x)

# f[0] *= np.cos()

# Central differences 2nd derivative
D_xx = np.zeros((n_points - 2, n_points - 2))
D_xx += np.eye(n_points - 2, k=0) * -2
D_xx += np.eye(n_points - 2, k=1)
D_xx += np.eye(n_points - 2, k=-1)
D_xx /= (delta_x ** 2)

# Central differences 1st derivative
D_x = np.zeros((n_points - 2, n_points - 2))
D_x += np.eye(n_points - 2, k=1)
D_x += np.eye(n_points - 2, k=-1)
D_x = D_x @ np.diag(np.sin(x_vals[1:(n_points-1)]))
D_x /= (2 * delta_x)

A = csr_matrix(D_xx + D_x + np.eye(n_points - 2))

phi = np.zeros(n_points)
phi[0] = phi_0
phi[-1] = phi_N

phi[1:-1] = spsolve(A, f)

plt.plot(x_vals, phi, 'o-', label='Numerical Approximation', linewidth=4, markersize=8)
plt.plot(x_vals, np.sin(x_vals), '--', label='Exact Solution', linewidth=4, markersize=8)

plt.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$u(x)$')
plt.tight_layout()
# plt.show()
plt.savefig('output/problem3_1.svg')
import numpy as np
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import seaborn as sns

import pickle as pkl

"""Plot setup"""
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

# Spacial Grid
N = 64
x_range = (0, 2*np.pi)
y_range = (0, 2*np.pi)
x_vals = np.linspace(x_range[0], x_range[1], N)
y_vals = np.linspace(y_range[0], y_range[1], N)
X,Y = np.meshgrid(x_vals, y_vals)
h = x_vals[1] - x_vals[0]

# Create the Laplacian
D = np.zeros((N-2,N-2))
D += np.eye(N - 2, k=0) * -2
D += np.eye(N - 2, k=1)
D += np.eye(N - 2, k=-1)
D /= h**2
D = csr_matrix(D)
L = kron(eye(N-2, format="csr"), D, format="csr") + kron(D, eye(N-2, format="csr"), format="csr")

# Create forcing function
forcing_fxn = lambda x,y,t: np.sin(x)*np.sin(y) * (2*np.cos(10*t) - 10*np.sin(10*t))

def dudt(t,u):
    F = forcing_fxn(X[1:-1,1:-1], Y[1:-1,1:-1], t)
    f = F.ravel(order="F")

    du = L@u + f
    return du

list_of_number_of_points = list(range(4000,10010,100))
error_infnorm = []
error_2norm = []

for N_t in list_of_number_of_points:
    # Temporal Grid
    t_range = [0, 2*np.pi]
    delta_t = (t_range[1] - t_range[0]) / N_t

    U = np.sin(X)*np.sin(Y)
    u0 = U[1:-1,1:-1].ravel(order="F")

    soln = solve_ivp(dudt, t_span=t_range, y0=u0, method="RK45")
    u = soln.y[:,-1]

    U[1:-1,1:-1] = u.reshape((N-2,N-2), order="F")
    exact_soln = np.cos(10*t_range[1])*np.sin(X)*np.sin(Y)

    err_inf = np.linalg.norm(exact_soln.ravel(order="F") - U.ravel(order="F"), ord=np.inf) / np.linalg.norm(exact_soln.ravel(order="F"), ord=np.inf)
    err_2 = np.linalg.norm(exact_soln.ravel(order="F") - U.ravel(order="F"), ord=2) / np.linalg.norm(exact_soln.ravel(order="F"), ord=2)

    error_infnorm.append(err_inf)
    error_2norm.append(err_2)

# Write 2 norm error values to a pickle file
pkl_obj = {
    "name": "ODE45",
    "two_norm": error_2norm,
    "time_points": list_of_number_of_points,
}
with open("output/ODE45.pkl", "wb") as pkl_file:
    pkl.dump(pkl_obj, pkl_file)

plt.figure(figsize=(8,5))

# plt.loglog(list_of_number_of_points, error_infnorm, "-o", linewidth=6, markersize=8, label="$\infty$-Norm Error")
plt.loglog(list_of_number_of_points, error_2norm, "-o", linewidth=4, markersize=8, label="2-Norm Error")
plt.loglog(list_of_number_of_points,  1/np.power(np.array(list_of_number_of_points),2), "k", label=r"Reference $O(n^2)$ Convergence")

plt.xlabel("Number of Time Points")
plt.ylabel(r"$\log \log$ Error")
plt.title("Error Convergence of the Numerical Solution\nUsing ODE45")
plt.legend()

plt.tight_layout()
plt.savefig("output/problem1_ODE45.svg")

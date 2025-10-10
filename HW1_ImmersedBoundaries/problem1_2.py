import numpy as np
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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

n = 40

n_list = list(range(20, 500, 20))
error_2norm = []
error_infnorm = []

for n in tqdm(n_list, desc="Solving Poisson Over Various Grid Sizes"):
    x_range = (0, 2*np.pi)
    y_range = (0, 2*np.pi)
    x_vals = np.linspace(x_range[0], x_range[1], n)
    y_vals = np.linspace(y_range[0], y_range[1], n)
    X,Y = np.meshgrid(x_vals,y_vals, indexing='ij')
    h = x_vals[1] - x_vals[0]


    # Defining that the forcing function has the h^2 term!!!
    forcing_fxn = lambda x, y: -8*np.cos(2*x) * np.sin(2*y)
    F = forcing_fxn(X[1:-1,1:-1],Y[1:-1,1:-1]) * h**2

    # Initialize solution matrix
    U = np.zeros((n,n))

    # Specify boundary conditions
    U[:,0] = X[:,0] * 0
    U[:, -1] = X[:,-1] * 0
    U[0, :] = np.sin(2*Y[0,:])
    U[-1, :] = np.sin(2*Y[-1,:])

    # Apply boundary conditions to the forcing function
    F[0,:] -= U[0,1:-1]
    F[-1,:] -= U[-1,1:-1]
    F[:,0] -= U[1:-1,0]
    F[:,-1] -= U[1:-1,-1]

    # Vectorize matrices for solving
    f = F.ravel(order="F")

    D = np.zeros((n-2,n-2))
    D += np.eye(n - 2, k=0) * -2
    D += np.eye(n - 2, k=1)
    D += np.eye(n - 2, k=-1)

    D = csr_matrix(D)

    L = kron(eye(n-2, format="csr"), D, format="csr") + kron(D, eye(n-2, format="csr"), format="csr")

    u = spsolve(L, f)

    # Reshape u to be a matrix to be easily plottable
    U[1:-1,1:-1] = u.reshape((n-2,n-2), order="F")

    exact_soln = np.cos(2*X) * np.sin(2*Y)

    error_2 = np.linalg.norm(exact_soln.ravel() - U.ravel(), ord=2) / np.linalg.norm(exact_soln.ravel(), ord=2)
    error_inf = np.linalg.norm(exact_soln.ravel() - U.ravel(), ord=np.inf) / np.linalg.norm(exact_soln.ravel(), ord=np.inf)

    error_2norm.append(error_2)
    error_infnorm.append(error_inf)

plt.figure(figsize=(8,5))

plt.loglog(n_list, error_infnorm, "-o", linewidth=4, markersize=8, label=r"$\infty$-Norm Error")
plt.loglog(n_list, error_2norm, "--o", linewidth=4, markersize=8, label="2-Norm Error")
plt.loglog(n_list,  1/np.square(np.array(n_list)), "k", label=r"Reference $O(n^{-2})$ Error")

plt.xlabel("Number of Grid Points")
plt.ylabel("Error")
plt.title("Error Convergence of the Numerical Solution of\nthe Poisson Equation as the number of Grid Points Increases")
plt.legend()

plt.tight_layout()
plt.savefig("output/problem1_2.svg")
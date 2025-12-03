import numpy as np
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

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

"""Numerical Approximation"""
# Space discretization
N = 500
x_range = (0, 5)
y_range = (0, 5)

x_vals = np.linspace(x_range[0], x_range[1], N)
y_vals = np.linspace(y_range[0], y_range[1], N)
X,Y = np.meshgrid(x_vals,y_vals, indexing='ij')
h = X[1,0] - X[0,0]

# Time discretization
N_t = 100
t_range = [0, 50]
dt = (t_range[1] - t_range[0]) / N_t

eps = h

# Defining a random initial condition
np.random.seed(102)
u = eps*np.random.rand(N * N)
u = u - np.mean(u)

D = np.zeros((N,N))
D += np.eye(N, k=0) * -2
D += np.eye(N, k=1)
D += np.eye(N, k=-1)
D /= (h**2)

D = csr_matrix(D)

L = kron(eye(N, format="csr"), D, format="csr") + kron(D, eye(N, format="csr"), format="csr")

LHS = cholesky(eye(N*N, format="csr") - eps**2 * dt * L)

for n in tqdm(range(1,N_t)):
    phi_star = LHS.solve_A(u)
    u = phi_star * np.exp(dt) / np.sqrt(1 + phi_star**2 * (np.exp(2*dt) - 1))

U = u.reshape((N,N), order="F")

plt.figure(figsize=(6,5))
plt.contourf(X, Y, U, levels=50, cmap='icefire')
plt.colorbar(label=r'$u(x,y)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Random Allen Cahn Solution', fontsize=16)

plt.tight_layout()
plt.savefig("output/problem1.svg")
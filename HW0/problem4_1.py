import numpy as np
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import spsolve

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
n = 20
x_range = (0, 5)
y_range = (0, 5)
h = (x_range[1] - x_range[0]) / (n-1)

x_vals = np.linspace(x_range[0], x_range[1], n)
y_vals = np.linspace(y_range[0], y_range[1], n)
X,Y = np.meshgrid(x_vals,y_vals, indexing='ij')

# Defining that the forcing function has the h^2 term!!!
forcing_fxn = lambda x, y: -2*np.sin(x)*np.cos(y)
F = forcing_fxn(X[1:-1,1:-1],Y[1:-1,1:-1]) * h**2

# Initialize solution matrix
U = np.zeros((n,n))

# Specify boundary conditions
U[:,0] = np.sin(X[:,0])                         # u(x,0) = sinx
U[:, -1] = np.sin(X[:,0]) * np.cos(y_range[1])  # u(x,5) = sinx * cos(5)
U[0, :] = np.zeros(n)                           # u(0,y) = 0
U[-1, :] = np.sin(x_range[1]) * np.cos(Y[-1,:]) # u(5,y) = sin(5) * cos(y)

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

plt.figure(figsize=(8,6))
plt.contourf(X, Y, U, levels=50, cmap='viridis')
plt.colorbar(label=r'$u(x,y)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Numerical Approximation to Poisson Equation\nwith Solution $u(x,y) = \sin{x} \cos{y}$', fontsize=16)
plt.tight_layout()
plt.savefig("output/problem4_1.svg")

plt.clf()
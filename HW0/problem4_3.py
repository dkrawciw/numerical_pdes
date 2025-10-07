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
x_vals = np.linspace(x_range[0], x_range[1], n)
y_vals = np.linspace(y_range[0], y_range[1], n)
X,Y = np.meshgrid(x_vals,y_vals, indexing='ij')

h = x_vals[1] - x_vals[0]


# Defining that the forcing function has the h^2 term!!!
forcing_fxn = lambda x, y: -2*np.sin(x)*np.cos(y)
F = forcing_fxn(X,Y) * h**2

# Initialize solution matrix
U = np.zeros((n,n))

# Specify boundary conditions
# U[:,0] = np.sin(X[:,0])                         # u(x,0) = sinx
# U[:, -1] = np.sin(X[:,0]) * np.cos(y_range[1])  # u(x,5) = sinx * cos(5)
# U[0, :] = np.zeros(n)                           # u(0,y) = 0
# U[-1, :] = np.sin(x_range[1]) * np.cos(Y[-1,:]) # u(5,y) = sin(5) * cos(y)

# Apply boundary conditions to the forcing function
F[0,:] += np.sin(X[0,:]) * np.sin(Y[0,:])
F[-1,:] += np.sin(X[-1,:]) * np.sin(Y[-1,:])
F[:,0] += - np.cos(X[:,0]) * np.cos(Y[:,0])
F[:,-1] += - np.cos(X[:,-1]) * np.cos(Y[:,-1])

# Vectorize matrices for solving
f = F.ravel(order="F")

D = np.zeros((n,n))
D += np.eye(n, k=0) * -2
D += np.eye(n, k=1)
D += np.eye(n, k=-1)
D[0,1] = 2
D[n-1,n-2] = 2

D = csr_matrix(D)

L = kron(eye(n, format="csr"), D, format="csr") + kron(D, eye(n, format="csr"), format="csr")

u = spsolve(L, f)

# Reshape u to be a matrix to be easily plottable
U = u.reshape((n,n), order="F")

plt.figure(figsize=(8,5))
plt.contourf(X, Y, U, levels=50, cmap='viridis')
plt.colorbar(label=r'$u(x,y)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Numerical Approximation to Poisson Equation\nwith Solution $u(x,y) = \sin{x} \cos{y}$\nGiven Neumann Boundary Conditions', fontsize=16, pad=18)
plt.tight_layout()
plt.savefig("output/problem4_3.svg")
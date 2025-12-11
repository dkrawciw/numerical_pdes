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
x_range = (-2.5, 2.5)
y_range = (-2.5, 2.5)

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
# np.random.seed(102)
# u = eps*np.random.rand(N * N)
# u = u - np.mean(u)

thetas = np.linspace(0, 2*np.pi, 100, endpoint=False)
inf_x_vals = np.cos(thetas)
inf_y_vals = np.sin(thetas)*np.cos(thetas)

ix = np.digitize(inf_x_vals, x_vals)-1
iy = np.digitize(inf_y_vals, y_vals)-1

np.random.seed(102)
u = eps*np.random.rand(N * N)
u = u - np.mean(u)

U = u.reshape((N, N), order="F")
# U = X*0 + 0.01
brush_radius = 5

for i in range(len(ix)):
    U[ix[i]-brush_radius:ix[i]+brush_radius+1, iy[i]-brush_radius:iy[i]+brush_radius+1] = -0.01 * np.ones((2*brush_radius+1,2*brush_radius+1))

u = U.reshape(N*N, order="F")
phi_0 = u.copy()

D = np.zeros((N,N))
D += np.eye(N, k=0) * -2
D += np.eye(N, k=1)
D += np.eye(N, k=-1)
D /= (h**2)

D = csr_matrix(D)

L = kron(eye(N, format="csr"), D, format="csr") + kron(D, eye(N, format="csr"), format="csr")

LHS = cholesky(eye(N*N, format="csr") - eps**2 * dt * L)

U = u.reshape((N, N), order="F")
areas = [np.trapezoid(np.trapezoid(U, x=x_vals, axis=0), x=y_vals)]

for n in tqdm(range(1,N_t)):
    phi_star = LHS.solve_A(u)
    u_star = phi_star * np.exp(dt) / np.sqrt(1 + phi_star**2 * (np.exp(2*dt) - 1))

    beta = (np.sum(phi_0 - u_star)/np.sum(np.sqrt( (u_star**2 - 1)**2 / 4 ))) / dt
    u = dt*beta*np.sqrt((u_star**2 - 1)**2 / 4) + u_star

    # U = u.reshape((N, N), order="F")
    # areas.append(np.trapezoid(np.trapezoid(U, x=x_vals, axis=0), x=y_vals))

U = u.reshape((N,N), order="F")

# I want to have a side-by-side plot showing the initial condition and the final condition
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Initial Condition Plot
c1 = ax[0].contourf(X, Y, phi_0.reshape((N,N), order="F"), levels=50, cmap='icefire')
fig.colorbar(c1, ax=ax[0], label=r'$u(x,y)$')
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].set_title('Initial Condition', fontsize=16)

# Final Condition Plot
c2 = ax[1].contourf(X, Y, U, levels=50, cmap='icefire')
fig.colorbar(c2, ax=ax[1], label=r'$u(x,y)$')
ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$y$')
ax[1].set_title('Allen Cahn After Time', fontsize=16)

plt.suptitle('Lines Initial Condition', fontsize=18)

plt.tight_layout()
plt.savefig("output/problem3.svg")
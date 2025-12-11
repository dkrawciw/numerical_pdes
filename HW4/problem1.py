import numpy as np
from scipy.sparse import csr_matrix, kron, eye, lil_matrix
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
import seaborn as sns
import pyvista as pv
from tqdm import tqdm
from scipy.spatial.distance import pdist

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

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z / r)       # polar angle
    phi = np.arctan2(y, x)         # azimuth
    return r, theta, phi

def get_FEM_mats(V,T) -> list:
    N_v = V.shape[0]

    S = lil_matrix((N_v, N_v), dtype=float)
    M = lil_matrix((N_v, N_v), dtype=float)

    for k, face in enumerate(T):
        k_1 = face[0]
        k_2 = face[1]
        k_3 = face[2]

        r_1 = V[k_1,:]
        r_2 = V[k_2,:]
        r_3 = V[k_3,:]

        E_1 = r_3 - r_2
        E_2 = r_1 - r_3
        E_3 = r_2 - r_1

        triangle_area = np.linalg.norm(np.cross(E_1, E_2), ord=2) / 2

        S_k = np.array([
            [E_1@E_1, E_1@E_2, E_1@E_3],
            [E_2@E_1, E_2@E_2, E_2@E_3],
            [E_3@E_1, E_3@E_2, E_3@E_3],
        ])
        S_k /= (4*triangle_area)

        M_k = np.ones((3,3))
        M_k += np.eye(3)
        M_k *= triangle_area / 12

        for col_ind in range(3):
            for row_ind in range(3):
                S[face[row_ind], face[col_ind]] += S_k[row_ind, col_ind]
                M[face[row_ind], face[col_ind]] += M_k[row_ind, col_ind]
    
    return S.tocsr(), M.tocsr()

forcing_fxn = lambda theta, phi: 2*np.cos(theta)
exact_soln_fxn = lambda theta, phi: np.cos(theta)

# num_divisions_list = list(range(1,6))
num_res_list = list(range(0,7))
num_verts = []
err_list = []
h_vals = []

for num_res in tqdm(num_res_list, desc=f"Solving on a mesh with different numbers of vertices"):
    # Create the Sphere
    sphere = pv.Icosphere(nsub=num_res)
    # sphere = pv.Sphere(theta_resolution=num_res, phi_resolution=num_res)
    T = sphere.faces.reshape((-1, 4))[:, 1:]   # drop the leading 3
    V = sphere.points
    N_v = V.shape[0]           # Number of Vertices
    _, THETA, PHI = cartesian_to_spherical(V[:,0], V[:,1], V[:,2])
    # THETA += np.pi/2

    # Given conditions: exact_soln and the forcing function to get that solution
    F = forcing_fxn(THETA, PHI)
    exact_sol = exact_soln_fxn(THETA, PHI)

    S, M = get_FEM_mats(V, T)

    eps=1e-8
    u = spsolve(S+eps*eye(N_v, format="csr"), M@F)

    diff = exact_sol - u
    err = np.sqrt(diff @ M @ diff)

    num_verts.append(N_v)
    err_list.append(err)
    h_vals.append(min(pdist(V)))

plt.figure(figsize=(8,5))

num_verts = np.array(num_verts)
h_vals = np.array(h_vals)

plt.loglog(h_vals, err_list, "-o", linewidth=4, markersize=8, label="2-Norm Absolute Error")
plt.loglog(h_vals, h_vals**2, "k", label=r"Reference $O(n^2)$ Convergence")

plt.title("Error of Solving the Poisson Equation on a Sphere\nMesh as the Triangle Edge Length Decreases")
plt.ylabel(r"$\log \log$ Error")
plt.xlabel(r"Minimum Edge Length $(h)$")

plt.legend()
plt.tight_layout()
plt.savefig("output/problem1_errorplot.svg")
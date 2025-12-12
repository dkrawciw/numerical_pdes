import numpy as np
from scipy.sparse import csr_matrix, kron, eye, lil_matrix
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
import seaborn as sns
import pyvista as pv
from tqdm import tqdm
from scipy.spatial.distance import pdist
from sksparse.cholmod import cholesky

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

mesh_obj = pv.Icosphere(radius=1.0, nsub=3)
# bunny = pv.examples.download_bunny()

T = mesh_obj.faces.reshape((-1, 4))[:, 1:]
V = mesh_obj.points

N_v = V.shape[0]

N_t = 10000
t_range = [0,1]
delta_t = (t_range[1] - t_range[0])/N_t
eps = 1e-4

U = np.zeros((V.shape[0],V.shape[1],N_t))
U[:,:,0] = V
U[:,:,1] = V

U[0,1,0] += -0.3     # Initial condition
U[1,1,0] += -0.3     # Initial condition
U[2,1,0] += -0.3     # Initial condition

c = 2  # Wave speed


# Use original mesh for FEM matrices
S, M = get_FEM_mats(V, T)
LHS = M 
LHS = cholesky(LHS)

for i in tqdm(range(1,N_t-1), desc="Evolving the mesh in time"):
    U_now = U[:, :, i]               # shape (N_v, 3)
    U_prev = U[:, :, i-1]

    RHS = (2*M - c**2 * delta_t * S)@U_now - M@U_prev

    X_new = np.zeros_like(U_now)
    for d in range(3):
        # X_new[:, d] = spsolve(LHS, RHS[:, d])
        X_new[:, d] = LHS(RHS[:, d])
    U[:, :, i+1] = X_new

initial_pts = U[:, :, 0]
middle_pts  = U[:, :, 100]
final_pts   = U[:, :, 200]

faces = np.hstack([np.full((T.shape[0], 1), 3), T]).astype(np.int32).ravel()

# Build meshes
mesh_initial = pv.PolyData(initial_pts, faces=faces)
mesh_middle  = pv.PolyData(middle_pts, faces=faces)
mesh_final   = pv.PolyData(final_pts, faces=faces)

# Create 3 side-by-side subplots
plotter = pv.Plotter(shape=(1, 3), border=True)

# --- Left: initial ---
plotter.subplot(0, 0)
plotter.add_mesh(mesh_initial, color="lightgray", show_edges=True, culling=False)
plotter.add_text("Initial", font_size=12)
plotter.camera_position = 'iso'

# --- Middle: middle timestep ---
plotter.subplot(0, 1)
plotter.add_mesh(mesh_middle, color="lightblue", show_edges=True, culling=False)
plotter.add_text("Middle", font_size=12)
plotter.camera_position = 'iso'

# --- Right: final ---
plotter.subplot(0, 2)
plotter.add_mesh(mesh_final, color="salmon", show_edges=True, culling=False)
plotter.add_text("Final", font_size=12)
plotter.camera_position = 'iso'

plotter.link_views()   # all three rotate together
plotter.show()
plotter.screenshot('output/wave_equation_evolution.png', window_size=[1800, 600])

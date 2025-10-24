import numpy as np
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import spsolve, gmres, LinearOperator

import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D  # Not strictly required in newer versions

"""Plot setup"""
sns.set_style("whitegrid")
sns.set_color_codes(palette="colorblind")

plt.rcParams.update({
	"text.usetex": False,
	"mathtext.fontset": "cm",
	"font.family": "serif",
	"font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})


"""Spreading and Interpolation Functions"""
def spreadQ(
        X: np.array, 
        Y: np.array,
        xq:np.array,
        yq: np.array,
        q: np.array,
        delta,
        ) -> np.array:
    
    """
    # Spread Q

    This is step one of solving an immersed boundary problem. You spread the effects from the immersed boundary to the background fluid.
    """
    Sq = 0*X
    num_q = xq.shape[0]
    for k in range(num_q):
        Rk = np.sqrt(np.square((X-xq[k])) + np.square((Y-yq[k])))
        Sq += q[k] * delta(Rk)
    
    return Sq

def interpPhi(
        X: np.array,
        Y: np.array,
        xq: np.array,
        yq: np.array,
        Phi: np.array,
        delta,
        ) -> np.array:
    """
    # Interp Phi

    This is step 2 of solving an immersed boundary method. From the background fluid, interpolate the forces onto the immersed boundary points.
    """
    
    Jphi = 0 * xq
    num_q = xq.shape[0]
    dx = X[1,0] - X[0,0]
    dy = Y[0,1] - Y[0,0]
    
    for k in range(num_q):
        #Rk = np.sqrt( np.square(X - xq[k]) + np.square(Y - yq[k]) )
        Rk = np.sqrt(np.square((X-xq[k])) + np.square((Y-yq[k])))
        # Wx = delta(X - xq[k])
        # Wy = delta(Y - yq[k])
        Wr = delta(Rk)
        Jphi[k] = dx * dy * np.sum(Phi * Wr)
    
    return Jphi

"""Numerical Approximation"""
n = 50
x_range = (0, 2*np.pi)
y_range = (0, 2*np.pi)
x_vals = np.linspace(x_range[0], x_range[1], n+2)
y_vals = np.linspace(y_range[0], y_range[1], n+2)

X_ext,Y_ext = np.meshgrid(x_vals,y_vals, indexing='ij')
X,Y = np.meshgrid(x_vals[1:n+1],y_vals[1:n+1], indexing='ij')

h = x_vals[1] - x_vals[0]

# Defining that the forcing function has the h^2 term!!!
forcing_fxn = lambda x, y: 0*x*y
F = forcing_fxn(X,Y)

# Initialize solution matrix
U = np.zeros((n+2,n+2))

# Specify boundary conditions
U[:,0] = X_ext[:,0] * 0
U[:, -1] = X_ext[:,-1] * 0
U[0, :] = 0*Y_ext[0,:]
U[-1, :] = 0*Y_ext[-1,:]

# Apply boundary conditions to the forcing function
F[0,:] -= U[0,1:-1] / h**2
F[-1,:] -= U[-1,1:-1] / h**2
F[:,0] -= U[1:-1,0] / h**2
F[:,-1] -= U[1:-1,-1] / h**2

# Vectorize matrices for solving
f = F.ravel(order="F")

D = np.zeros((n,n))
D += np.eye(n, k=0) * -2
D += np.eye(n, k=1)
D += np.eye(n, k=-1)
D /= h**2

L = kron(eye(n, format="csr"), D, format="csr") + kron(D, eye(n, format="csr"), format="csr")

# Immersed Boundary Piece
num_charges = n-1
thetas = np.linspace(0, 2*np.pi, num_charges, endpoint=False)
x_ib = np.pi + np.cos(thetas)
y_ib = np.pi + np.sin(thetas)
charge_h = np.sqrt((x_ib[1] - x_ib[0])**2 + (y_ib[1] - y_ib[0])**2)
internal_bc = np.cos(2*x_ib) * np.sin(2*y_ib)

# Defining a delta function
def delta(r, sigma = 1.2*h):
    return (1/(2*np.pi*sigma**2))*np.exp(-0.5*np.square(r/sigma))

def Amult(x) -> np.array:
    u = x[0:n**2]
    q = x[n**2::]

    Q = spreadQ(X,Y,x_ib,y_ib,q,delta)
    Phi = u.reshape((n,n), order="F")

    top = L@u - Q.ravel(order="F")
    bottom = interpPhi(X,Y,x_ib,y_ib,Phi,delta)

    return np.concatenate([top, bottom])

# build LinearOperator
def make_LinearOp():
    n_op = (n)**2 + num_charges
    return LinearOperator((n_op, n_op), matvec=Amult, dtype=np.float64)

A = make_LinearOp()
b = np.concatenate([f, internal_bc])
soln, exit_code = gmres(A, b, rtol=0.01*(h**2))

u = soln[0:n**2]
q = soln[n**2::]

U[1:-1,1:-1] = u.reshape((n,n), order="F")

plt.figure(figsize=(8,7))
mappable = plt.contourf(X_ext, Y_ext, U, levels=50, cmap='icefire')
plt.scatter(x_ib, y_ib, c="yellow", label="Immersed Boundary Points")

plt.colorbar(label=r'$u(x,y)$', mappable=mappable)
plt.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('The *Almost* Minimal Surface', fontsize=16)
plt.tight_layout()

# plt.show()
plt.savefig("output/problem3_1_2D.svg")

plt.clf()

plt.figure(figsize=(8,7))
plt.plot(thetas, q, "--o", label="Immersed Boundary", linewidth=6, markersize=10)

plt.xlabel(r'$\theta$ $[0,2 \pi]$', fontsize=16)
plt.ylabel('Immersed Boundary Charge (Coulombs)', fontsize=16)
plt.title("The Charge of the Immersed Boundary Points\nfrom Solving for the Almost Minimal Surface", fontsize=20)

plt.tight_layout()
plt.legend()
plt.savefig("output/problem3_1_scatter.svg")

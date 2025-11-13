import numpy as np
from scipy.sparse import csr_matrix, kron, eye
from sksparse.cholmod import cholesky

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

# Spacial Grid
N = 64
x_range = (0, 2*np.pi)
y_range = (0, 2*np.pi)
x_vals = np.linspace(x_range[0], x_range[1], N)
y_vals = np.linspace(y_range[0], y_range[1], N)
X,Y = np.meshgrid(x_vals, y_vals)
h = X[0,1] - X[0,0]

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

U = np.sin(X)*np.sin(Y)
u = U[1:-1,1:-1].ravel(order="F")

# Defining t_range
t_range = (0, 2*np.pi)

exact_soln = np.cos(10*t_range[1])*np.sin(X)*np.sin(Y)

def forward_euler(N_t: int):
    delta_t = (t_range[1] - t_range[0]) / N_t
    soln = np.tile(U, (N_t, 1, 1))

    u = soln[0,1:-1,1:-1].ravel(order="F")

    for i in range(N_t-1):
        t = delta_t * i

        F = forcing_fxn(X[1:-1,1:-1], Y[1:-1,1:-1], t)
        f = F.ravel(order="F")

        u = (delta_t * L + eye((N-2)**2, format="csr"))@u + delta_t*f
        soln[i+1,1:-1,1:-1] = u.reshape((N-2,N-2), order="F")
    
    return soln

def backward_euler(N_t: int):
    delta_t = (t_range[1] - t_range[0]) / N_t
    soln = np.tile(U, (N_t, 1, 1))
    u = soln[0,1:-1,1:-1].ravel(order="F")

    LHS = cholesky(eye((N-2)**2, format="csr") - delta_t * L)

    # Backward Euler
    for i in range(N_t-1):
        t = delta_t * i

        F = forcing_fxn(X[1:-1,1:-1], Y[1:-1,1:-1], t)
        f = F.ravel(order="F")

        # LHS = eye((N-2)**2, format="csr") - delta_t * L
        RHS = u + delta_t*f
        # u = spsolve(LHS, RHS)
        u = LHS(RHS)
        soln[i+1,1:-1,1:-1] = u.reshape((N-2,N-2), order="F")
    
    return soln

def rk2(N_t: int):
    delta_t = (t_range[1] - t_range[0]) / N_t
    soln = np.tile(U, (N_t, 1, 1))
    u = soln[0,1:-1,1:-1].ravel(order="F")

    for i in range(N_t-1):
        t = delta_t * i

        # Step 1
        F = forcing_fxn(X[1:-1,1:-1], Y[1:-1,1:-1], t)
        f = F.ravel(order="F")
        k1 = L@u + f
        
        u_half = u + (delta_t / 2) * k1

        # Step 2
        F_half = forcing_fxn(X[1:-1,1:-1], Y[1:-1,1:-1], t + delta_t/2)
        f_half = F_half.ravel(order="F")
        k2 = L@u_half + f_half

        u = u + delta_t * k2
        soln[i+1,1:-1,1:-1] = u.reshape((N-2,N-2), order="F")
    
    return soln

def crank_nicolson(N_t: int):
    delta_t = (t_range[1] - t_range[0]) / N_t
    soln = np.tile(U, (N_t, 1, 1))
    u = soln[0,1:-1,1:-1].ravel(order="F")

    # Crank-Nicolson
    LHS = cholesky(eye((N-2)**2, format="csr") - delta_t/2 * L)

    for i in range(N_t-1):
        t = delta_t * i

        F_now = forcing_fxn(X[1:-1,1:-1], Y[1:-1,1:-1], t)
        F_next = forcing_fxn(X[1:-1,1:-1], Y[1:-1,1:-1], t+delta_t)
        F = (F_next + F_now)/2
        f = F.ravel(order="F")

        # LHS = eye((N-2)**2, format="csr") - delta_t * L
        RHS = (eye((N-2)**2, format="csr") + delta_t/2 * L)@u + delta_t*f
        # u = spsolve(LHS, RHS)
        u = LHS(RHS)
        soln[i+1,1:-1,1:-1] = u.reshape((N-2,N-2), order="F")
    
    return soln

def bdf2(N_t: int):
    delta_t = (t_range[1] - t_range[0]) / N_t
    soln = np.tile(U, (N_t, 1, 1))
    u = soln[0,1:-1,1:-1].ravel(order="F")

    """Before BDF2, I need to run RK2 once"""
    # Step 1
    F_now = forcing_fxn(X[1:-1,1:-1], Y[1:-1,1:-1], 0)
    f = F_now.ravel(order="F")
    k1 = L@u + f

    u_half = u + (delta_t / 2) * k1

    # Step 2
    F_half = forcing_fxn(X[1:-1,1:-1], Y[1:-1,1:-1], delta_t/2)
    f = F_half.ravel(order="F")
    k2 = L@u_half + f

    u = u + delta_t * k2
    soln[1,1:-1,1:-1] = u.reshape((N-2,N-2), order="F")

    """After running RK2, we now have a history point to use for BDF2"""
    LHS = cholesky(eye((N-2)**2, format="csr") - 2/3* delta_t * L)

    # BDF2
    for i in range(1,N_t-1):
        t = delta_t * i
        u_hist = soln[i-1, 1:-1, 1:-1].ravel(order="F")
        u_curr = soln[i, 1:-1, 1:-1].ravel(order="F")

        F = forcing_fxn(X[1:-1,1:-1], Y[1:-1,1:-1], t + delta_t)
        f = F.ravel(order="F")

        # LHS = eye((N-2)**2, format="csr") - delta_t * L
        RHS = 4/3 * u_curr - 1/3 * u_hist + 2/3 * delta_t*f
        # u = spsolve(LHS, RHS)
        u = LHS(RHS)
        soln[i+1,1:-1,1:-1] = u.reshape((N-2,N-2), order="F")

    return soln

def optimal_num_points(time_stepper, suff_err: float, starting_point: int = 2500, max_num: int = 1e16):
    not_reached = True
    prev_num_points = 0
    curr_num_points = starting_point // 2
    err = np.inf

    # while err > suff_err:
    #     prev_num_points = curr_num_points
    #     curr_num_points *= 2
    #     soln = time_stepper(curr_num_points)
    #     U_approx = soln[-1,:,:]

    #     err = np.linalg.norm(exact_soln.ravel(order="F") - U_approx.ravel(order="F"), ord=2) / np.linalg.norm(exact_soln.ravel(order="F"), ord=2)
    #     print(f"{curr_num_points} - {err}")

    #     if max_num <= curr_num_points:
    #         return [np.inf, np.inf]

    while not_reached:
        soln = time_stepper(curr_num_points)
        U_approx = soln[-1,:,:]

        err = np.linalg.norm(exact_soln.ravel(order="F") - U_approx.ravel(order="F"), ord=2) / np.linalg.norm(exact_soln.ravel(order="F"), ord=2)
        print(f"{curr_num_points} - {err}")

        pt_diff = curr_num_points - prev_num_points
        prev_num_points = curr_num_points

        if err >= suff_err:
            curr_num_points += abs(pt_diff) // 2
        else:
            curr_num_points -= abs(pt_diff) // 2
    
        if abs(pt_diff) < 3:
            not_reached = False
    
    return [curr_num_points, err]
            
suff_errs = [5e-2, 1e-3, 5e-6]
suff_err = 1e-3

# for suff_err in suff_errs:
print(f"To achieve an error of {suff_err}")

# fe_num_points, fe_err = optimal_num_points(time_stepper=forward_euler, suff_err=suff_err, starting_point=1600000)
# print(f"Forward Euler: {fe_num_points} points - error: {fe_err}")

# be_num_points, be_err = optimal_num_points(time_stepper=backward_euler, suff_err=suff_err, starting_point=20000)
# print(f"Backward Euler: {be_num_points} points - error: {be_err}")

# cn_num_points, cn_err = optimal_num_points(time_stepper=crank_nicolson, suff_err=suff_err, starting_point=7200)
# print(f"Crank-Nicolson: {cn_num_points} points - error: {cn_err}")

rk_num_points, rk_err = optimal_num_points(time_stepper=rk2, suff_err=suff_err, starting_point=402000)
print(f"RK2: {rk_num_points} points - error: {rk_err}")

# bdf2_num_points, bdf2_err = optimal_num_points(time_stepper=bdf2, suff_err=suff_err, starting_point=1750)
# print(f"BDF2: {bdf2_num_points} points - error: {bdf2_err}")
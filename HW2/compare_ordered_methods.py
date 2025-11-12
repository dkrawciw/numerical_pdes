import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import numpy as np
from pathlib import Path

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

OUTPUT_DIR = Path("output/")
pkl_files = list(OUTPUT_DIR.glob('*.pkl'))

first_order_files = [
    "output/ForwardEuler.pkl",
    "output/BackwardEuler.pkl",
]

second_order_files = [
    "output/CrankNicolson.pkl",
    "output/rk2.pkl",
    "output/BDF2.pkl",
]

plt.figure(figsize=(8,6))

for i, file in enumerate(first_order_files):
    with open(file, 'rb') as pkl_file:
        pkl_obj = pkl.load(pkl_file)
    
    label = pkl_obj['name']
    err_points = pkl_obj['two_norm']
    time_points = pkl_obj['time_points']

    marker_style = "-o"

    plt.loglog(time_points, err_points, marker_style, linewidth=3, markersize=5, label=label)

with open(pkl_files[0], 'rb') as pkl_file:
    pkl_obj = pkl.load(pkl_file)

time_vals = np.array(pkl_obj['time_points'])


"""First Plot"""
plt.loglog(time_vals, 1/(time_vals), "k", label=r"Reference $O(n)$ Convergence")

plt.title("Comparing the Error Convergence of First Order Methods")
plt.xlabel("Number of Discrete Time Values Used")
plt.ylabel(r"$\log \log$ Error")
plt.legend()

plt.tight_layout()
plt.savefig("output/compare_first_order_methods.svg")


"""Second Plot"""
plt.clf()  # Clear the current figure

for i, file in enumerate(second_order_files):
    with open(file, 'rb') as pkl_file:
        pkl_obj = pkl.load(pkl_file)
    
    label = pkl_obj['name']
    err_points = pkl_obj['two_norm']
    time_points = pkl_obj['time_points']

    # if i % 2 == 0:
    #     marker_style = "--o"
    # else:
    marker_style = "-o"

    plt.loglog(time_points, err_points, marker_style, linewidth=3, markersize=5, label=label)

plt.loglog(time_vals, 1/(time_vals**2), "k", label=r"Reference $O(n^2)$ Convergence")

plt.title("Comparing the Error Convergence of Second Order Methods")
plt.xlabel("Number of Discrete Time Values Used")
plt.ylabel(r"$\log \log$ Error")
plt.legend()

plt.tight_layout()
plt.savefig("output/compare_second_order_methods.svg")
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

plt.figure(figsize=(8,6))

for i, file in enumerate(pkl_files):
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

with open(pkl_files[0], 'rb') as pkl_file:
    pkl_obj = pkl.load(pkl_file)

time_vals = np.array(pkl_obj['time_points'])

plt.loglog(time_vals, 1/(time_vals**2), "k", label=r"Reference $O(n^2)$ Convergence")

plt.title("Comparing the Error Convergence of\nDifferent Time Stepping Methods")
plt.xlabel("Number of Discrete Time Values Used")
plt.ylabel(r"$\log \log$ Error")
plt.legend()

plt.tight_layout()
plt.savefig("output/compare_all_methods.svg")
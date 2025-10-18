import matplotlib.pyplot as plt
import numpy as np
from utils import *


def plot_distributions_Y(graph_family, p, n, distributions, t_functions):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, (name, f) in enumerate(t_functions[:4]):
        Y = distributions[name]
        t = min(f(n, p), n)
        row, col = positions[idx]
        ax = axes[row, col]

        values, counts = np.unique(Y, return_counts=True)
        probs = counts / np.sum(counts)
        mean_val = np.mean(Y)

        ax.bar(values, probs, color="skyblue", edgecolor="black")
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2)
        ax.set_title(f"t={name} (t={t})", fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"./img/{graph_family}/PMF_Y_p{p}_n{n}.png", dpi=150)
    plt.close()


def plot_expectations_Y(graph_family, p, expectations_data, t_functions):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, (name, f) in enumerate(t_functions[:4]):
        data = expectations_data[name]
        n_values = data['n_values']
        t_values = data['t_values']
        means = data['means']

        row, col = positions[idx]
        ax = axes[row, col]
        
        ax.plot(n_values, means, marker='o', color="darkcyan")

        theoretical = theoretical_expectation_Y(graph_family, np.array(n_values), p, t_values)
        ax.plot(n_values, theoretical, '--', color="darkcyan", alpha=0.7)
        ax.set_title(f"t = {name}", fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"./img/{graph_family}/EY_p{p}.png", dpi=150)
    plt.close()


def plot_distribution_Z(Z_samples, graph_family, p, n):
    plt.figure(figsize=(8, 5))
    
    n_bins = min(50, int(np.sqrt(len(Z_samples))))
    plt.hist(Z_samples, bins=n_bins, color='crimson', edgecolor='black', 
             density=True, alpha=0.8)
    
    mean_z = np.mean(Z_samples)
    plt.axvline(mean_z, color='darkred', linestyle='--', linewidth=2.5)

    plt.grid(alpha=0.3, axis='y')
    plt.savefig(f"./img/{graph_family}/PMF_Z_p{p}_n{n}.png", dpi=150)
    plt.close()


def plot_expectation_Z(graph_family, p, n_values, means):
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, means, 'o-', color='deeppink')

    theoretical = theoretical_expectation_Z(graph_family, np.array(n_values), p)
    plt.plot(n_values, theoretical, '--', color="darkcyan", alpha=0.7)

    plt.grid(alpha=0.3)
    plt.savefig(f"./img/{graph_family}/EZ_p{p}.png", dpi=150)
    plt.close()
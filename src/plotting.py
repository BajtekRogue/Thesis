import matplotlib.pyplot as plt
import numpy as np
from utils import *


def plot_Yt_distribution_SI(graph_family, distribution, n, t, p):
    values, counts = np.unique(distribution, return_counts=True)
    empirical_probs = counts / np.sum(counts)
    theoretical_probs = np.array([theoretical_probs_Yt_SI(graph_family, k, n, t, p) for k in values])

    plt.figure(figsize=(8, 5))
    plt.bar(values, empirical_probs, color="blue", alpha=0.7, label="Symulacja")
    plt.plot(values, theoretical_probs, '-', color="indigo", linewidth=2.2, label="Teoria")
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("pmf $Y_t$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./img/SI/{graph_family}/Yt_distribution.png", dpi=150)
    plt.close()


def plot_Yt_expectation_SI(graph_family, t_values, means, theoretical):
    plt.figure(figsize=(8, 5))
    plt.plot(t_values, means, 'o-', color="darkcyan",  linewidth=2, label='Symulacja', markersize=6)
    plt.plot(t_values, theoretical, '--', color="orange", linewidth=2, alpha=0.8, label='Teoria')
    plt.xlabel("t")
    plt.ylabel("E[$Y_t$]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./img/SI/{graph_family}/Yt_expectation.png", dpi=150)
    plt.close()


def plot_Z_distribution_SI(graph_family, samples, n, p):
    values, counts = np.unique(samples, return_counts=True)
    empirical_probs = counts / np.sum(counts)
    
    # Determine if we need binning
    num_unique = len(values)
    max_bars = 50  # Maximum number of bars to display
    
    if num_unique > max_bars:
        # Create bins
        num_bins = min(max_bars, num_unique // 2)
        bin_edges = np.linspace(values.min(), values.max() + 1, num_bins + 1)
        
        # Bin the empirical data
        binned_counts, _ = np.histogram(samples, bins=bin_edges)
        binned_probs = binned_counts / len(samples)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Compute theoretical probabilities for bins
        theoretical_binned = np.zeros(num_bins)
        for k in values:
            bin_idx = np.searchsorted(bin_edges[1:], k)
            theoretical_binned[bin_idx] += theoretical_probs_Z_SI(graph_family, k, n, p)
        
        # Plot
        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers, binned_probs, width=bin_width * 0.9, 
                color='darkgreen', alpha=0.7, label="Symulacja")
        plt.plot(bin_centers, theoretical_binned, '-', color="darkred", 
                 linewidth=2.2, markersize=4, label="Teoria")
    else:
        # Original plotting for small number of values
        theoretical_probs = np.array([theoretical_probs_Z_SI(graph_family, k, n, p) 
                                      for k in values])
        plt.figure(figsize=(8, 5))
        plt.bar(values, empirical_probs, color='darkgreen', alpha=0.7, label="Symulacja")
        plt.plot(values, theoretical_probs, '-', color="darkred", 
                 linewidth=2.2, markersize=3, label="Teoria")
    
    plt.legend()
    plt.xlabel('k')
    plt.ylabel("pmf Z")
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"./img/SI/{graph_family}/Z_distribution.png", dpi=150)
    plt.close()



def plot_Z_expectation_SI(graph_family, n_values, means, theoretical):
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, means, 'o-', color='deeppink', linewidth=2, markersize=6, label='Symulacja')
    plt.plot(n_values, theoretical, '--', color="forestgreen", linewidth=2, alpha=0.7, label='Teoria')
    plt.xlabel('n')
    plt.ylabel("E[Z]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./img/SI/{graph_family}/Z_expectation.png", dpi=150)
    plt.close()
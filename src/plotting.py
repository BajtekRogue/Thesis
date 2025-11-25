import matplotlib.pyplot as plt
import numpy as np
from utils import *


def plot_Yt_distribution_SI(graph_family, distribution, n, t, p, title):
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
    plt.savefig(f"./img/SI/{graph_family}/{title}.png", dpi=150)
    plt.close()


def plot_Yt_expectation_SI(graph_family, t_values, means, theoretical, title):
    plt.figure(figsize=(8, 5))
    plt.plot(t_values, means, 'o-', color="darkcyan",  linewidth=2, label='Symulacja', markersize=6)
    plt.plot(t_values, theoretical, '--', color="orange", linewidth=2, alpha=0.8, label='Teoria')
    plt.xlabel("t")
    plt.ylabel("E[$Y_t$]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./img/SI/{graph_family}/{title}.png", dpi=150)
    plt.close()


def plot_Z_distribution_SI(graph_family, samples, n, p, title):
    values, counts = np.unique(samples, return_counts=True)
    empirical_probs = counts / np.sum(counts)
    theoretical_probs = np.array([theoretical_probs_Z_SI(graph_family, k, n, p) for k in values])

    plt.figure(figsize=(8, 5))
    plt.bar(values, empirical_probs, color="darkgreen", alpha=0.7, label="Symulacja")
    plt.plot(values, theoretical_probs, '-', color="darkred", linewidth=2.2, label="Teoria")
    plt.legend()
    plt.xlabel('k')
    plt.ylabel("pmf Z")
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"./img/SI/{graph_family}/{title}.png", dpi=150)
    plt.close()


def plot_Z_expectation_SI(graph_family, n_values, means, theoretical, title):
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, means, 'o-', color='deeppink', linewidth=2, markersize=6, label='Symulacja')
    plt.plot(n_values, theoretical, '--', color="forestgreen", linewidth=2, alpha=0.7, label='Teoria')
    plt.xlabel('n')
    plt.ylabel("E[Z]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./img/SI/{graph_family}/{title}.png", dpi=150)
    plt.close()


def plot_W_distribution_SIR(graph_family, distribution, n, p, a, title):
    values, counts = np.unique(distribution, return_counts=True)
    empirical_probs = counts / np.sum(counts)
    theoretical_probs = np.array([theoretical_probs_W_SIR(graph_family, k, n, p, a) for k in values])

    plt.figure(figsize=(8, 5))
    plt.bar(values, empirical_probs, color="blue", alpha=0.7, label="Symulacja")
    plt.plot(values, theoretical_probs, '-', color="indigo", linewidth=2.2, label="Teoria")
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("pmf $W$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./img/SIR/{graph_family}/{title}.png", dpi=150)
    plt.close()


def plot_W_expectation_SIR(graph_family, n_values, means, theoretical, title):
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, means, 'o-', color="darkcyan",  linewidth=2, label='Symulacja', markersize=6)
    plt.plot(n_values, theoretical, '--', color="orange", linewidth=2, alpha=0.8, label='Teoria')
    plt.xlabel("n")
    plt.ylabel("E[W]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./img/SIR/{graph_family}/{title}.png", dpi=150)
    plt.close()


def plot_Z_distribution_SIR(graph_family, samples, n, p, a, title):
    values, counts = np.unique(samples, return_counts=True)
    empirical_probs = counts / np.sum(counts)
    theoretical_probs = np.array([theoretical_probs_Z_SIR(graph_family, k, n, p, a) for k in values])

    plt.figure(figsize=(8, 5))
    plt.bar(values, empirical_probs, color="darkgreen", alpha=0.7, label="Symulacja")
    plt.plot(values, theoretical_probs, '-', color="darkred", linewidth=2.2, label="Teoria")
    plt.legend()
    plt.xlabel('k')
    plt.ylabel("pmf Z")
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"./img/SIR/{graph_family}/{title}.png", dpi=150)
    plt.close()


def plot_Z_expectation_SIR(graph_family, n_values, means, theoretical, title):
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, means, 'o-', color='deeppink', linewidth=2, markersize=6, label='Symulacja')
    plt.plot(n_values, theoretical, '--', color="forestgreen", linewidth=2, alpha=0.7, label='Teoria')
    plt.xlabel('n')
    plt.ylabel("E[Z]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./img/SIR/{graph_family}/{title}.png", dpi=150)
    plt.close()



def plot_Yt_distribution_SIS(distribution, n, t, p, a, folder, title):
    values, counts = np.unique(distribution, return_counts=True)
    empirical_probs = counts / np.sum(counts)
    
    P = P_matrix_SIS(n, p, a)
    y_t = y_t_vector_SIS(n, t, P)
    theoretical_probs = np.array([theoretical_probs_Yt_SIS(k, y_t) for k in values])

    plt.figure(figsize=(8, 5))
    plt.bar(values, empirical_probs, color="blue", alpha=0.7, label="Symulacja")
    plt.plot(values, theoretical_probs, '-', color="indigo", linewidth=2.2, label="Teoria")
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("pmf $Y_t$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./img/SIS/{folder}/{title}.png", dpi=150)
    plt.close()


def plot_Yt_expectation_SIS(t_values, means, theoretical, folder, title):
    plt.figure(figsize=(8, 5))
    plt.plot(t_values, means, 'o-', color="darkcyan",  linewidth=2, label='Symulacja', markersize=6)
    plt.plot(t_values, theoretical, '--', color="orange", linewidth=2, alpha=0.8, label='Teoria')
    plt.xlabel("t")
    plt.ylabel("E[$Y_t$]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./img/SIS/{folder}/{title}.png", dpi=150)
    plt.close()


def plot_Z_distribution_SIS(samples, n, p, a, folder, title):
    values, counts = np.unique(samples, return_counts=True)
    n_bins=20
    min_unique_for_binning=50

    if n_bins is not None and len(values) > min_unique_for_binning:
        min_val, max_val = values.min(), values.max()
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)
            
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        
        empirical_binned = np.zeros(n_bins)
        for val, count in zip(values, counts):
            bin_idx = np.searchsorted(bin_edges[1:], val, side='left')
            bin_idx = min(bin_idx, n_bins - 1)
            empirical_binned[bin_idx] += count
        empirical_probs = empirical_binned / np.sum(empirical_binned)
        
        P = P_matrix_SIS(n, p, a)
        theoretical_probs = np.zeros(n_bins)
        
        for bin_idx in range(n_bins):
            bin_start = int(np.floor(bin_edges[bin_idx]))
            bin_end = int(np.ceil(bin_edges[bin_idx + 1]))
            
            survival_start = theoretical_survival_Z_SIS(bin_start - 1, n, P) if bin_start > 0 else 1.0
            survival_end = theoretical_survival_Z_SIS(bin_end, n, P)
            theoretical_probs[bin_idx] = survival_start - survival_end
        
        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers, empirical_probs, width=bin_widths * 0.8, color="darkgreen", alpha=0.7, label="Symulacja")
        plt.plot(bin_centers, theoretical_probs, '-o', color="darkred", linewidth=2.2, markersize=4, label="Teoria")
        
    else:
        empirical_probs = counts / np.sum(counts)
        
        P = P_matrix_SIS(n, p, a)
        theoretical_probs = np.array([theoretical_probs_Z_SIS(k, n, P) for k in values])
        
        plt.figure(figsize=(8, 5))
        plt.bar(values, empirical_probs, color="darkgreen", alpha=0.7, label="Symulacja")
        plt.plot(values, theoretical_probs, '-o', color="darkred", linewidth=2.2, markersize=4, label="Teoria")
    
    plt.legend()
    plt.xlabel('k')
    plt.ylabel("pmf Z")
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"./img/SIS/{folder}/{title}.png", dpi=150)
    plt.close()



def plot_Z_expectation_SIS(n_values, means, theoretical, folder, title):
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, means, 'o-', color='deeppink', linewidth=2, markersize=6, label='Symulacja')
    plt.plot(n_values, theoretical, '--', color="forestgreen", linewidth=2, alpha=0.7, label='Teoria')
    plt.xlabel('n')
    plt.ylabel("E[Z]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./img/SIS/{folder}/{title}.png", dpi=150)
    plt.close()
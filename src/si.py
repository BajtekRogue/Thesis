import numpy as np
from utils import *
from simulations import simulate_final_infection_SI, simulate_full_infection_time_SI
from plotting import plot_Yt_distribution_SI, plot_Yt_expectation_SI, plot_Z_distribution_SI, plot_Z_expectation_SI


def compute_Yt_distribution_SI(graph_family, n, p, t, source, trials):
    graph = graph_constructor(graph_family, n)
    return simulate_final_infection_SI(graph, p, source, t, trials)


def compute_Yt_expectation_SI(graph_family, n, p, source, t_values, trials):
    graph = graph_constructor(graph_family, n)
    total_infection_count = np.zeros(len(t_values), dtype=float)

    for idx, t in enumerate(t_values):
        result = simulate_final_infection_SI(graph, p, source, t, trials)
        total_infection_count[idx] = np.mean(result)

    return total_infection_count


def compute_Z_data_SI(graph_family, n, p, source, trials):
    graph = graph_constructor(graph_family, n)
    data = simulate_full_infection_time_SI(graph, p, source, trials)
    return data, np.mean(data)


def main():

    source = 0
    p = 0.2
    n = 1000
    n_values = np.linspace(1, 1000, 20, dtype=int)

    # Path 
    total_path_time = n/p
    for idx, r in enumerate([0.2, 0.5, 0.8, 0.9]):
        t = int(r * total_path_time)
        title = f"Yt_dist_t{idx}"

        distribution_final_infection = compute_Yt_distribution_SI("path", n, p, t, source, trials=2000)
        plot_Yt_distribution_SI("path", distribution_final_infection, n, t, p, title)

    data_full_infection, _ = compute_Z_data_SI("path", n, p, source, trials=2000)
    plot_Z_distribution_SI("path", data_full_infection, n, p, "Z_dist")

    theoretical = [theoretical_expectation_Z_SI("path", n_val, p) for n_val in n_values]
    means_full_infection = []
    for n_val in n_values:
        _, mean_full_infection = compute_Z_data_SI("path", n_val, p, source, trials=1000)
        means_full_infection.append(mean_full_infection)
    plot_Z_expectation_SI("path", n_values, means_full_infection, theoretical, "Z_expectation")

    # Star
    t = int(0.5* np.log(n)/p)
    distribution_final_infection = compute_Yt_distribution_SI("star", n, p, t, source, trials=2000)
    plot_Yt_distribution_SI("star", distribution_final_infection, n, t, p, "Yt_dist")

    t_values = np.linspace(1, t, 20, dtype=int)
    total_infection_count = compute_Yt_expectation_SI("star", n, p, source, t_values, trials=1000)
    theoretical = [theoretical_expectation_Yt_SI("star", n, p, t_val) for t_val in t_values]
    plot_Yt_expectation_SI("star", t_values, total_infection_count, theoretical, f"Yt_expectation")

    data_full_infection, _ = compute_Z_data_SI("star", n, p, source, trials=2000)
    plot_Z_distribution_SI("star", data_full_infection, n, p, "Z_dist")

    theoretical = [theoretical_expectation_Z_SI("star", n_val, p) for n_val in n_values]
    means_full_infection = []
    for n_val in n_values:
        _, mean_full_infection = compute_Z_data_SI("star", n_val, p, source, trials=1000)
        means_full_infection.append(mean_full_infection)
    plot_Z_expectation_SI("star", n_values, means_full_infection, theoretical, "Z_expectation")

    # Cycle
    t = int(0.4*n/p)
    t_values = np.linspace(1, t, 20, dtype=int)
    total_infection_count = compute_Yt_expectation_SI("cycle", n, p, source, t_values, trials=1000)
    theoretical = [theoretical_expectation_Yt_SI("cycle", n, p, t_val) for t_val in t_values]
    plot_Yt_expectation_SI("cycle", t_values, total_infection_count, theoretical, f"Yt_expectation")

    theoretical = [theoretical_expectation_Z_SI("cycle", n_val, p) for n_val in n_values]
    means_full_infection = []
    for n_val in n_values:
        _, mean_full_infection = compute_Z_data_SI("cycle", n_val, p, source, trials=1000)
        means_full_infection.append(mean_full_infection)
    plot_Z_expectation_SI("cycle", n_values, means_full_infection, theoretical, "Z_expectation")

    # Complete
    n_values = np.linspace(1, 2000, 50, dtype=int)
    for p in [0.2, 0.1]: 
        print(f"p={p}")
        theoretical = [theoretical_expectation_Z_SI("complete", n_val, p) for n_val in n_values]
        means_full_infection = []
        for n_val in n_values:
            _, mean_full_infection = compute_Z_data_SI("complete", n_val, p, source, trials=100)
            means_full_infection.append(mean_full_infection)
        plot_Z_expectation_SI("complete", n_values, means_full_infection, theoretical, f"Z_expectation_p{p}")


if __name__ == "__main__":
    main()
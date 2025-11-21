import numpy as np
from tqdm import tqdm
import os
from utils import *
from simulations import simulate_final_infection_SI, simulate_full_infection_time_SI
from plotting import plot_Yt_distribution_SI, plot_Yt_expectation_SI, plot_Z_distribution_SI, plot_Z_expectation_SI


def compute_final_infection_distribution_SI(graph_family, n, p, source, trials):
    graph = graph_constructor(graph_family, n)
    t = time_function_SI(graph_family, n, p)
    return t, simulate_final_infection_SI(graph, p, source, t, trials)


def compute_final_infection_expectation_SI(graph_family, n, p, source, trials, t_values):
    graph = graph_constructor(graph_family, n)
    total_infection_count = np.zeros(len(t_values), dtype=float)

    for idx, t in enumerate(tqdm(t_values)):
        result = simulate_final_infection_SI(graph, p, source, t, trials)
        total_infection_count[idx] = np.mean(result)

    return t_values, total_infection_count


def compute_full_infection_data_SI(graph_family, n, p, source, trials):
    graph = graph_constructor(graph_family, n)
    data = simulate_full_infection_time_SI(graph, p, source, trials)
    return data, np.mean(data)


def main():
    graph_family = "star" 
    source = 0
    p = 0.2
    N = 1000
    trials_distribution_Yt = 2000
    trials_distribution_Z = 200
    trials_expectation_Yt = 1000 
    trials_expectation_Z = 100      
    n_values = np.linspace(1, N, 20, dtype=int)
    t_limit = theoretical_expectation_full_infection_SI(graph_family, N, p)
    t_values = np.linspace(1, int(t_limit), 20, dtype=int)

    os.makedirs(f"./img/SI/{graph_family}", exist_ok=True)

    # print("pmf Yt")
    # t, distribution_final_infection = compute_final_infection_distribution_SI(graph_family, N, p, source, trials_distribution_Yt)
    # plot_Yt_distribution_SI(graph_family, distribution_final_infection, N, t, p)

    # print("E[Yt]")
    # t_values, total_infection_count = compute_final_infection_expectation_SI(graph_family, N, p, source, trials_expectation_Yt, t_values)
    # theoretical = theoretical_expectation_Yt_SI(graph_family, N, p, t_values)
    # plot_Yt_expectation_SI(graph_family, t_values, total_infection_count, theoretical)

    print("pmf Z")
    data_full_infection, _ = compute_full_infection_data_SI(graph_family, N, p, source, trials_distribution_Z)
    plot_Z_distribution_SI(graph_family, data_full_infection, N, p)

    print("E[Z]")
    theoretical = theoretical_expectation_full_infection_SI(graph_family, n_values, p)
    means_full_infection = []
    for n in tqdm(n_values):
        _, mean_full_infection = compute_full_infection_data_SI(graph_family, n, p, source, trials_expectation_Z)
        means_full_infection.append(mean_full_infection)
    plot_Z_expectation_SI(graph_family, n_values, means_full_infection, theoretical)


if __name__ == "__main__":
    main()
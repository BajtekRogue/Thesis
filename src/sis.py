import numpy as np
from utils import *
from simulations import simulate_final_infection_SIS, simulate_extinction_time_SIS
from plotting import plot_Yt_distribution_SIS, plot_Yt_expectation_SIS, plot_Z_distribution_SIS, plot_Z_expectation_SIS
import os


def compute_Yt_distribution_SIS(graph_family, n, p, a, t, source, trials):
    graph = graph_constructor(graph_family, n)
    return simulate_final_infection_SIS(graph, p, a, source, t, trials)


def compute_Yt_expectation_SIS(graph_family, n, p, a, source, t_values, trials):
    graph = graph_constructor(graph_family, n)
    total_infection_count = np.zeros(len(t_values), dtype=float)

    for idx, t in enumerate(t_values):
        result = simulate_final_infection_SIS(graph, p, a, source, t, trials)
        total_infection_count[idx] = np.mean(result)

    return total_infection_count


def compute_Z_data_SIS(graph_family, n, p, a, source, trials):
    graph = graph_constructor(graph_family, n)
    data = simulate_extinction_time_SIS(graph, p, a, source, trials)
    return data, np.mean(data)


def main():

    source = 0
    n = 10
    n_values = np.linspace(1, n, n, dtype=int)

    for (p, a) in zip([0.1, 0.2, 0.5], [0.9, 0.8, 0.9]):

        folder = f"p{p}_a{a}"
        os.makedirs(f"./img/SIS/{folder}", exist_ok=True)

        total_time = max(theoretical_expectation_Z_SIS(n, Q_matrix_SIS(n, p, a)), 20)
        print(total_time)
        for idx, r in enumerate([0.2, 0.4, 0.6, 0.8]):
            t = int(r * total_time)
            distribution_final_infection = compute_Yt_distribution_SIS("complete", n, p, a, t, source, trials=1000)
            plot_Yt_distribution_SIS(distribution_final_infection, n, t, p, a, folder, f"Yt_dist_t{idx}")

        for idx, t_limit in enumerate([total_time, 10*total_time]):
            t_values = np.linspace(1, t_limit, 20, dtype=int)
            total_infection_count = compute_Yt_expectation_SIS("complete", n, p, a, source, t_values, trials=100)
            P = P_matrix_SIS(n, p, a)
            theoretical = [theoretical_expectation_Yt_SIS(n, y_t_vector_SIS(n, t_val, P)) for t_val in t_values]
            plot_Yt_expectation_SIS(t_values, total_infection_count, theoretical, folder, f"Yt_expectation{idx}")

        data_full_infection, _ = compute_Z_data_SIS("complete", n, p, a, source, trials=200)
        plot_Z_distribution_SIS(data_full_infection, n, p, a, folder, "Z_dist")

        theoretical = [theoretical_expectation_Z_SIS(n_val, Q_matrix_SIS(n_val, p, a)) for n_val in n_values]
        means_full_infection = []
        for n_val in n_values:
            _, mean_full_infection = compute_Z_data_SIS("complete", n_val, p, a, source, trials=100)
            means_full_infection.append(mean_full_infection)
        plot_Z_expectation_SIS(n_values, means_full_infection, theoretical, folder,  "Z_expectation")


if __name__ == "__main__":
    main()
import numpy as np
from tqdm import tqdm
from utils import *
from simulations import simulate_final_infection_SIR, simulate_extinction_time_SIR
from plotting import plot_W_distribution_SIR, plot_W_expectation_SIR, plot_Z_distribution_SIR, plot_Z_expectation_SIR


def main():

    source = 0
    p = 0.2
    a = 0.05
    n = 100
    n_values = np.linspace(1, 300, 30, dtype=int)
    family = "path"

    graph = graph_constructor(family, n)
    data_W = simulate_final_infection_SIR(graph, p, a, source, trials=2000)
    plot_W_distribution_SIR(family, data_W, n, p, a, "W_dist")

    theoretical_W = [theoretical_expectation_W_SIR(family, n_val, p, a) for n_val in n_values]
    means_final_infection = []
    for n_val in tqdm(n_values):
        graph = graph_constructor(family, n_val)
        data = simulate_final_infection_SIR(graph, p, a, source, trials=1000)
        means_final_infection.append(np.mean(data))  
    plot_W_expectation_SIR(family, n_values, means_final_infection, theoretical_W, "W_expectation")

    graph = graph_constructor(family, n)
    data_Z = simulate_extinction_time_SIR(graph, p, a, source, trials=2000)
    plot_Z_distribution_SIR(family, data_Z, n, p, a, "Z_dist")

    theoretical_Z = [theoretical_expectation_Z_SIR(family, n_val, p, a) for n_val in n_values]
    means_extinction_time = []
    for n_val in tqdm(n_values):
        graph = graph_constructor(family, n_val)
        data = simulate_extinction_time_SIR(graph, p, a, source, trials=1000)
        means_extinction_time.append(np.mean(data))
    plot_Z_expectation_SIR(family, n_values, means_extinction_time, theoretical_Z, "Z_expectation")


if __name__ == "__main__":
    main()
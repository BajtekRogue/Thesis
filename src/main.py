import numpy as np
from utils import *
from plot import *


def main():
    graph_family = "tree" 
    N = 1000             
    n_values = [100 * i for i in range(1, 11)]  
    p_values = [0.1, 0.2, 0.5]  
    trials = 2000         
    t_functions = generate_t_functions()

    for p in p_values:

        # PMF for Y
        graph = graph_constructor(graph_family, N)
        distributions_Y = {}

        for (f_name, f) in t_functions:
            distributions_Y[f_name] = compute_Y_distribution(graph, p, N, f, trials)
        
        plot_distributions_Y(graph_family, p, N, distributions_Y, t_functions)

        # E[Y]
        expectations_data_Y = {}

        for (f_name, f) in t_functions:

            t_values = []
            means = []

            for n in n_values:
                graph_n = graph_constructor(graph_family, n)
                t, mean = compute_Y_expectation(graph_n, p, n, f, trials)

                t_values.append(t)
                means.append(mean)

            expectations_data_Y[f_name] = {
                "n_values" : n_values,
                "t_values" : t_values,
                "means" : means
            }

        plot_expectations_Y(graph_family, p, expectations_data_Y, t_functions)

        # PMF for Z
        Z_data = compute_Z_data(graph, p, trials)
        plot_distribution_Z(Z_data, graph_family, p, N)

        # E[Z]
        means = []
        for n in n_values:
            graph_n = graph_constructor(graph_family, n)
            Z_data = compute_Z_data(graph_n, p, trials)

            mean = np.mean(Z_data)
            means.append(mean)

        plot_expectation_Z(graph_family, p, n_values, means)


if __name__ == "__main__":
    main()
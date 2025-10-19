import networkx as nx
import random
import numpy as np
from tqdm import tqdm

def graph_constructor(graph_family, n):
    builders = {
        "path": lambda k: nx.path_graph(k),
        "star": lambda k: nx.star_graph(k),
        "tree": lambda k: nx.random_labeled_tree(k)
    }
    return builders[graph_family](n)


def simulate_propagation(graph, p, source, rounds, trials):
    n = len(graph.nodes())
    results = np.zeros((trials, rounds + 1), dtype=int)
    neighbors = {v: list(graph.neighbors(v)) for v in graph.nodes()}
    
    for trial in tqdm(range(trials), desc=f"Y: n={n}, p={p}"):
        infected = set([source])
        counts = [1]
        
        for t in range(1, rounds + 1):
            new_infected = set()
            for v in infected:
                for u in neighbors[v]:
                    if u not in infected and random.random() < p:
                        new_infected.add(u)

            infected |= new_infected
            current_count = len(infected)
            counts.append(current_count)

            if len(infected) == n:
                counts.extend([n] * (rounds - t))
                break

        results[trial, :] = counts

    return results


def simulate_full_infection_time(graph, p, source, trials):
    n = len(graph.nodes())
    neighbors = {v: list(graph.neighbors(v)) for v in graph.nodes()}
    result = []
    
    for trial in tqdm(range(trials), desc=f"Z: n={n}, p={p}"):
        infected = set([source])
        t = 0

        while len(infected) < n:
            new_infected = set()
            for v in infected:
                for u in neighbors[v]:
                    if u not in infected and random.random() < p:
                        new_infected.add(u)

            infected |= new_infected
            t += 1

            if len(infected) == n:
                break

        result.append(t)

    return np.array(result)


def generate_t_functions():
    return [
        ("log(n)/p", lambda n, p: int(np.log(n) / p)),
        ("log(n)", lambda n, p: int(np.log(n))),
        ("√n", lambda n, p: int(np.sqrt(n))),
        ("n·p", lambda n, p: int(n * p)),
    ]


def theoretical_expectation_Y(graph_family, n, p, t_values):
    functions = {
        "path": lambda n, p, t: 1 + p * t,
        "star": lambda n, p, t: n * (1 - (1 - p) ** t),
        "tree": lambda n, p, t: np.zeros_like(n, dtype=float)
    }
    return [functions[graph_family](n, p, t) for t in t_values]


def theoretical_expectation_Z(graph_family, n, p):
    functions = {
        "path": lambda n, p: (n - 1) / p,
        "star": lambda n, p: np.log(n) / np.log(1 / (1 - p)),
        "tree": lambda n, p: np.zeros_like(n, dtype=float)
    }
    return functions[graph_family](n, p)


def compute_Y_distribution(graph, p, n, f, trials):
    t = min(f(n, p), n)
    results = simulate_propagation(graph, p, 0, t, trials)
    Y = results[:, -1]
    return Y


def compute_Y_expectation(graph, p, n, f, trials):
    t = min(f(n, p), n)
    results = simulate_propagation(graph, p, 0, t, trials)
    Y = results[:, -1]
    return t, np.mean(Y)


def compute_Z_data(graph, p, trials):
    return simulate_full_infection_time(graph, p, 0, trials)
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm
import os


def simulate_final_infection(graph, n, p, source, rounds, trials):
    results = np.zeros(trials, dtype=int)  
    neighbors = {v: list(graph.neighbors(v)) for v in graph.nodes()}
    
    for trial in range(trials):
        infected = set([source])
        
        for t in range(1, rounds + 1):
            new_infected = set()
            for v in infected:
                for u in neighbors[v]:
                    if u not in infected and random.random() < p:
                        new_infected.add(u)
            
            infected |= new_infected
            
            if len(infected) == n:
                break
        
        results[trial] = len(infected)  
    
    return results


def simulate_full_infection_time(graph, n, p, source, trials):
    neighbors = {v: list(graph.neighbors(v)) for v in graph.nodes()}
    result = np.arange(trials)
    
    for trial in range(trials):
        infected = set([source])
        time_taken = 0

        while len(infected) < n:
            new_infected = set()
            for v in infected:
                for u in neighbors[v]:
                    if u not in infected and random.random() < p:
                        new_infected.add(u)

            infected |= new_infected
            time_taken += 1

            if len(infected) == n:
                break

        result[trial] = time_taken

    return result


def compute_final_infection_distribution(graph_family, n, p, source, trials):
    graph = graph_constructor(graph_family, n)
    t = time_function(graph_family, graph, n, p)
    return t, simulate_final_infection(graph, n, p, source, t, trials)


def compute_final_infection_expectation(graph_family, n, p, source, trials, t_values):
    graph = graph_constructor(graph_family, n)
    total_infection_count = np.zeros(len(t_values), dtype=float)

    for idx, t in enumerate(tqdm(t_values)):
        result = simulate_final_infection(graph, n, p, source, t, trials)
        total_infection_count[idx] = np.mean(result)

    return t_values, total_infection_count


def compute_full_infection_data(graph_family, n, p, source, trials):
    graph = graph_constructor(graph_family, n)
    data = simulate_full_infection_time(graph, n, p, source, trials)
    return data, np.mean(data)


def plot_final_infection_distribution(graph_family, n, p, distribution, t_value):
    values, counts = np.unique(distribution, return_counts=True)
    probs = counts / np.sum(counts)
    mean_val = np.mean(distribution)

    plt.figure(figsize=(8, 5))
    plt.bar(values, probs, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Średnia: {mean_val:.2f}')
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("P[$Y_t$ = k]")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./img/{graph_family}/final_infection_distribution.png", dpi=150)
    plt.close()


def plot_final_infection_expectations(graph_family, n, p, t_values, means):
    theoretical = np.array([theoretical_expectation_final_infection(graph_family, n, p, int(t)) for t in t_values])

    plt.figure(figsize=(8, 5))
    plt.plot(t_values, means, 'o-', color="darkcyan",  linewidth=2, label='Symulacja', markersize=6)
    plt.plot(t_values, theoretical, '--', color="orange", linewidth=2, alpha=0.8, label='Teoria')
    plt.xlabel("t")
    plt.ylabel("Oczekiwana liczba zainfekowanych")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./img/{graph_family}/final_infection_expectations.png", dpi=150)
    plt.close()


def plot_full_infection_distribution(graph_family, n, p, samples):
    mean_z = np.mean(samples)
    n_bins = min(50, int(np.sqrt(len(samples))))

    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=n_bins, color='crimson', edgecolor='black', density=True, alpha=0.8)
    plt.axvline(mean_z, color='darkred', linestyle='--', linewidth=2.5, label=f'Średnia: {mean_z:.2f}')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel("P[Z = k]")
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"./img/{graph_family}/full_infection_distribution.png", dpi=150)
    plt.close()


def plot_full_infection_expectation(graph_family, p, n_values, means):
    theoretical = theoretical_expectation_full_infection(graph_family, np.array(n_values), p)

    plt.figure(figsize=(8, 5))
    plt.plot(n_values, means, 'o-', color='deeppink', linewidth=2, markersize=6, label='Symulacja')
    plt.plot(n_values, theoretical, '--', color="darkcyan", linewidth=2, alpha=0.7, label='Teoria')
    plt.xlabel('n')
    plt.ylabel("Oczekiwany czas zarażenia")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./img/{graph_family}/full_infection_expectation.png", dpi=150)
    plt.close()


def graph_constructor(graph_family, n):
    builders = {
        "path": lambda k: nx.path_graph(k),
        "star": lambda k: nx.star_graph(k),
        "tree": lambda k: nx.random_labeled_tree(k),
        "complete": lambda k: nx.complete_graph(k),
        "cycle": lambda k: nx.cycle_graph(k)
    }
    return builders[graph_family](n)


def theoretical_expectation_final_infection(graph_family, n, p, t):
    functions = {
        "path": lambda n, p, t: 1 + p * t,
        "star": lambda n, p, t: 1 + n * (1 - (1 - p) ** t),
        "tree": lambda n, p, t: 0,
        "complete": lambda n, p, t: 1 + (n - 1) * (1 - (1 - p) ** t) if t < 2 else n,
        "cycle": lambda n, p, t: 1 + 2 * p * t 
    }
    return functions[graph_family](n, p, t)


def theoretical_expectation_full_infection(graph_family, n, p):
    functions = {
        "path": lambda n, p: (n - 1) / p,
        "star": lambda n, p: np.log(n) / np.log(1 / (1 - p)),
        "tree": lambda n, p: np.zeros_like(n, dtype=float) if isinstance(n, np.ndarray) else 0,
        "complete": lambda n, p: np.full_like(n, 5, dtype=float) if isinstance(n, np.ndarray) else 5,
        "cycle": lambda n, p: n / (2 * p)  - np.sqrt(n * (1 - p) / (2 * np.pi)) / p
    }
    return functions[graph_family](n, p)


def time_function(graph_family, graph, n, p):
    functions = {
        "path": lambda G, n, p: n - 1,
        "star": lambda G, n, p: int(np.log(n)),
        "tree": lambda G, n, p: 0,
        "complete": lambda G, n, p: 5,
        "cycle": lambda G, n, p: (n - 1) // 2  
    }
    return functions[graph_family](graph, n, p)


def main():
    graph_family = "complete" 
    source = 0
    p = 0.2
    N = 1000
    trials_distribution = 5000
    trials_expectation = 2000        
    n_values = np.linspace(1, N, 20, dtype=int)
    t_limit = theoretical_expectation_full_infection(graph_family, N, p)
    t_values = np.linspace(1, t_limit, 20, dtype=int)

    os.makedirs(f"./img/{graph_family}", exist_ok=True)

    t_value, distribution_final_infection = compute_final_infection_distribution(graph_family, N, p, source, trials_distribution)
    plot_final_infection_distribution(graph_family, N, p, distribution_final_infection, t_value)

    t_values, total_infection_count = compute_final_infection_expectation(graph_family, N, p, source, trials_expectation, t_values)
    plot_final_infection_expectations(graph_family, N, p, t_values, total_infection_count)

    data_full_infection, _ = compute_full_infection_data(graph_family, N, p, source, trials_distribution)
    plot_full_infection_distribution(graph_family, N, p, data_full_infection)

    means_full_infection = []
    for n in tqdm(n_values):
        _, mean_full_infection = compute_full_infection_data(graph_family, n, p, source, trials_expectation)
        means_full_infection.append(mean_full_infection)

    plot_full_infection_expectation(graph_family, p, n_values, means_full_infection)


if __name__ == "__main__":
    main()
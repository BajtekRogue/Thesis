import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm
import os


def simulate_final_infection(graph, n, p, a, source, rounds, trials):
    """
    Run SIR for `rounds` steps and return final total ever infected (I+R) for each trial.
    Synchronous step: 1) infection attempts from current infected, 2) recovery of current infected.
    Newly infected this round cannot recover in the same round.
    """
    results = np.zeros(trials, dtype=int)
    neighbors = {v: list(graph.neighbors(v)) for v in graph.nodes()}

    for trial in range(trials):
        infected = set([source])
        recovered = set()

        for t in range(1, rounds + 1):
            # infection attempts
            new_infected = set()
            for v in infected:
                for u in neighbors[v]:
                    if u not in infected and u not in recovered:
                        if random.random() < p:
                            new_infected.add(u)

            # recovery attempts (only for currently infected)
            new_recovered = set()
            for v in infected:
                if random.random() < a:
                    new_recovered.add(v)

            # update sets
            infected |= new_infected
            infected -= new_recovered
            recovered |= new_recovered

            # if no active infections, epidemic is over
            if len(infected) == 0:
                break

        results[trial] = len(infected) + len(recovered)

    return results


def simulate_extinction_time(graph, n, p, a, source, trials, max_steps=10000):
    """
    Return array of times until process cannot spread anymore.
    We define extinction time Z as the first time when there are no infected nodes (I==0)
    OR when active infected nodes have no susceptible neighbours (so no future infections are possible).
    The function returns the stopping time (number of rounds simulated).
    """
    neighbors = {v: list(graph.neighbors(v)) for v in graph.nodes()}
    result = np.zeros(trials, dtype=int)

    for trial in range(trials):
        infected = set([source])
        recovered = set()
        t = 0

        while True:
            t += 1
            # infection attempts
            new_infected = set()
            for v in infected:
                for u in neighbors[v]:
                    if u not in infected and u not in recovered:
                        if random.random() < p:
                            new_infected.add(u)

            # recoveries
            new_recovered = set()
            for v in infected:
                if random.random() < a:
                    new_recovered.add(v)

            # update
            infected |= new_infected
            infected -= new_recovered
            recovered |= new_recovered

            # if no currently infected => epidemic ended now
            if len(infected) == 0:
                break

            # check if current infected nodes have any susceptible neighbours
            no_more_possible = True
            for v in infected:
                for u in neighbors[v]:
                    if u not in infected and u not in recovered:
                        no_more_possible = False
                        break
                if not no_more_possible:
                    break

            if no_more_possible:
                break

            if t >= max_steps:
                # safety cap to avoid infinite loops in weird cases
                break

        result[trial] = t

    return result


def graph_constructor(graph_family, n):
    builders = {
        "path": lambda k: nx.path_graph(k),
        "star": lambda k: nx.star_graph(k),
        "complete": lambda k: nx.complete_graph(k),
        "cycle": lambda k: nx.cycle_graph(k)
    }
    return builders[graph_family](n)


def time_function(graph_family, graph, n, p, a):
    """
    Heuristic time horizon for Y_t experiments. It's fine to be conservative.
    """
    functions = {
        "path": lambda G, n, p, a: max(10, n - 1),
        "star": lambda G, n, p, a: max(10, int(np.ceil(np.log(n+1) / np.log(1/(1-p+1e-12))))),
        "complete": lambda G, n, p, a: max(5, 2),
        "cycle": lambda G, n, p, a: max(10, (n - 1) // 2)
    }
    return functions[graph_family](graph, n, p, a)


# --- Theoretical helpers (optional, kept simple) ---
def theoretical_expectation_final_infection(graph_family, n, p, a, t):
    """
    Return expected number of ever-infected (Y_t) at time t for given small graph families.
    If no simple closed form is implemented, return np.nan (or a scalar array of nan of appropriate shape).
    We try to return a scalar or array matching input `n`.
    """
    q = 1 - p
    beta = 1 - a

    if graph_family == "star":
        # center infected at time 0. Expectation as derived earlier:
        # E[Y_t] = 1 + n * p * (1 - (q*beta)^t) / (1 - q*beta)
        val = 1 + n * p * (1 - (q * beta)**t) / (1 - q * beta)
        return val
    elif graph_family == "path":
        # no simple closed form implemented here; return nan(s)
        if isinstance(n, np.ndarray):
            return np.full_like(n, np.nan, dtype=float)
        return np.nan
    else:
        if isinstance(n, np.ndarray):
            return np.full_like(n, np.nan, dtype=float)
        return np.nan


def theoretical_expectation_extinction(graph_family, n_vals, p, a):
    """
    If possible return theoretical expected extinction time as an array matching n_vals.
    Otherwise return array of np.nan.
    """
    # For now we only implement a rough placeholder for star; other families are left NaN.
    if graph_family == "star":
        # crude heuristic: expected extinction time grows like ~ 1/alpha for large n,
        # but we return something monotone in n for plotting convenience.
        def star_est(n):
            return (1 / (a + 1e-12)) * (1 - np.exp(-n / max(10, n)))
        if isinstance(n_vals, np.ndarray):
            return np.array([star_est(int(x)) for x in n_vals], dtype=float)
        else:
            return star_est(int(n_vals))
    else:
        if isinstance(n_vals, np.ndarray):
            return np.full_like(n_vals, np.nan, dtype=float)
        return np.nan


# --- plotting helpers with safe checks ---
def plot_with_optional_theory(x, y, theory, xlabel, ylabel, outpath, sim_label="Simulation", theory_label="Theory"):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'o-', color="tab:blue", linewidth=2, markersize=6, label=sim_label)
    if theory is not None:
        try:
            theory = np.asarray(theory)
            if theory.shape == np.asarray(x).shape:
                plt.plot(x, theory, '--', color="orange", linewidth=2, alpha=0.8, label=theory_label)
        except Exception:
            pass
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# --- main experiment driver ---
def main():
    graph_family = "path"   # choose "path","star","complete","cycle"
    source = 0
    p = 0.2
    a = 0.1
    N = 100            # graph size for distributions
    trials_distribution = 2000
    trials_expectation = 1000

    outdir = f"./img/SIR/{graph_family}"
    os.makedirs(outdir, exist_ok=True)

    # --- final infection distribution at t = t_limit ---
    graph = graph_constructor(graph_family, N)
    t_limit = time_function(graph_family, graph, N, p, a)
    print(f"Using simulation horizon t_limit = {t_limit}")
    t_value, distribution_final_infection = t_limit, simulate_final_infection(graph, N, p, a, source, t_limit, trials_distribution)
    # plot distribution
    values, counts = np.unique(distribution_final_infection, return_counts=True)
    probs = counts / np.sum(counts)
    mean_val = np.mean(distribution_final_infection)
    plt.figure(figsize=(8, 5))
    plt.bar(values, probs, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.legend()
    plt.xlabel("k (final infected count after t)")
    plt.ylabel("Probability")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "final_infection_distribution.png"), dpi=150)
    plt.close()

    # --- expectation vs t ---
    t_values = np.linspace(1, max(10, t_limit), min(20, max(10, t_limit)), dtype=int)
    means = []
    for t in tqdm(t_values, desc="simulate E[Y_t]"):
        res = simulate_final_infection(graph, N, p, a, source, int(t), trials_expectation)
        means.append(np.mean(res))
    means = np.array(means)
    theory_vals = theoretical_expectation_final_infection(graph_family, N, p, a, t_values)
    plot_with_optional_theory(t_values, means, theory_vals, "t", "E[Y_t]", os.path.join(outdir, "final_infection_expectations.png"))

    # --- extinction times ---
    samples = simulate_extinction_time(graph, N, p, a, source, trials_distribution)
    mean_z = np.mean(samples)
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=min(50, int(np.sqrt(len(samples)))), color='crimson', edgecolor='black', density=True, alpha=0.8)
    plt.axvline(mean_z, color='darkred', linestyle='--', linewidth=2.5, label=f'Mean: {mean_z:.2f}')
    plt.legend()
    plt.xlabel('extinction time (Z)')
    plt.ylabel("Density")
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "extinction_time_distribution.png"), dpi=150)
    plt.close()

    # --- extinction expectation vs n ---
    n_values = np.linspace(10, N, 10, dtype=int)
    mean_times = []
    for ni in tqdm(n_values, desc="simulate E[Z] vs n"):
        g = graph_constructor(graph_family, ni)
        res = simulate_extinction_time(g, ni, p, a, source, trials_expectation)
        mean_times.append(np.mean(res))
    mean_times = np.array(mean_times)
    theory_times = theoretical_expectation_extinction(graph_family, n_values, p, a)
    plot_with_optional_theory(n_values, mean_times, theory_times, "n", "E[Z]", os.path.join(outdir, "extinction_time_expectation.png"))

    print("Done. Results saved in", outdir)


if __name__ == "__main__":
    main()

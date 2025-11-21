import numpy as np
import networkx as nx
import math
import sympy
from scipy import stats

def graph_constructor(graph_family, n):
    builders = {
        "path": lambda k: nx.path_graph(k),
        "star": lambda k: nx.star_graph(k),
        "complete": lambda k: nx.complete_graph(k),
        "cycle": lambda k: nx.cycle_graph(k)
    }
    return builders[graph_family](n)


def time_function_SI(graph_family, n, p):
    functions = {
        "path": lambda n, p: n - 1,
        "star": lambda n, p: int(np.log(n)),
        "complete": lambda n, p: 2,
        "cycle": lambda n, p: (n - 1) // 2  
    }
    return functions[graph_family](n, p)


def theoretical_probs_Yt_SI(graph_family, k, n, t, p):
    if graph_family == "path":
        if k < n:
            return stats.binom.pmf(k, t-1, p)
        else:
            return stats.binom.sf(n-2, t-1, p)

    if graph_family == "star":
        return stats.binom.pmf(k-1, n, 1 - (1-p) ** t)

    if graph_family == "cycle":
        if k < n:
            return stats.binom.pmf(k, 2*t-1, p)
        else:
            return stats.binom.sf(n-2, 2*t-1, p)


def theoretical_expectation_Yt_SI(graph_family, n, p, t):
    functions = {
        "path": lambda n, p, t: 1 + p * t,
        "star": lambda n, p, t: 1 + n * (1 - (1 - p) ** t),
        "cycle": lambda n, p, t: 1 + 2 * p * t,
        "complete": lambda n, p, t: 1 + (n - 1) * p if t == 1 else n
    }
    return functions[graph_family](n, p, t)


def theoretical_probs_Z_SI(graph_family, k, n, p):
    if graph_family == "path":
        return stats.nbinom.pmf(k - n + 1, n - 1, p)

    if graph_family == "star":
        cdf_k = stats.geom.cdf(k, p)
        cdf_k_minus_1 = stats.geom.cdf(k - 1, p)
        return cdf_k**n - cdf_k_minus_1**n

    if graph_family == "cycle":
        cdf_k = stats.nbinom.cdf(k - n // 2, n // 2, p)
        cdf_k_minus_1 = stats.nbinom.cdf(k - 1 - n // 2, n // 2, p)
        return cdf_k**2 - cdf_k_minus_1**2


def theoretical_expectation_full_infection_SI(graph_family, n, p):
    functions = {
        "path": lambda n, p: (n - 1) / p,
        "star": lambda n, p: np.array([sympy.harmonic(k) for k in n]) / np.log(1 / (1 - p)) if isinstance(n, np.ndarray) else sympy.harmonic(n) / np.log(1 / (1 - p)),
        "cycle": lambda n, p: n / (2 * p)  - np.sqrt(n * (1 - p) / (2 * np.pi)) / p,
        "complete": lambda n, p: np.full_like(n, 2, dtype=float) if isinstance(n, np.ndarray) else 2
    }
    return functions[graph_family](n, p)

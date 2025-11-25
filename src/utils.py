import numpy as np
import networkx as nx
import sympy
from scipy import stats
import math
from math import log, exp, lgamma, ceil, comb

def graph_constructor(graph_family, n):
    builders = {
        "path": lambda k: nx.path_graph(k),
        "star": lambda k: nx.star_graph(k),
        "cycle": lambda k: nx.cycle_graph(k),
        "complete": lambda k: nx.complete_graph(k)
    }
    return builders[graph_family](n)


def theoretical_probs_Yt_SI(graph_family, k, n, t, p):
    if graph_family == "path":
        if k < n:
            return stats.binom.pmf(k-1, t, p)
        else:
            return stats.binom.sf(n-2, t, p)

    if graph_family == "star":
        return stats.binom.pmf(k-1, n, 1 - (1-p) ** t)

    if graph_family == "cycle":
        if k < n:
            return stats.binom.pmf(k, 2*t-1, p)
        else:
            return stats.binom.sf(n-2, 2*t-1, p)


def theoretical_expectation_Yt_SI(graph_family, n, p, t):
    functions = {
        "path": lambda n, p, t: sum([min(n, 1+j) * stats.binom.pmf(j, t, p) for j in range(t+1)]),
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


def theoretical_expectation_Z_SI(graph_family, n, p):
    functions = {
        "path": lambda n, p: (n - 1) / p,
        "star": lambda n, p: sympy.harmonic(n) / np.log(1 / (1 - p)),
        "cycle": lambda n, p: n / (2 * p)  - np.sqrt(n * (1 - p) / (2 * np.pi)) / p,
        "complete": lambda n, p: 2
    }
    return functions[graph_family](n, p)


def theoretical_probs_W_SIR(graph_family, k, n, p, a):
    q = 1 - p
    b = 1 - a

    if graph_family == "path":
        tau = p / (1 - q * b)
        if k < n:
            return stats.geom.pmf(k, 1-tau)
        else:
            return tau ** (n-1)

    if graph_family == "star":
        if k < 0 or k > n+1:
            return 0.0
        k -= 1

        log_pref = log(a) - log(b) + lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
        r = b * (q ** (n - k))
        max_J=100000
        tol=1e-12
        if r == 0:
            J = 5
        elif r >= 1.0:
            J = max_J
        else:
            J = max(1, int(ceil((log(tol*(1-r)) / log(r)) - 1)))
            J = min(J, max_J)

        j = np.arange(1, J+1, dtype=float)
        log_tj = j*log(b) + (n-k)*j*log(q) + k * np.log1p(-q**j)  
        m = np.max(log_tj)
        sum_t = np.exp(log_pref) * np.exp(m) * np.sum(np.exp(log_tj - m))
        return float(sum_t)
    
    return 0


def theoretical_expectation_W_SIR(graph_family, n, p, a):
    q = 1 - p
    b = 1 - a

    if graph_family == "path":
        theta = p / (1 - q * b)
        return (1 - theta ** n) / (1 - theta)
    
    if graph_family == "star":
        return 1 + n * p / (1 - q*b)    

    return 0


def theoretical_probs_Z_SIR(graph_family, k, n, p, a):
    q = 1 - p
    b = 1 - a

    if graph_family == "path":
        theta = p / (1 - q * b)
        terms = np.array([stats.nbinom.pmf(k - m, m, 1 - q * b) * stats.geom.pmf(m, 1 - theta) for m in range(1,n-1)])
        terms = np.append(terms, stats.nbinom.pmf(k - (n-1), n-1, 1 - q * b) * theta ** (n-2))
        return np.sum(terms)
    
    if graph_family == "star":
        survival = lambda t: b**t * (1 - (1 - q**t) ** n)
        if k == 0:
            return 1 - survival(0)
        else:
            return survival(k-1) - survival(k)


def theoretical_expectation_Z_SIR(graph_family, n, p, a):
    q = 1 - p
    b = 1 - a

    if graph_family == "path":
        theta = p / (1 - q * b)
        return (1- theta**(n-1)) / (q * a)

    if graph_family == "star":
        A = np.log(q)
        B = np.log(b)
        r = B / A
        poch = exp(lgamma(n+1) - lgamma(r+1+n) + lgamma(r+1))
        return (poch - 1) / B
        
    return 0


def P_matrix_SIS(n, p, a):
    P = np.zeros((n+1, n+1))
    q = 1 - p
    b = 1 - a

    for i in range(n+1):
        for j in range(n+1):
            l_min = max(0, j - (n - i))
            l_max = min(i, j)
            if l_min > l_max:
                continue
            terms = [
                comb(i, l) * comb(n - i, j - l) * b**l * a**(i - l) * (1 - q**i)**(j - l) * (q**i)**(n - i - (j - l))
                for l in range(l_min, l_max + 1)
            ]
            P[j, i] = np.sum(terms)

    return P


def Q_matrix_SIS(n, p, a):
    P = P_matrix_SIS(n, p, a)
    Q = P[1:, 1:]
    return Q


def y_t_vector_SIS(n, t, P):
    y0 = np.zeros(n+1)
    y0[1] = 1
    return np.linalg.matrix_power(P, t) @ y0


def theoretical_probs_Yt_SIS(k, y_t):
    return y_t[k]


def theoretical_expectation_Yt_SIS(n, y_t):
    return np.arange(n+1) @ y_t


def theoretical_probs_Z_SIS(k, n, P):
    y_t = y_t_vector_SIS(n, k, P)
    y_t_minus1 = y_t_vector_SIS(n, k-1, P) if k > 0 else np.zeros(n+1)
    cdf = 1 - y_t[0]
    cdf_minus1 = 1 - y_t_minus1[0]
    return cdf_minus1 - cdf


def theoretical_survival_Z_SIS(k, n, P):
    y_t = y_t_vector_SIS(n, k, P)
    return 1 - y_t[0]



def theoretical_expectation_Z_SIS(n, Q):
    I = np.eye(n)
    z0 = np.zeros(n)
    z0[0] = 1 
    inv = np.linalg.inv(I - Q)
    return np.sum(inv @ z0)
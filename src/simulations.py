import random
import numpy as np
from tqdm import tqdm 

def simulate_final_infection_SI(graph, p, source, t, trials):
    n = graph.number_of_nodes()
    neighbors = {v: list(graph.neighbors(v)) for v in graph.nodes()}
    results = np.zeros(trials, dtype=int)  
    
    for trial in range(trials):
        infected = set([source])
        
        for _ in range(1, t + 1):
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


def simulate_full_infection_time_SI(graph, p, source, trials):
    n = graph.number_of_nodes()
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

        result[trial] = time_taken

    return result


def simulate_final_infection_SIR(graph, p, a, source, trials):
    neighbors = {v: list(graph.neighbors(v)) for v in graph.nodes()}
    results = np.zeros(trials, dtype=int)  
    
    for trial in range(trials):
        infected = set([source])
        recovered = set()
        susceptible = set(graph.nodes())
        susceptible.remove(source) 

        while infected and any(u in susceptible for v in infected for u in neighbors[v]):
            new_infected = set()
            for v in infected:
                for u in neighbors[v]:
                    if u in susceptible and random.random() < p:
                        new_infected.add(u)
            
            new_recovered = set()
            for v in infected:
                if random.random() < a:
                    new_recovered.add(v)
            
            infected |= new_infected
            susceptible -= new_infected
            infected -= new_recovered
            recovered |= new_recovered
        
        recovered |= infected
        results[trial] = len(recovered)
    
    return results


def simulate_extinction_time_SIR(graph, p, a, source, trials):
    neighbors = {v: list(graph.neighbors(v)) for v in graph.nodes()}
    result = np.arange(trials)
    
    for trial in range(trials):
        infected = set([source])
        recovered = set()
        susceptible = set(graph.nodes())
        susceptible.remove(source)  
        time_taken = 0

        while infected and any(susceptible & set(neighbors[v]) for v in infected):
            new_infected = set()
            for v in infected:
                for u in neighbors[v]:
                    if u in susceptible and random.random() < p:
                        new_infected.add(u)

            new_recovered = set()
            for v in infected:
                if random.random() < a:
                    new_recovered.add(v)

            infected |= new_infected
            susceptible -= new_infected
            infected -= new_recovered
            recovered |= new_recovered
            time_taken += 1

        result[trial] = time_taken

    return result


def simulate_final_infection_SIS(graph, p, a, source, t, trials):
    neighbors = {v: list(graph.neighbors(v)) for v in graph.nodes()}
    results = np.zeros(trials, dtype=int)  
    
    for trial in range(trials):
        infected = set([source])
        
        for _ in range(1, t + 1):
            new_infected = set()
            for v in infected:
                for u in neighbors[v]:
                    if u not in infected and random.random() < p:
                        new_infected.add(u)
            
            recovered = set()
            for v in infected:
                if random.random() < a:
                    recovered.add(v)
            
            infected |= new_infected
            infected -= recovered
            if len(infected) == 0:
                break
        
        results[trial] = len(infected)  
    
    return results


def simulate_extinction_time_SIS(graph, p, a, source, trials):
    neighbors = {v: list(graph.neighbors(v)) for v in graph.nodes()}
    result = np.arange(trials)
    
    for trial in range(trials):
        infected = set([source])
        time_taken = 0

        while infected:
            new_infected = set()
            for v in infected:
                for u in neighbors[v]:
                    if u not in infected and random.random() < p:
                        new_infected.add(u)

            recovered = set()
            for v in infected:
                if random.random() < a:
                    recovered.add(v)
            
            infected |= new_infected
            infected -= recovered
            time_taken += 1

        result[trial] = time_taken

    return result
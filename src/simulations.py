import random
import numpy as np

def simulate_final_infection_SI(graph, p, source, rounds, trials):
    n = graph.number_of_nodes()
    neighbors = {v: list(graph.neighbors(v)) for v in graph.nodes()}
    results = np.zeros(trials, dtype=int)  
    
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

            if len(infected) == n:
                break

        result[trial] = time_taken

    return result
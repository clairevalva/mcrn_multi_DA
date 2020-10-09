'''
Takes in a population and returns a network (i.e., the adjacancy and degree matrices)
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def make(population, seed):
    total_nodes = population
    nearest_neighbors = 4
    prob_rewiring = 0.5

    G = nx.watts_strogatz_graph(n=total_nodes, k=nearest_neighbors, p=prob_rewiring, seed=seed)

    return G
'''
Takes in a population and returns a network (i.e., the adjacancy and degree matrices)
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def make(population, seed, nearest_neighbors, prob_rewiring):

    G = nx.watts_strogatz_graph(n=population, k=nearest_neighbors, p=prob_rewiring, seed=seed)

    return G

def trivial(population):
    return nx.empty_graph(population)

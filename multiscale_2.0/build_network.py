'''
Takes in a population and returns a network (i.e., the adjacancy and degree matrices)
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def build_network(population):
    total_nodes = population
    nearest_neighbors = 2
    prob_rewiring = 0.5

    #G = nx.watts_strogatz_graph(n = total_nodes,
    #                            k = nearest_neighbors,
    #                            p = prob_rewiring)

    G = nx.watts_strogatz_graph(n = total_nodes, k = nearest_neighbors, p = prob_rewiring, seed=4)

    # Visualizing
#    nx.draw(G, with_labels=True, font_weight='bold')
#    plt.show()
    return G
'''
Function to construct population with given attributes
'''

import numpy as np
import networkx as nx
import network



def build_L(population):
    layers = 7 # number of schedule layers
    L = []

    # Make Weekday L matrices
    # Classes
    networkWD1 = network.make(population, seed=5, nearest_neighbors=4, prob_rewiring=.5)
    networkWD2 = network.make(population, seed=6, nearest_neighbors=4, prob_rewiring=.5)
    networkWD3 = network.make(population, seed=7, nearest_neighbors=4, prob_rewiring=.5)
    networkWD4 = network.make(population, seed=8, nearest_neighbors=4, prob_rewiring=.5)
    LWD1 = nx.laplacian_matrix(networkWD1)
    LWD2 = nx.laplacian_matrix(networkWD2)
    LWD3 = nx.laplacian_matrix(networkWD3)
    LWD4 = nx.laplacian_matrix(networkWD4)
    L.append(LWD1)
    L.append(LWD2)
    L.append(LWD3)
    L.append(LWD4)
    # Outside of class
    networkWD5 = network.make(population, seed=1, nearest_neighbors=2, prob_rewiring=.5)
    LWD5 = nx.laplacian_matrix(networkWD5)
    L.append(LWD5)

    # Weekend L matrices
    networkWE = network.make(population, seed=1, nearest_neighbors=2, prob_rewiring=0.5)
    LWE = nx.laplacian_matrix(networkWE)
    L.append(LWE)

    # Trivial L matrix
    network_trivial = network.trivial(population)
    L_trivial = nx.laplacian_matrix(network_trivial)
    L.append(L_trivial)

    return L
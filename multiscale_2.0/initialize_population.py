'''
Function to construct population with given attributes
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from build_network import build_network

population = 10
network = build_network(population)
L = nx.laplacian_matrix(network)

I = np.zeros(10)

# Initial condition
I[0] = 1



# # Visualization
# nx.draw(network, with_labels=True, font_weight='bold')
# plt.show()
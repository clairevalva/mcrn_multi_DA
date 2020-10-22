import numpy as np
import networkx as nx
import random

def flip(p):
    "utlility function"
    
    return True if random.random() < p else False


def initialize_tree(num_infected, total_pop, seed = False, return_initial = True):
    tree = nx.DiGraph()
    if seed != False:
        # for purposes of reproducibility
        random.seed(seed)
    
    initial_infected = np.array([random.randrange(0, total_pop) for _ in range(num_infected)])
    tree.add_nodes_from(initial_infected)
    
    for node in initial_infected:
        tree.nodes[node]["I_time"] = 0
        
    if return_initial:
        return tree, initial_infected
    
    return tree

def addinfected_fromnode(tree, root, persons_infected, I_time):
    """ function takes a graph (tree), 
    the person infecting others (root, should be integer label of the node),
    the persons infected by the root (passed as an np.array),
    and the date/time (I_time) -> probably should be integer (maybe float)
    
    """
    
    tree.add_edges_from([(root, person) for person in persons_infected])
    for node in persons_infected:
        tree.nodes[node]["I_time"] = I_time               
    return tree
                    
    
def addinfected_all(tree, add_list, I_time):
    """
    this function takes the graph (tree)
    add list should be of shape (# of roots, 2) where [i,0] is a root and [i,1] are the persons infected by the root
    and the date/time (I_time) -> probably should be integer (maybe float)
    
    this code just really functions as a wrapper for ease
    """
    
    for entry in add_list:
        tree = addinfected_fromnode(tree, entry[0], entry[1], I_time)
        
    return tree                           
    

def connected(node, L):
    """
    i don't know how the network is structured, this should return a list of everyone people are connected to
    """
    print("scream")
    

def suseptible_connects(node, L, tree):
    connects = connected(node, L)
    return connects[connects not in tree.nodes()]
    
    
def is_sick(tree, L, p_infected):
    
    return_list = []
    
    for node in tree.nodes():
        connects = suseptible_connects(node, L, tree)
        now_infected = []
        sick = [flip(p_infected) for _ in range(len(connects))]
        
        now_infected = [connects[j] for j in range(len(connects)) if sick[j]]
         
        if len(now_infected) > 0:
            return_list.append([node,now_infected])
    
    return np.array(return_list)
    
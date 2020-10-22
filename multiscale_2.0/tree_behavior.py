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
        tree.nodes[node]["I_time"] = -1
        
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
    print(add_list)
    for entry in add_list:
        # print("ent1", entry[1])
        # print(len(entry[1]))
        if np.sum(entry[1]) > 0:
            tree = addinfected_fromnode(tree, entry[0], entry[1], I_time)
        
    return tree                           
    
    

def suseptible_connects(node, G, tree):
    connects = np.array([n for n in G.neighbors(node)])
    # print([connects not in np.array(tree.nodes())])
    
    return connects[connects not in np.array(tree.nodes())]
    
    
def is_sick(tree, G, p_infected,I_time,sicklen):
    
    """
    takes the tree (directed graph)
    G (the schedule for whenever)
    p_infected (the probability of infection)
    """
    
    return_list = []
    
    for entry in tree.nodes(data="I_time"):
        node = entry[0]
        still_sick = (entry[1] + sicklen) >= I_time
        
        if still_sick:
        
            connects = suseptible_connects(node, G, tree).flatten()
            # print("less", connects)
            now_infected = []
            sick = [flip(p_infected) for _ in range(len(connects))]
        
        
            now_infected = connects[sick]
         
            if np.sum(sick) > 0:
                return_list.append([node,now_infected])
    
    return np.array(return_list)

def remove_recovered(tree, G, current_day, sicklen):
    for node in tree.nodes(data="I_time"):
        #print(node[1],"node")
        if node[1] + sicklen < current_day:
            if node[0] in G.nodes():
                G.remove_node(node[0])
                
    return G
# this file has functions to remove agents to act as a "quarantine," 
# can either use the same agents as previous day or not

import numpy as np


def remove_ls(num, tick, N, model, previous = False, uselist = []):
    ''' 
    unless other list is prescribed, 
    choses random num of agents to remove
    where N is the total number of agents in the simulation
    ''' 
    if previous:
        return np.array(uselist)[:num]
    else:
        people = np.array(range(N))
        random.shuffle(people)
        
        return np.array(people)[:num]


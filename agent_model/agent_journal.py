import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
import itertools
from mesa.space import MultiGrid

# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'agent_model/')

import class_assign_funcs
import random

class Person(Agent):
    # start with fixing the number of contacts = 0
    def __init__(self, unique_id, persontype, model):
    
        super().__init__(unique_id, model)
        self.type = persontype
        self.metlast = []
        self.classes = model.classes[:,unique_id]
        self.model = model
        
    def move(self):
        if self.model.tick < len(self.classes):
            # this assumes that the number of grid points is greater 
            # than the number of classes
            # i.e. TODO: implement fancy mod math / and or better class room grid
            
            new_position = 0,self.classes[self.model.tick]
            
        else:
            possible_steps = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=False)
            new_position = self.random.choice(possible_steps)
        
        
        self.model.grid.move_agent(self, new_position)
        
    def step(self):
        self.move()
       
       
class UnivModel(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height, class_periods = 3, class_size = 3, majors = False):
        # majors is false if not implemented
        #otherwise majors should be the number of persons per major
        
        self.num_agents = N
        self.grid = MultiGrid(width, height, False)
        self.schedule = RandomActivation(self)
        self.contactjournal = np.zeros((N,N), dtype=int)
        self.contactrep = np.zeros((N,N), dtype=int) # type 2 contact
        self.class_periods = class_periods
        self.class_size = class_size
        
        self.shuffled = np.array(random.shuffle(np.array(range(N))))
        
        if not majors:
            classdet = class_assign_funcs.class_assign(N, class_periods, class_size)
            self.classes = classdet[0]
        else:
            classdet = class_assign_funcs.class_assign_majors(N, class_periods, class_size)
            self.classes = class_assign_funcs.reshape_formod(classdet[0])
        
        numberclasses = classdet[1]
        #print("the number of classes is: ", numberclasses)
        
        self.tick = 0
        
        # Create agents
        for i in range(self.num_agents):
            a = Person(i, "student", self)
            self.schedule.add(a)
            
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            
        

    def step(self, toremove = [], toadd = []):
        '''Advance the model by one step, 
        then update the contact matrix. 
        removes agents in the toremove list, 
        adds agents in the to add list'''
        
        for ag in toremove:
            del self.schedule._agents[ag]
            
        for ag in toadd:
            self.schedule._agents[ag] = Person(ag, "student", self)
        
        self.schedule.step()
        self.tick += 1
        # need to iterate (hopefully not) through the gridpoints
        # then add up all the contacts
        
        
        for cell in self.grid.coord_iter():
            agents, x, y = cell
            ids = [a.unique_id for a in agents]
            if len(agents) > 1:
                combos = list(itertools.combinations(agents, 2))
            
                for pair in combos:

                    self.contactjournal[pair[0].unique_id,pair[1].unique_id] += 1
                    self.contactjournal[pair[1].unique_id,pair[0].unique_id] += 1
                    
                    # tracks type 2 contacts
                    if pair[0].unique_id not in pair[1].metlast:
                        self.contactrep[pair[0].unique_id,pair[1].unique_id] += 1
                        self.contactrep[pair[1].unique_id,pair[0].unique_id] += 1
                        
                        
                
            # assign met last
            for agent in agents:
                agent.metlast = ids
                    
                
                
def contactnumbers(model, returnarr = True):
        # total number of contacts*len of each person 
        total_contacts = np.sum(model.contactjournal, axis = -1)  
        
        # type 2
        total_interactions = np.sum(model.contactrep, axis = -1)
        
        # type 3
        typ3true = (model.contactjournal != 0)
        total_type3 = np.sum(typ3true, axis = -1)
        
        if returnarr:
               return total_contacts, total_interactions, total_type3
            
        else:
            mean1 = np.mean(total_contacts)
            mean2 = np.mean(total_interactions)
            mean3 = np.mean(total_type3)
            
            return mean1, mean2, mean3
            





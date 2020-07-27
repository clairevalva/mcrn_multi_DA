import numpy as np
import math
import random

def class_assign(num_students,class_periods,class_size):
    # note this may give some extra students schedules, just throw those out

    num_classes = math.ceil(num_students/class_size)

    class_slots = np.repeat(np.arange(0,num_classes),class_size)
    schedule = np.repeat([class_slots],class_periods,0)
    [np.random.shuffle(x) for x in schedule]
    
    return schedule, num_classes

def class_assign_majors(num_students, class_periods, class_size, major_size = 500):
    # note this may give some extra students schedules, just throw those out

    num_classes = math.ceil(num_students/class_size)
    num_majors = math.ceil(num_students/major_size)
    
    classes_permajor = math.ceil(num_classes/num_majors)
    
    class_slots = np.repeat(np.arange(0,classes_permajor*num_majors), class_size)
    
    
    major_slots = np.reshape(class_slots, (num_majors, -1))
    # so this is an array of num_majors X class_slots
    
    schedule = np.array([np.repeat([slots], class_periods, axis = 0) 
                for slots in major_slots])
    
    
    # shuffler
    rng = np.random.default_rng()
    [rng.shuffle(x) for m in schedule for x in m]
    
    # shape of schedule should be num of majors x periods x classes
    return schedule, num_classes
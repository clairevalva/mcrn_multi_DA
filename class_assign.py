import numpy as np
import math
import random

def class_assign(num_students,class_periods,class_size):
    # note this may give some extra students schedules, just throw those out

    num_classes = math.ceil(num_students/class_size)

    class_slots = np.repeat(np.arange(0,num_classes),class_size)
    schedule = np.repeat([class_slots],class_periods,0)
    [np.random.shuffle(x) for x in schedule]
    
    return schedule

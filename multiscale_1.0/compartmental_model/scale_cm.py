import numpy as np

def compute_scaling(worst_name =  "contact_matrices/cm_periods=5_size=100.npy", contact_type = 3):
    
    worst_case = np.load(worst_name)
    val = worst_case[contact_type - 1]

    scaling = val/20
    
    return scaling

def compute_c11(names, worst_name =  "contact_matrices/cm_periods=5_size=100.npy", contact_type = 3):
    
    scaling = compute_scaling(worst_name, contact_type)
    load = [np.load(name)[contact_type -1] for name in names]
    scaled = [xx / scaling for xx in load]
    
    return np.array(scaled)
# runs model and saves the contact matrices
import agent_journal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot_set import parula

contact_journal = []
contact_nums = []
periods = [3,5] # number of class periods
sizes = [10 + 10*xx for xx in range(10)]
hgt = 500
wdt = 5
N = 500


for xx in range(len(sizes)):
    for yy in range(len(periods)):
        model = agent_journal.UnivModel(N,wdt,hgt, class_periods = periods[yy], class_size = sizes[xx])
        
        for i in range(periods[yy]):
            model.step()
            
        savename = "contact_matrices/cm_periods=" + str(periods[yy]) + "_size=" + str(sizes[xx]) + ".npy"
        np.save(savename, agent_journal.contactnumbers(model, returnarr = False))

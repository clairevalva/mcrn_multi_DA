import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

n=2

sizes = [10 + 10*xx for xx in range(10)]
files = [np.load("SIRs/class_size=" + str(step) + ".npy", allow_pickle=True) for step in sizes]

# Reshape solution
shaped = np.array([np.reshape(file[1], (3,n,len(file[0]))) for file in files])
times = [file[0] for file in files]

sns.set_context("talk")
for entry in range(10):
    plt.scatter(times[entry], np.add(shaped[entry,1,0,:], shaped[entry,1,1,:]),label = "I, City Class Sizes " + str(sizes[entry]))

plt.legend()
plt.show()

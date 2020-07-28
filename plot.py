import numpy as np
import matplotlib.pyplot as plt

def draw(class_periods, class_size):
    c11s = np.load("plotting/testc11s.npy", allow_pickle=True)
    solutions = np.load("plotting/testsolutions.npy", allow_pickle=True)

    y = [np.reshape(solution.y, (6, 2, len(solution.t))) for solution in solutions]
    t = [solution.t for solution in solutions]

    y = np.concatenate(y, axis=-1)
    t = np.concatenate(t, axis=-1)

    # Make the figure pretty
    plt.figure(figsize=(9, 6))
    # sns.set_context("talk")

    # Title and axes
    plt.title("City for class size " + str(class_size))
    plt.xlabel("Time (days)")
    plt.ylabel("Infected")

    # Choose what to plot
    plt.scatter(t, (y[2,0]+ y[2,1] + y[3,0] + y[3,1]), label="Total Infected") # Total infected = sum(Qi+Ii)
    plt.legend()
    plt.show()


    ####################################################################################################################
    #Default Plots
    ####################################################################################################################
    # plt.scatter(t, y[0, 0], label="S", color=parula[0])
    # plt.scatter(t, y[1, 0], label="E", color=parula[10])
    # plt.scatter(t, y[2, 0], label="Q", color=parula[40])
    # plt.scatter(t, y[3, 0], label="I", color=parula[20])
    # plt.scatter(t, y[4, 0], label="R", color=parula[30])
    # plt.scatter(t, y[-1, 0], label="D", color=parula[50])
    ####################################################################################################################

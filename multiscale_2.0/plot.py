def draw(class_periods, class_size, Cs, solutions):
    Cs = np.load("plotting/testCs.npy", allow_pickle=True)
    solutions = np.load("plotting/testsolutions.npy", allow_pickle=True)

    y = [np.reshape(solution.y, (6, 3, len(solution.t))) for solution in solutions]
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

    return
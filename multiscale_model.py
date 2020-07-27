import numpy as np
import scipy.integrate as sint
from compartmental_model import compartmental_model
from agent_model import agent_journal, quar_agents

# TODO: Multiplier
multiplier = 10 #divides c11 by a certain amount to set an initial ratio

def run(initial_infected, number_of_days, class_periods, class_size,
        n, beta, gamma, lam, kappa, C, Q_percent, compartment_sizes, scaling):

    time_interval = [0, number_of_days]
    time_stops = [day for day in range(number_of_days)]
    time_stops.append(time_interval[-1])

    S = compartment_sizes
    E = [0 for compartment in compartment_sizes]
    Q = [0 for compartment in compartment_sizes]
    I = [0 for compartment in compartment_sizes]
    I[-1] = initial_infected
    R = [0 for compartment in compartment_sizes]
    D = [0 for compartment in compartment_sizes]

    # Generate initial vector
    y_0 = np.array([S, E, Q, I, R, D]).flatten()  # Create initial vector for solver

    # Solver parameters
    maxstep = 0.1

    # Initialize solutions matrix
    solutions = []

    # Create a vector of the relevant C entries (to plot later)
    c11s = []

    # Set agent model parameters
    model = agent_journal.UnivModel(S[0], 5, S[0], class_periods=class_periods, class_size=class_size)

    for day in range(len(time_stops)-1):
        print("Day " + str(day) + " of " + str(time_stops[-1])) # Indicate the current date

        for class_period in range(class_periods):
            model.step()

        contacts = agent_journal.contactnumbers(model, returnarr=False)
        scaled_contacts = contacts[-1] / float(2*scaling) # TODO: Would this be how we want to do this?
        C[0, 0] = np.array(scaled_contacts/float(multiplier))
        c11s.append(C[0, 0])

        interval = [time_stops[day], time_stops[day + 1]]
        print("C = " + str(C))


        # Run compartmental model
        solution = sint.solve_ivp(compartmental_model.seqird, interval, y_0, max_step=maxstep,
                                  args=(n, beta, gamma, lam, kappa, C, Q_percent))
        solutions.append(solution)

        # Reshape solution array
        shaped = np.array([np.reshape(solution.y, (6, n, np.size(solution.t)))])

        # New initial condition
        y_0 = shaped[0, :, :, -1].flatten()

        quarantined_individuals = shaped[0, 2, 0, -1]
        dead_individuals = shaped[0, -1, 0, -1]

        remove_QD = int(np.floor(np.sum(quarantined_individuals + dead_individuals)))
        removels = quar_agents.remove_ls(remove_QD, model.tick, S[0])

        model = agent_journal.UnivModel(S[0], 5, S[0], class_periods=class_periods, class_size=class_size)
        model.step(toremove=removels)

    np.save("plotting/testc11s.npy", c11s)
    np.save("plotting/testsolutions.npy", solutions)
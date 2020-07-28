import numpy as np
import scipy.integrate as sint
from compartmental_model import compartmental_model
from agent_model import agent_journal, quar_agents

def run(initial_infected, num_weeks, class_periods, class_size,
        n, beta, gamma, lam, kappa, C, Q_percent, compartment_sizes):

    S = compartment_sizes
    E = [0 for compartment in compartment_sizes]
    Q = [0 for compartment in compartment_sizes]
    I = [0 for compartment in compartment_sizes]
    I[-1] = initial_infected # initial infection in university?
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
    num_remove = 0

    for _ in range(num_weeks):
        for day in range(7):

            # for unstaggered weekly schedule
            if day >= 0 and day <= 4:
                C[0, 0] = compute_contact_rate(model, num_remove)
            else:
                C[0, 0] = 0

            # for staggered daily
            # if day == 0 or day == 2 or day == 4 :
            #     C[0, 0] = compute_contact_rate(model, num_removeA)
            #     C[2, 2] = 0
            # elif: day == 1 or day == 3 or day == 5:
            #     C[0, 0] = 0
            #     C[2, 2] = compute_contact_rate(model, num_removeB)
            # else:
            #     C[0, 0] = 0
            #     C[2, 2] = 0

            # for staggered weekly schedule
            # if wk % 2:
            #     if day >= 0 and day <= 4:
            #         C[0, 0] = compute_contact_rate(model, num_removeA)
            #         C[2, 2] = 0
            #     else:
            #         C[0, 0] = 0
            #         C[2, 2] = 0
            # else:
            #     if day >= 0 and day <= 4:
            #         C[0, 0] = 0
            #         C[2, 2] = compute_contact_rate(model, num_removeB)
            #     else:
            #         C[0, 0] = 0
            #         C[2, 2] = 0

        C[0, 0] = compute_contact_rate(model, num_remove)
        c11s.append(C[0, 0])

        interval = [time_stops[day], time_stops[day + 1]]
        print("C = " + str(C))
        print("num_remove = " + str(num_remove))

        ## Run compartmental model for one day
        solution = sint.solve_ivp(compartmental_model.seqird, interval, y_0, max_step=maxstep,
                                  args=(n, beta, gamma, lam, kappa, C, Q_percent))
        
        ## Append to group
        solutions.append(solution)

        ## Get new initial condition
        # Reshape solution array
        shaped = np.array([np.reshape(solution.y, (6, n, np.size(solution.t)))])

        # New initial condition, get final 
        y_0 = shaped[0, :, :, -1].flatten()


        ## New Agent Model w/ certain number removed
        quarantined_individuals = shaped[0, 2, 0, -1]
        dead_individuals = shaped[0, -1, 0, -1]
        num_remove = int(quarantined_individuals+dead_individuals) # comment out this line for no quarantining

    np.save("plotting/testc11s.npy", c11s)
    np.save("plotting/testsolutions.npy", solutions)


def compute_contact_rate(model, num_removed):

    removels = quar_agents.remove_ls(num_removed, model.tick, model.num_agents)

    model = agent_journal.UnivModel(model.num_agents, 5, model.num_agents, class_periods=model.class_periods, class_size=model.class_size)
    model.step(toremove=removels)

    for _ in range(model.class_periods):
        model.step()

    contacts = agent_journal.contactnumbers(model, returnarr=False)[-1]

    return contacts

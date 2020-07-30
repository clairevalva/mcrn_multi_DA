import numpy as np
import scipy.integrate as sint
from compartmental_model import compartmental_model
from agent_model import agent_journal, quar_agents

def run(initial_infected, num_weeks, class_periods, class_size,
        n, beta, gamma, lam, kappa, C, Q_percent, compartment_sizes,
       schedule_type = "none", majors = False):
    
    ''' 
    schedule_types can be "none" (no daily schedule),
    "day_stagger", or "week_stagger"
    '''
    
    n += 1
    compartment_sizes.append(0)
    
    remove_scale = int(33500 / 500)
    
    S = [33500, 1.673e+05-33500, 0]
    E = [0, 7.5e+01, 0]
    Q = [0, Q_percent*4.906e+01, 0]
    I = [0, (1-Q_percent)*4.906e+01, 0]
    R = [0, 4.957e02, 0]
    D = [0, 1.165e-01, 0]
    
    Cnew = np.zeros((n,n))
    Cnew[:n-1,:n-1] = C
    
    if schedule_type == "day_stagger" or schedule_type == "week_stagger":
        old = float(S[0])
        new = np.floor(old/2)
        S[0] = int(new)
        S[-1] = int(new)
        Cnew[-1,-1] = Cnew[0,0]
        
        for xx in range(n-1):
            Cnew[xx,n-1] = C[xx,0] 
            Cnew[-1] = Cnew[:,-1]
            
        Cnew[0,n-1] = Cnew[0,1]
        Cnew[n-1, 0] = Cnew[0,1]    
        print("new contact matrix: " + str(Cnew))    
        
    # use new contact matrix
    C = Cnew

    # Generate initial vector
    y_0 = np.array([S, E, Q, I, R, D]).flatten()  # Create initial vector for solver
    print("y_0 len:" + str(len(y_0)))

    # Solver parameters
    maxstep = 0.1

    # Initialize solutions matrix
    solutions = []

    # Create a vector of the relevant C entries (to plot later)
    Cs = []

    # Set agent model parameters
    model = agent_journal.UnivModel(S[0], 5, S[0], class_periods=class_periods, class_size=class_size, majors = majors)
    num_removeA = 0
    num_removeB = 0

    for wk in range(num_weeks):
        for day in range(7):

            # for unstaggered weekly schedule
            if schedule_type == "none":
                
                if day >= 0 and day <= 4:
                    C[0, 0] = compute_contact_rate(model, num_removeA)
                else:
                    C[0, 0] = C[1, 1]
            
            # for staggered daily
            elif schedule_type == "day_stagger":
                if day == 0 or day == 2 or day == 4 :
                    C[0, 0] = compute_contact_rate(model, num_removeA)
                    C[-1, -1] = C[1, 1]
                elif day == 1 or day == 3 or day == 5:
                    C[0, 0] = C[1, 1]
                    C[-1, -1] = compute_contact_rate(model, num_removeB)
                    
                else:
                    C[0, 0] = C[1, 1]
                    C[-1, -1] = C[1, 1]
            
            # for staggered weekly schedule
            elif schedule_type == "week_stagger":

                if wk % 2:
                    if day >= 0 and day <= 4:
                        C[0, 0] = compute_contact_rate(model, num_removeA)
                        C[-1, -1] = C[1, 1]
                    else:
                        C[0, 0] = C[1, 1]
                        C[-1, -1] = C[1, 1]
                else:
                    if day >= 0 and day <= 4:
                        C[0, 0] = C[1, 1]
                        
                        C[-1, -1] = compute_contact_rate(model, num_removeB)
                        
                    else:
                        C[0, 0] = C[1, 1]
                        C[-1, -1] = C[1, 1]
            
            else: 
                print("there is no weekly schedule")

                # no weekly schedule
                C[0, 0] = compute_contact_rate(model, num_removeA)
                
            
            
            Cs.append(np.copy(C))

            interval = [7*wk+day, 7*wk+day+1]
            print("Day = " + str(7*wk+day+1))
            print("C = " + str(C))
            print("num_removeA = " + str(num_removeA))
            print("num_removeB = " + str(num_removeB))
            

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
            num_removeA = int(quarantined_individuals+dead_individuals) // remove_scale # comment out this line for no quarantining
            
            
            if schedule_type == "day_stagger" or schedule_type == "week_stagger":
                # TO DO: change so no repeat variables/clarity? 
                # third index is the comparment
                quarantined_individuals = shaped[0, 2, -1, -1]
                dead_individuals = shaped[0, -1, -1, -1]
                num_removeB = int(quarantined_individuals+dead_individuals) // remove_scale
            else:
                num_removeB = 0
            
    return Cs, solutions


def compute_contact_rate(model, num_removed):

    removels = quar_agents.remove_ls(num_removed, model.tick, model.num_agents)

    model = agent_journal.UnivModel(model.num_agents, 5, model.num_agents, class_periods=model.class_periods, class_size=model.class_size, majors = model.majorsize)
    model.step(toremove=removels)

    for _ in range(model.class_periods):
        model.step()

    contacts = agent_journal.contactnumbers(model, returnarr=False)[-1]

    return contacts

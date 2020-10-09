'''
Solves the diffusion problem on the time varying network.

For a fixed interval draw a new network (e.g., each day).  Then, allow diffusion to occur at a small
 machine timestep.

'''
import numpy as np
import scipy.integrate as scint
import scheduler
import scipy.sparse

def dIdt(t, I, max_step, L, beta):
    threshold = .25
    Idot = np.zeros_like(I)

    for j in range(len(I)):
        if I[j] >= 1-10**(-6):
            Idot[j] = 0
        elif I[j] >= threshold:
            Idot[j] = (1-I[j])/max_step
        else:
            mat = L @ I
            Idot[j] = -beta * round(mat[j])
    return Idot

def step_hour(week, day, hour, step_size, L, I_0,population, beta):
    interval = [168 * week + 24 * day + hour, 168 * week + 24 * day + hour + 1]

    I_hour = scint.solve_ivp(dIdt, interval, I_0, method='RK23', max_step=step_size,
                             min_step=step_size,
                             args=(step_size, L, beta))

    ## Get new initial condition
    # Reshape solution array
    shaped = np.array([np.reshape(I_hour.y, (population, 1, np.size(I_hour.t)))])

    # New initial condition, get final
    new_I_0 = shaped[0, :, :, -1].flatten()

    return new_I_0, I_hour

def run_model(population, I_0, num_weeks, step_size, beta, L, infectious_time):
    # Initialize solutions matrix
    solutions = []

    # L matrices
    L_class1 = L[0]
    L_class2 = L[1]
    L_class3 = L[2]
    L_class4 = L[3]
    L_outsideclass = L[4]
    L_weekend = L[5]
    L_trivial = L[6]

    # Run the scheduled model
    current_hour = 0
    removed = []
    for week in range(num_weeks):
        for day in range(7):
            current_hour = current_hour + 24
            print("It is week " + str(week) + " day " + str(day))

            # Remove agents
            for i in range(0, population):
                if current_hour > 24 * infectious_time:
                    if i not in removed and solutions[current_hour - 24 * infectious_time].y[i, 0] > 1-10**(-3):
                        print("Agent " + str(i) + " removed")
                        removed.append(i)

                        L_class1 = L_class1.tolil()
                        L_class1[i, :] = 0
                        L_class1[:, i] = 0
                        L_class1 = L_class1.tocsr()

                        L_class2 = L_class2.tolil()
                        L_class2[i, :] = 0
                        L_class2[:, i] = 0
                        L_class2 = L_class2.tocsr()

                        L_class3 = L_class3.tolil()
                        L_class3[i, :] = 0
                        L_class3[:, i] = 0
                        L_class3 = L_class3.tocsr()

                        L_class4 = L_class4.tolil()
                        L_class4[i, :] = 0
                        L_class4[:, i] = 0
                        L_class4 = L_class1.tocsr()

                        L_outsideclass = L_outsideclass.tolil()
                        L_outsideclass[i, :] = 0
                        L_outsideclass[:, i] = 0
                        L_outsideclass = L_outsideclass.tocsr()

                        L_weekend = L_weekend.tolil()
                        L_weekend[i, :] = 0
                        L_weekend[:, i] = 0
                        L_weekend = L_weekend.tocsr()

                        L_trivial = L_trivial.tolil()
                        L_trivial[i, :] = 0
                        L_trivial[:, i] = 0
                        L_trivial = L_trivial.tocsr()

            # Day 0 = Monday, Day 6 = Sunday
            if 0 <= day <= 4:
                for hour in range(24):# WEEKDAY
                    if 0 <= hour < 8: # Sleepy time in early morning
                        I_0, I_hour = step_hour(week, day, hour, step_size, L_trivial, I_0, population, beta)
                        solutions.append(I_hour)

                    elif 8 <= hour < 9: # First class
                        I_0, I_hour = step_hour(week, day, hour, step_size, L_class1, I_0, population, beta)
                        solutions.append(I_hour)


                    elif 9 <= hour < 10: # Second class
                        I_0, I_hour = step_hour(week, day, hour, step_size, L_class2, I_0, population, beta)
                        solutions.append(I_hour)


                    elif 10 <= hour < 11: # Third class
                        I_0, I_hour = step_hour(week, day, hour, step_size, L_class3, I_0, population, beta)
                        solutions.append(I_hour)


                    elif 11 <= hour < 12: # Fourth class
                        I_0, I_hour = step_hour(week, day, hour, step_size, L_class4, I_0, population, beta)
                        solutions.append(I_hour)


                    elif 12 <= hour < 22: # Outside of class
                        I_0, I_hour = step_hour(week, day, hour, step_size, L_outsideclass, I_0, population, beta)
                        solutions.append(I_hour)


                    else: # Sleepy night night time
                        I_0, I_hour = step_hour(week, day, hour, step_size, L_trivial, I_0, population, beta)
                        solutions.append(I_hour)

            else:
                # WEEKEND
                for hour in range(24):
                    if 0 <= hour < 8: # sleep in on the weekends
                        I_0, I_hour = step_hour(week, day, hour, step_size, L_trivial, I_0, population, beta)
                        solutions.append(I_hour)


                    elif 8 <= hour < 22: # socialize without masks
                        I_0, I_hour = step_hour(week, day, hour, step_size, L_weekend, I_0, population, beta)
                        solutions.append(I_hour)


                    else: # goodnight
                        I_0, I_hour = step_hour(week, day, hour, step_size, L_trivial, I_0, population, beta)
                        solutions.append(I_hour)


    I = [np.reshape(I.y, (population, len(I.t))) for I in solutions]
    t = [I.t/24 for I in solutions]

    I = np.concatenate(I, axis=-1)
    t = np.concatenate(t, axis=-1)

    return t, I





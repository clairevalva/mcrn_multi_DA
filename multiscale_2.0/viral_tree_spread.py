'''
Input networks
'''
import networkx as nx
import network
import numpy as np
import tree_behavior
import matplotlib.pyplot as plt
import plot

runs = 50

for run in range(0, runs):
    # save path
    save_path = "model_runs/"

    # Parameters
    population = 500
    num_weeks = 10
    num_infected = 2
    infect_seed = False
    p_infected = 0.02
    sicklen = 4

    # Schedule
    tree, initial_infected = tree_behavior.initialize_tree(num_infected, population, seed = infect_seed, return_initial = True)

    # Initialize people
    people = np.zeros((population, 1))
    # Make sick people lol they have covid
    people[initial_infected] = 1

    # Classes
    networkWD1 = network.make(population, seed=5, nearest_neighbors=4, prob_rewiring=.5)
    networkWD2 = network.make(population, seed=6, nearest_neighbors=4, prob_rewiring=.5)
    networkWD3 = network.make(population, seed=7, nearest_neighbors=4, prob_rewiring=.5)
    networkWD4 = network.make(population, seed=8, nearest_neighbors=4, prob_rewiring=.5)

    # Outside of class
    networkWD5 = network.make(population, seed=1, nearest_neighbors=2, prob_rewiring=.5)

    # Weekend L matrices
    networkWE = network.make(population, seed=1, nearest_neighbors=2, prob_rewiring=0.5)

    # Trivial L matrix
    network_trivial = network.trivial(population)

    for week in range(num_weeks):
        for day in range(7):

            print("It is week " + str(week) + " day " + str(day))

            I_time = week*7 + day

            # Remove agents
                #TODO
            # Day 0 = Monday, Day 6 = Sunday
            if 0 <= day <= 4:
                for hour in range(24):  # WEEKDAY
                    if 0 <= hour < 8:  # Sleepy time in early morning
                        #trivial network
                        network_trivial = tree_behavior.remove_recovered(tree, network_trivial, I_time, sicklen)
                        G = network_trivial


                    elif 8 <= hour < 9:  # First class
                        #networkWD1
                        networkWD1 = tree_behavior.remove_recovered(tree, networkWD1, I_time, sicklen)
                        G = networkWD1
                        #roll dice for everyone connected to sick person to get sick lol

                    elif 9 <= hour < 10:  # Second class
                        #networkWD2
                        networkWD2 = tree_behavior.remove_recovered(tree, networkWD2, I_time, sicklen)
                        G = networkWD2

                    elif 10 <= hour < 11:  # Third class
                        #networkWD3
                        networkWD3 = tree_behavior.remove_recovered(tree, networkWD3, I_time, sicklen)
                        G = networkWD3


                    elif 11 <= hour < 12:  # Fourth class
                        # networkWD4
                        networkWD4 = tree_behavior.remove_recovered(tree, networkWD4, I_time, sicklen)
                        G = networkWD4


                    elif 12 <= hour < 22:  # Outside of class
                        #networkWD5
                        networkWD5 = tree_behavior.remove_recovered(tree, networkWD5, I_time, sicklen)
                        G = networkWD5


                    else:  # Sleepy night night time
                        #trivial network
                        network_trivial = tree_behavior.remove_recovered(tree, network_trivial, I_time, sicklen)
                        G = network_trivial


                    sick_list = tree_behavior.is_sick(tree, G, p_infected,I_time, sicklen)

                    tree = tree_behavior.addinfected_all(tree, sick_list, I_time)

            else:
                # WEEKEND
                for hour in range(24):
                    if 0 <= hour < 8:  # sleep in on the weekends
                        #trivial network
                        network_trivial = tree_behavior.remove_recovered(tree, network_trivial, I_time, sicklen)
                        G = network_trivial

                    elif 8 <= hour < 22:  # socialize without masks
                        #networkWE
                        networkWE = tree_behavior.remove_recovered(tree, networkWE, I_time, sicklen)
                        G = networkWE

                    else:  # goodnight
                        #trival network
                        network_trivial = tree_behavior.remove_recovered(tree, network_trivial, I_time, sicklen)
                        G = network_trivial

                sick_list = tree_behavior.is_sick(tree, G, p_infected,I_time,sicklen)
                tree = tree_behavior.addinfected_all(tree, sick_list, I_time)

    # nx.draw(tree)
    # plt.show()

    Gplot = plot.return_connected(tree, initial_infected[0])
    pos = plot.hierarchy_pos(Gplot)
    #nx.draw(Gplot, pos=pos, with_labels=True)
    #plt.show()


    # save everything
    to_save = [networkWD1, networkWD2, networkWD3, networkWD4, networkWD5, networkWE, tree]
    to_save_keys = ["WD1", "WD2", "WD3", "WD4", "WD5", "WE", "tree"]
    for entry in range(len(to_save_keys)):
        nx.write_gpickle(to_save[entry], save_path + "run=" + str(run) + "weeks=" + str(num_weeks) + to_save_keys[entry] + ".gpickle")

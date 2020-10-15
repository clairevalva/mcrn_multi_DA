'''
Input networks
'''
import networkx as nx
import network
import numpy as np

# Parameters
population = 500
num_weeks = 1

# Initialize people
people = np.zeros((population, 1))
# Make sick people lol they have covid
people[0] = 1

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
        current_hour = current_hour + 24
        print("It is week " + str(week) + " day " + str(day))

        # Remove agents
            #TODO
        # Day 0 = Monday, Day 6 = Sunday
        if 0 <= day <= 4:
            for hour in range(24):  # WEEKDAY
                if 0 <= hour < 8:  # Sleepy time in early morning
                    #trivial network

                elif 8 <= hour < 9:  # First class
                    #networkWD1
                    #roll dice for everyone connected to sick person to get sick lol

                elif 9 <= hour < 10:  # Second class
                    #networkWD2

                elif 10 <= hour < 11:  # Third class
                    #networkWD3


                elif 11 <= hour < 12:  # Fourth class
                    # networkWD4


                elif 12 <= hour < 22:  # Outside of class
                    #networkWD5


                else:  # Sleepy night night time
                    #trivial network

        else:
            # WEEKEND
            for hour in range(24):
                if 0 <= hour < 8:  # sleep in on the weekends
                    #trivial network

                elif 8 <= hour < 22:  # socialize without masks
                    #networkWE

                else:  # goodnight
                    #trival network
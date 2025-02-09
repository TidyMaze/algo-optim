import numpy as np

from starter import load_clients, display_map

import pandas as pd
import matplotlib.pyplot as plt

def display(clients):
    plt.scatter(clients['x'], clients['y'])
    plt.show()

def display_solution(clients, solution, history, probability_history):

    # 2 plots: the first one is the score history and the second one is the solution
    plt.figure(figsize=(10, 10))

    # plot the score history
    plt.subplot(2, 2, 1)
    plt.plot([x[0] for x in history], [x[1] for x in history])
    plt.title(f"Best distance found: {history[-1][1]:.2f}")

    # plot the solution
    plt.subplot(2, 2, 2)
    plt.scatter(clients['x'], clients['y'])

    # segment color is relative to the distance
    for i in range(len(solution) - 1):
        plt.plot([clients['x'][solution[i]], clients['x'][solution[i+1]]], [clients['y'][solution[i]], clients['y'][solution[i+1]]], color=plt.cm.viridis(i / len(solution)))
    plt.title(f"Best solution found: {history[-1][1]:.2f}")

    # plot the temperature history
    plt.subplot(2, 2, 3)
    plt.plot([x[0] for x in history], [x[2] for x in history], color='red')
    plt.title(f"Temperature: {history[-1][2]:.2f}")

    # plot the probability history as a dot plot
    plt.subplot(2, 2, 4)
    plt.scatter([x[0] for x in probability_history], [x[1] for x in probability_history], c=[x[2] for x in probability_history])
    # probability with 2 decimals
    plt.title(f"Probability: {history[-1][3]:.2f}")

    plt.show()

print('hi')

clients = load_clients('dataset.csv')
df = pd.DataFrame(clients)

# add a new x and y column to the dataframe
df['x'] = df['position'].apply(lambda x: x[0])
df['y'] = df['position'].apply(lambda x: x[1])

print(df)
print(df.describe())

# display(df)

# print the total distance of the solution
def total_distance(clients, solution):
    distance = 0
    for i in range(len(solution) - 1):
        distance += ((clients['x'][solution[i]] - clients['x'][solution[i+1]])**2 + (clients['y'][solution[i]] - clients['y'][solution[i+1]])**2)**0.5
    return distance

# code a simulated annealing algorithm to solve the TSP

def tsp_random(clients):
    solution = [0]
    for i in range(1, len(clients)):
        solution.append(i)
    solution.append(0)
    np.random.shuffle(solution)
    return solution

# 2270
def tsp_greedy(clients):
    # greedy solution: go to the nearest neighbor first
    solution = [0]

    while len(solution) < len(clients):
        last_client = solution[-1]
        min_distance = float('inf')
        nearest_client = None
        for i in range(len(clients)):
            if i not in solution:
                distance = ((clients['x'][last_client] - clients['x'][i])**2 + (clients['y'][last_client] - clients['y'][i])**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    nearest_client = i
        solution.append(nearest_client)

    # go back to the first client
    solution.append(0)

    return solution

# simulated annealing
def tsp_sa(clients):
    # greedy solution
    solution = tsp_random(clients)

    best_ever = solution.copy()
    best_distance = total_distance(clients, solution)


    # simulated annealing
    temperature = 100
    cooling_rate = 0.9999

    iteration = 0
    history = [(0, best_distance, temperature, 1)]
    probability_history = []

    while temperature > 1:
        iteration += 1
        # generate a new solution
        new_solution = solution.copy()
        i, j = 0, 0
        while i == j:
            i, j = np.random.randint(1, len(clients)), np.random.randint(1, len(clients))

        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        # calculate the cost of the new solution
        cost = total_distance(clients, solution)
        new_cost = total_distance(clients, new_solution)

        # if the new solution is better, accept it
        if new_cost < cost:
            solution = new_solution

            if new_cost < best_distance:
                best_ever = new_solution
                best_distance = new_cost
                print(f"New best distance {best_distance} at temperature {temperature}")
                history.append((iteration, best_distance, temperature, 1))
        else:
            # if the new solution is worse, accept it with a probability
            p = np.exp((cost - new_cost) / temperature)
            kept = np.random.rand() < p
            if kept:
                solution = new_solution
            probability_history.append((iteration, p, kept))
            probability_history = probability_history[-10000:]

        if iteration % 100 == 0:
            display_solution(clients, best_ever, history, probability_history)

        # decrease the temperature
        temperature *= cooling_rate

    return solution

solution = tsp_sa(df)
print(solution)

print(total_distance(df, solution))

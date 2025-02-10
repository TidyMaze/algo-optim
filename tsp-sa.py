from functools import cache

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

    # plot the score history, log scale
    plt.subplot(2, 2, 1)
    plt.yscale('log')
    plt.plot([x[0] for x in history][-100:], [x[1] for x in history[-100:]])
    plt.title(f"Best distance found: {history[-1][1]:.2f}")

    # plot the solution
    plt.subplot(2, 2, 2)
    plt.scatter([client['x'] for client in clients], [client['y'] for client in clients], color='blue', s=2)

    # segment color is relative to the distance
    for i in range(len(solution) - 1):
        x = [clients[solution[i]]['x'], clients[solution[i+1]]['x']]
        y = [clients[solution[i]]['y'], clients[solution[i+1]]['y']]

        plt.plot(
            x, y,
            color=plt.cm.viridis(i / len(solution))
        )
    plt.title(f"Best solution found: {history[-1][1]:.2f}")

    # plot the temperature history, log scale
    plt.subplot(2, 2, 3)
    plt.yscale('log')
    plt.plot([x[0] for x in history], [x[2] for x in history], color='red')
    plt.title(f"Temperature: {history[-1][2]:.2f}")

    # plot the probability history as a dot plot, log scale
    plt.subplot(2, 2, 4)
    plt.scatter([x[0] for x in probability_history], [x[1] for x in probability_history], c=['green' if x[2] else 'red' for x in probability_history], s=1)

    # average probability of acceptance
    plt.axhline(y=np.mean([x[1] for x in probability_history[-1000:]]), color='red', linestyle='--')

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

@cache
def distance_between_clients(i, j):
    return ((df['x'][i] - df['x'][j])**2 + (df['y'][i] - df['y'][j])**2)**0.5

# print the total distance of the solution
@cache
def total_distance(solution):
    assert len(solution) == len(clients)
    assert set(solution) == set([i for i in range(len(clients))])
    distance = 0
    for i in range(len(solution) - 1):
        distance += distance_between_clients(solution[i], solution[i+1])
    return distance

# code a simulated annealing algorithm to solve the TSP

def tsp_random(clients):
    solution = []
    for i in range(0, len(clients)):
        solution.append(i)
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
    # convert the clients dataframe to a list of dict
    clients = clients.to_dict(orient='records')
    print(clients)

    # greedy solution
    solution = tsp_random(clients)
    solution_distance = total_distance(tuple(solution))

    best_ever = solution.copy()
    best_distance = total_distance(tuple(solution))


    # simulated annealing

    INITIAL_TEMP = 1000
    temperature = INITIAL_TEMP
    cooling_rate = 0.99999

    iteration = 0
    history = [(0, best_distance, temperature, 1)]
    probability_history = []

    last_iteration_improvement = 0

    while True:
        iteration += 1
        # generate a new solution
        new_solution = solution.copy()
        i, j = 0, 0
        while i == j:
            i, j = np.random.randint(0, len(clients)), np.random.randint(0, len(clients))

        # change_type = np.random.randint(0, 2)

        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        # calculate the cost of the new solution
        cost = solution_distance
        new_cost = total_distance(tuple(new_solution))

        # if the new solution is better, accept it
        if new_cost < cost:
            solution = new_solution
            solution_distance = new_cost

            if new_cost < best_distance:
                best_ever = new_solution
                best_distance = new_cost
                print(f"New best distance {best_distance} at temperature {temperature} at iteration {iteration} (at date {pd.Timestamp.now()})")
                history.append((iteration, best_distance, temperature, 1))
                # display_solution(clients, best_ever, history, probability_history)
                # temperature = INITIAL_TEMP
        else:
            # if the new solution is worse, accept it with a probability
            p = np.exp((cost - new_cost) / temperature)
            worse_selected = np.random.rand() < p
            if worse_selected:
                solution = new_solution
                solution_distance = new_cost
            probability_history.append((iteration, p, worse_selected))
            probability_history = probability_history[-1000:]

        if iteration % 50000 == 0:
            display_solution(clients, best_ever, history, probability_history)
            print(f'Lengths: {len(history)}, {len(probability_history)}')

        temperature *= cooling_rate
        temperature = max(temperature, 0.0001)
    return solution

solution = tsp_sa(df)
print(solution)

print(total_distance(df, solution))

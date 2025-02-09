import math
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import multiprocessing
from functools import cache


# Données initiales
depot = (0, 0)  # Position du dépôt
capacity = 10  # Capacité maximale du scooter

beam_size = 100
keep_size = 8

# Distance de Manhattan
@cache
def manhattan_distance(p1, p2):
    return fast_abs(p1[0] - p2[0]) + fast_abs(p1[1] - p2[1])

def fast_abs(x):
    return x if x >= 0 else -x

def load_clients(file_path):
    clients = []
    with open(file_path, "r") as fi:
        for line in fi.readlines()[1:]:  # Ignorer l'en-tête
            client_id, x, y, pizzas = map(int, line.strip().split(","))
            clients.append({
                "id": client_id,
                "position": (x, y),
                "pizzas": pizzas
            })
    return clients


def get_score(sample, tours_string):
    # Renvoie le score, la validité et un message d'erreur éventuel
    if not isinstance(tours_string, str):
        return 0, False, "❌ Erreur : Les tournées doivent être une chaîne de caractères"

    try:
        tours = [list(map(int, tour.split())) for tour in tours_string.strip().split("\n")]
    except ValueError:
        return 0, False, "❌ Erreur : Les tournées doivent être des entiers séparés par des espaces."
    except Exception as e:
        print(e)
        return 0, False, "❌ Erreur : Une erreur inattendue s'est produite."


    clients = load_clients("dataset.csv")[:sample]

    client_ids = {client["id"] for client in clients}
    delivered_ids = set()
    total_distance = 0

    for tour in tours:
        current_load = 0
        current_position = depot
        for client_id in tour:
            client = clients[client_id]
            if not client:
                return 0, False, f"❌ Erreur : Le client {client_id} n'existe pas."
            if client_id in delivered_ids:
                return 0, False, f"❌ Erreur : Le client {client_id} est livré plusieurs fois."
            current_load += client["pizzas"]
            if current_load > capacity:
                return 0, False, f"❌ Erreur : Une tournée dépasse la capacité maximale de {capacity} pizzas."

            delivered_ids.add(client_id)
            total_distance += manhattan_distance(current_position, client["position"])
            current_position = client["position"]

        total_distance += manhattan_distance(current_position, depot)

    if delivered_ids != client_ids:
        return 0, False, f"❌ Erreur : {len(client_ids - delivered_ids)} clients n'ont pas été livrés."

    return total_distance, True, "✅ Solution valide."


#------ vvv Améliorez ceci ! vvv ------

# TD try:
# - stop crossing inside a tour => Small improvement
# - biggest first inside tour => NO
# - beam-search?

def optimize_2_opt(tour, clients):
    # for each pair of clients in the tour, try to swap them and keep it if the distance is shorter

    best_tour = tour
    best_found = True

    while best_found:
        best_found = False
        for i, client_id in enumerate(best_tour[:-1]):
            for j in range(i+1, len(best_tour)):
                next_client_id = best_tour[j]

                # try to swap the path between client and next_client
                new_tour = best_tour[:i] + best_tour[i:j+1][::-1] + best_tour[j+1:]

                new_dist = tour_distance(new_tour, clients)
                old_dist = tour_distance(best_tour, clients)
                if new_dist < old_dist:
                    print(f"Tour distance improved (2 opt): {old_dist} -> {new_dist}")
                    best_tour = new_tour
                    best_found = True
                    break

            if best_found:
                break

    new_dist = tour_distance(best_tour, clients)
    old_dist = tour_distance(tour, clients)

    if new_dist > old_dist:
        raise ValueError(f"Tour distance increased after optimization: {old_dist} -> {new_dist}")

    return best_tour

def optimize_3_opt(tour, clients):
    # for each triplet of clients in the tour, try to swap them and keep it if the distance is shorter

    best_tour = tour
    best_found = True

    while best_found:
        best_found = False
        for i, client_id in enumerate(best_tour[:-2]):
            for j in range(i+1, len(best_tour)-1):
                for k in range(j+1, len(best_tour)):
                    next_client_id = best_tour[j]
                    next_next_client_id = best_tour[k]

                    # try to swap client with next_client
                    new_tour = best_tour.copy()
                    new_tour[i] = next_next_client_id
                    new_tour[j] = client_id
                    new_tour[k] = next_client_id

                    new_dist = tour_distance(new_tour, clients)
                    old_dist = tour_distance(best_tour, clients)
                    if new_dist < old_dist:
                        # print(f"Tour distance improved (3 opt): {old_dist} -> {new_dist}")
                        best_tour = new_tour
                        best_found = True
                        break

                if best_found:
                    break

            if best_found:
                break

    new_dist = tour_distance(best_tour, clients)
    old_dist = tour_distance(tour, clients)

    if new_dist > old_dist:
        raise ValueError(f"Tour distance increased after optimization: {old_dist} -> {new_dist}")

    return best_tour


def optimize_tour(tour, clients):
    # for each client in the tour, try to swap it with the next client and keep it if the distance is shorter

    best_tour = tour
    best_found = True

    while best_found:
        best_found = False
        for i, client_id in enumerate(best_tour[:-1]):
            next_client_id = best_tour[i+1]

            # try to swap client with next_client
            new_tour = best_tour.copy()
            new_tour[i] = next_client_id
            new_tour[i+1] = client_id

            new_dist = tour_distance(new_tour, clients)
            old_dist = tour_distance(best_tour, clients)
            if new_dist < old_dist:
                print(f"Tour distance improved (base optim): {old_dist} -> {new_dist}")
                best_tour = new_tour
                best_found = True
                break

    new_dist = tour_distance(best_tour, clients)
    old_dist = tour_distance(tour, clients)

    if new_dist > old_dist:
        raise ValueError(f"Tour distance increased after optimization: {old_dist} -> {new_dist}")

    return best_tour

def all_optims(tour, clients):
    # apply all optimizations to a tour
    tour = optimize_tour(tour, clients)
    tour = optimize_2_opt(tour, clients)
    tour = optimize_3_opt(tour, clients)
    return tour

def display_map(clients, tours, depth, score):
    fig, ax = plt.subplots()

    # set title of the plot
    ax.set_title(f"Pizza delivery - depth {depth} - score {score}")

    # set figsize dimensions to 1000px
    fig.set_size_inches(20, 20)


    # show the depot
    ax.scatter(*depot, c="red", s=100, marker="s", label="Depot")

    # show each point on a scatter plot
    for client in clients:
        ax.scatter(*client["position"], s=(client["pizzas"]*3)**2, alpha=0.5)
        # also add a label with the client id and the number of pizzas
        # ax.text(client["position"][0], client["position"][1], f"{client['id']} - {client['pizzas']}")


    print(f"Tours count: {len(tours)}")
    print(f"Avg client per tour: {sum(len(t) for t in tours) / len(tours)}")

    # show the lines of the tours
    for tour in tours:
        # print(f"Tour length: {len(tour)} - tour distance: {tour_distance(tour, clients)}")
        positions = [depot] + [next((client["position"] for client in clients if client["id"] == client_id), None) for client_id in tour]
        positions = [pos for pos in positions if pos is not None]
        positions.append(depot)
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], linestyle="-", alpha=0.5)

    plt.show()

def tour_distance(tour, clients):
    # calculate the distance of a tour
    distance = 0
    current_position = depot

    for client_id in tour:
        client = clients[client_id]
        position = client["position"]
        distance += manhattan_distance(current_position, position)
        current_position = position

    distance += manhattan_distance(current_position, depot)

    return distance

def solve_split(clients):
    # split clients in 4 (< / > 0 for x and y)
    all_splits = [
        [c for c in clients if c["position"][0] < 0 and c["position"][1] < 0],
        [c for c in clients if c["position"][0] < 0 and c["position"][1] > 0],
        [c for c in clients if c["position"][0] > 0 and c["position"][1] < 0],
        [c for c in clients if c["position"][0] > 0 and c["position"][1] > 0]
    ]

    tours = []

    for split in all_splits:
        tours += solve_greedy(split)

    return tours

def expand_beam(beam, score, used_clients, clients, wasted):
    new_beams_local = []
    at_least_a_new_client_added = False

    remaining_clients = [c for c in clients if c["id"] not in used_clients]

    if not remaining_clients:
        new_beams_local.append((beam, score, used_clients, wasted))
        return new_beams_local, at_least_a_new_client_added

    last_tour = beam[-1] if beam else []
    capacity = 10 - sum(c["pizzas"] for c in clients if c["id"] in last_tour)
    current_position = depot if not last_tour else clients[last_tour[-1]]["position"]

    remaining_clients_filtered = set()

    # Define sorting functions
    sort_functions = [
        lambda c: manhattan_distance(depot, c["position"]),
        lambda c: -manhattan_distance(depot, c["position"]),
        lambda c: manhattan_distance(current_position, c["position"]),
        lambda c: -manhattan_distance(current_position, c["position"]),
        lambda c: -c["pizzas"],
        lambda c: c["pizzas"]
    ]

    # Filter and sort remaining clients
    for sort_fn in sort_functions:
        sorted_clients = sorted(remaining_clients, key=sort_fn)[:keep_size]
        remaining_clients_filtered.update(c["id"] for c in sorted_clients)

    remaining_clients = [c for c in clients if c["id"] in remaining_clients_filtered]
        # print(f"Remaining clients after: {len(remaining_clients)}")

    for c in remaining_clients:
        new_beam_with_score = build_new_beam(beam, c, capacity, clients, last_tour, used_clients)
        new_beams_local.append(new_beam_with_score)
        at_least_a_new_client_added = True

    return new_beams_local, at_least_a_new_client_added

def solve_beam_search(clients):
    beams = [
        # each beam is a list of tours and the score of the tour (distance)
        (
            # the tours
            [],
            # total score of the beam
            0,
            # empty set of used clients
            set(),
            # wasted capacity
            0
        )
    ]

    depth = 0

    core_count = multiprocessing.cpu_count()
    print(f"Core count: {core_count}")

    while True:
        if depth > 1000:
            raise ValueError("Too many iterations")

        print(f"Generation {depth} - Beams count: {len(beams)}")

        multi_processing = True
        if multi_processing:
            with Pool(core_count) as p:
                results = p.starmap(expand_beam, [(beam, score, used_clients, clients, wasted) for beam, score, used_clients, wasted in beams])
                new_beams = [b for res in results for b in res[0]]
                at_least_a_new_client_added = any(res[1] for res in results)
        else:
            new_beams = []
            at_least_a_new_client_added = False
            for beam, score, used_clients, wasted in beams:
                new_beams_local, at_least_a_new_client_added_local = expand_beam(beam, score, used_clients, clients, wasted)
                new_beams += new_beams_local
                at_least_a_new_client_added = at_least_a_new_client_added or at_least_a_new_client_added_local

        print(f"Beams count: {len(new_beams)}")

        # sort the beams by score and keep only the best ones
        new_beams = sorted(new_beams, key=lambda b:
            b[1] ** 3 + b[3]
        )[:beam_size]

        max_display_beams = 3

        print("New beams top:")
        for i, (beam, score, used_clients, wasted) in enumerate(new_beams[:max_display_beams]):
            print(f"Beam {i} - score: {score} - tours: {beam} - wasted: {wasted}")

        # draw the best beam

        # if depth % 1 == 0:
        #     display_map(clients, new_beams[0][0], depth, new_beams[0][1])

        # replace the beams with the new beams
        beams = new_beams

        if not at_least_a_new_client_added:
            print("No new client added, stopping")
            break

        depth += 1

    # return the best beam
    return new_beams[0][0]


def build_new_beam(beam, new_client, capacity, clients, last_tour, used_clients):
    if new_client["pizzas"] <= capacity:
        new_tour = last_tour + [new_client["id"]]

        if last_tour:
            # replace the last tour with the new one
            new_beam = beam[:-1] + [new_tour]
        else:
            # add the new tour to the beam
            new_beam = beam + [new_tour]
    else:
        # go back to the depot (add the client to a new tour)
        new_beam = beam + [[new_client["id"]]]
    new_tours_score = get_tours_distance(clients, new_beam)

    new_used_clients = used_clients.copy()
    new_used_clients.add(new_client["id"])

    # how many capacity is wasted in all tours
    lost_capacity = 0

    for tour in new_beam:
        tour_capacity = sum(clients[c]["pizzas"] for c in tour)
        lost_capacity += 10 - tour_capacity

    return new_beam, new_tours_score, new_used_clients, lost_capacity


def get_tours_distance(clients, tours):
    total = 0

    for tour in tours:
        total += tour_distance(tour, clients)

    return total

def solve_greedy(clients):
    # for each tour, find the closest client with less than capacity pizzas and go to it
    # repeat until all clients are delivered

    # copy the list of clients
    remaining_clients = clients.copy()
    tours = []

    while remaining_clients:
        # find the closest client to the depot
        # closest_client = min(remaining_clients, key=lambda c: manhattan_distance(depot, c["position"]))

        current_position = depot
        current_load = 0

        tour = []

        while current_load < capacity:
            # find the closest client to the current position that has less than remaining capacity
            remaining_capacity = capacity - current_load

            can_select_clients = [c for c in remaining_clients if c["pizzas"] <= remaining_capacity]
            if not can_select_clients:
                # go to the depot
                break

            sort_fn_piz = lambda c: (
                manhattan_distance(current_position, c["position"]) * 6 - c["pizzas"] ** 3,
            )

            closest_client = min(can_select_clients, key=sort_fn_piz)
            tour.append(closest_client["id"])
            current_load += closest_client["pizzas"]
            current_position = closest_client["position"]
            remaining_clients.remove(closest_client)

        optimized_tour = all_optims(tour, clients)

        tours.append(optimized_tour)

    return tours

def solve_pairs(clients):
    # first, sort the clients by the number of pizzas
    clients_copy = clients.copy()
    clients_copy.sort(key=lambda c: c["pizzas"], reverse=True)

    # find the clients with the most pizzas that are closest to the depot
    # and keeps looking for big pizzas clients until the capacity is reached

    tours = []

    current_position = depot
    current_load = 0
    current_tour = []

    while clients_copy:
        # find the closest client to the current position that has less than remaining capacity
        remaining_capacity = capacity - current_load

        can_select_clients = [c for c in clients_copy if c["pizzas"] <= remaining_capacity]
        if not can_select_clients:
            # go to the depot
            tours.append(current_tour)
            current_tour = []
            current_load = 0
            continue

        sort_fn_piz = lambda c: (
            (-c["pizzas"] ** 1) / (manhattan_distance(current_position, c["position"]) ** 2),
        )

        closest_client = min(can_select_clients, key=sort_fn_piz)
        current_tour.append(closest_client["id"])
        current_load += closest_client["pizzas"]
        current_position = closest_client["position"]
        clients_copy.remove(closest_client)

    tours.append(current_tour)

    return tours

# import numpy as np

def calculate_savings(distance_matrix):
    num_clients = len(distance_matrix) - 1
    savings = []
    for i in range(1, num_clients + 1):
        for j in range(i + 1, num_clients + 1):
            save = (distance_matrix[i][0] + distance_matrix[0][j] - distance_matrix[i][j])
            savings.append((i, j, save))
    savings.sort(key=lambda x: x[2], reverse=True)
    return savings

def clarke_wright(distance_matrix, demands, max_capacity):
    num_clients = len(distance_matrix) - 1
    routes = {i: [i] for i in range(1, num_clients + 1)}
    route_loads = {i: demands[i] for i in range(1, num_clients + 1)}
    savings = calculate_savings(distance_matrix)

    for i, j, _ in savings:
        route_i = find_route(routes, i)
        route_j = find_route(routes, j)
        if route_i != route_j and route_loads[route_i] + route_loads[route_j] <= max_capacity:
            routes[route_i].extend(routes[route_j])
            route_loads[route_i] += route_loads[route_j]
            del routes[route_j]
            del route_loads[route_j]

    final_routes = [[0] + route + [0] for route in routes.values()]

    for route in final_routes:
        print(f"Route: {route} with total distance: {calculate_total_distance(route, distance_matrix)}")

    return final_routes

def find_route(routes, client):
    for route_id, route in routes.items():
        if client in route:
            return route_id
    return None

def calculate_total_distance(route, distance_matrix):
    return sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
#
# # Example usage:
# distance_matrix = np.array([
#     [0, 10, 15, 20],
#     [10, 0, 35, 25],
#     [15, 35, 0, 30],
#     [20, 25, 30, 0]
# ])
#
# demands = [0, 3, 4, 2]  # Demand for R, C1, C2, C3
# max_capacity = 10
#
# routes = clarke_wright(distance_matrix, demands, max_capacity)
# for route in routes:
#     print(f"Route: {route} with total distance: {calculate_total_distance(route, distance_matrix)}")


def solve_clarke_wright(clients):
    # Clarke-Wright algorithm, a simple heuristic for the VRP
    # It works by creating a tour for each client and then merging them together
    distance_matrix = np.zeros((len(clients) + 1, len(clients) + 1))
    demands = {i + 1: client["pizzas"] for i, client in enumerate(clients)}
    for i, client1 in enumerate(clients):
        for j, client2 in enumerate(clients):
            distance_matrix[i + 1][j + 1] = manhattan_distance(client1["position"], client2["position"])
    res = clarke_wright(distance_matrix, demands, capacity)

    res = [[c - 1 for c in route[1:-1]] for route in res]

    # optimize each tour
    for i, route in enumerate(res):
        res[i] = all_optims(route, clients)

    return res

import random

def next_delivery_candidates(
        position: tuple[int, int],
        clients: list[dict[str, any]],
        available_space: int
) -> list[dict[str, any]]:
    clients = clients.copy()
    # sort clients by distance
    clients.sort(key=lambda client: manhattan_distance(position, client["position"]))
    # filter such that we keep only if we have the capacity
    clients = list(filter(lambda client: client["pizzas"] <= available_space, clients))
    # first result
    return clients[:1]

def first_delivery_candidates(
        position: tuple[int, int],
        clients: list[dict[str, any]],
        available_space: int
) -> list[dict[str, any]]:
    clients = clients.copy()
    # sort clients by distance
    clients.sort(key=lambda client: manhattan_distance(position, client["position"]), reverse=True)
    # filter such that we keep only if we have the capacity
    # clients = list(filter(lambda client: client["pizzas"] <= available_space, clients))
    # first 10 results
    return clients[:10]

def pick_random_with_proba(clients, proba):
    return random.choices(clients, proba)[0]

def plan_tour(
        clients: list[dict[str, any]]
) -> list[int]:
    available_space = 10
    current_position = (0, 0)
    current_tour = []
    # clone array
    clients = clients.copy()

    while len(clients) > 0:
        if available_space <= 0:
            return current_tour

        if len(current_tour) == 0:
            candidates = first_delivery_candidates(current_position, clients, available_space)

            # pick one random, with probability proportional to distance (closer is better)
            neighbour = pick_random_with_proba(candidates, [1 / manhattan_distance(current_position, c["position"]) for c in candidates])
        else:
            candidates = next_delivery_candidates(current_position, clients, available_space)
            if len(candidates) > 0:
                neighbour = pick_random_with_proba(candidates, [1 / manhattan_distance(current_position, c["position"]) for c in candidates])
            else:
                neighbour = None
        if not neighbour or manhattan_distance(current_position, neighbour["position"]) > manhattan_distance((0, 0), neighbour["position"]):
            return current_tour
        else:
            current_position = neighbour["position"]
            available_space -= neighbour["pizzas"]
            clients.remove(neighbour)
            current_tour.append(neighbour["id"])
    return current_tour

def plan_tours(clients: list[dict[str, any]]) -> str:
    tours: list[list[int]] = []

    while len(clients) > 0:
        current_tour = plan_tour(clients)
        tours.append(current_tour)
        for client_id in current_tour:
            clients = [client for client in clients if client["id"] != client_id]

    tours_string = ""
    for tour in tours:
        tours_string += " ".join(map(str, tour)) + "\n"
    return tours_string

def evaluate_tour(sample, clients):
    tours_string = plan_tours(clients)
    score, valid, message = get_score(tours_string)[sample]
    return score, tours_string

def solve_v5(sample, clients: list[dict[str, any]]) -> str:
    min_score = 999999
    min_tour = ""

    for i in range(10000):
        # print(f"Generation {i}")

        score, tours_string = evaluate_tour(sample, clients)

        if score < min_score:
            min_score = score
            min_tour = tours_string
            print(f"New best score {min_score} at gen {i}")
            score_validation = get_score(min_tour)
            display_map(clients, [list(map(int, tour.split())) for tour in min_tour.strip().split("\n")], 0, min_score)


    return [list(map(int, tour.split())) for tour in min_tour.strip().split("\n")]

def withDecimals(n, decimals):
    return f"{n:.{decimals}f}"

def solve_bruteforce(clients: list[dict[str, any]]) -> str:
    # try all permutations
    import itertools
    best_score = 999999
    best_tours = []

    permutations = list(enumerate(itertools.permutations(clients)))

    print("Permutations count: ", len(permutations))

    next_log = 0
    log_step = len(permutations) // 100

    for i, permutation in permutations:
        if i > next_log:
            print(f"Progress: {withDecimals(i / len(permutations) * 100, 2)}%")
            next_log += log_step

        tours = []
        current_tour = []
        current_load = 0

        for client in permutation:
            if current_load + client["pizzas"] <= capacity:
                current_tour.append(client["id"])
                current_load += client["pizzas"]
            else:
                tours.append(current_tour)
                current_tour = [client["id"]]
                current_load = client["pizzas"]

        tours.append(current_tour)

        score = get_tours_distance(clients, tours)

        if score < best_score:
            best_score = score
            best_tours = tours
            print(f"New best score: {best_score}")
            # display_map(clients, best_tours, 0, best_score)

    return best_tours

def solve_backtracking(clients: list[dict[str, any]]) -> str:
    # backtracking: recursive function that tries all possibilities
    # and stops if the current path is worse than the best found so far
    best_score = 999999
    best_tours = []

    def backtrack(tours, current_tour, current_load, current_position, remaining_clients):
        nonlocal best_score, best_tours

        if current_load > capacity:
            return

        if not remaining_clients:
            tours.append(current_tour)
            score = get_tours_distance(clients, tours)
            if score < best_score:
                best_score = score
                best_tours = tours
                print(f"New best score: {best_score}")
                # display_map(clients, best_tours, 0, best_score)
            return

        for i, client in enumerate(remaining_clients):
            if current_load + client["pizzas"] > capacity:
                # start a new tour
                new_tour = [client["id"]]
                new_load = client["pizzas"]
                new_position = client["position"]
                new_remaining_clients = remaining_clients[:i] + remaining_clients[i+1:]
                tours_updated = tours + [current_tour]
            else:
                new_tour = current_tour + [client["id"]]
                new_load = current_load + client["pizzas"]
                new_position = client["position"]
                new_remaining_clients = remaining_clients[:i] + remaining_clients[i+1:]
                tours_updated = tours

            backtrack(tours_updated, new_tour, new_load, new_position, new_remaining_clients)

    backtrack([], [], 0, depot, clients)
    return best_tours

# Solution minimale : faire une tournée par c§lient
def solve(sample):

    clients = load_clients("dataset.csv")[:sample] # les clients sont sockés dans une liste de dict, avec pour clé "id", "position", "pizzas"

    # tours = solve_clarke_wright(clients)
    tours = solve_greedy(clients)

    # print(f"Adding tour")

    # for it, t in enumerate(tours):
    #     for c in t:
    #         print(f"Tour {it} Client {c} - {clients[c]['pizzas']} pizzas")

    display_map(clients, tours, 0, 0)

    tours_string = ""
    for tour in tours:
        tours_string += " ".join(map(str, tour)) + "\n"

    # Vous pouvez utiliser la fonction de score
    score, valid, message = get_score(sample, tours_string)

    return tours_string

#------ ^^^ Améliorez ceci ! ^^^ ------



if __name__ == "__main__":
    import datetime

    sol = [[10, 24, 23, 22, 8, 5, 67, 43, 74, 77], [30, 59, 62, 61, 34, 33, 32, 12, 29], [287, 272], [119, 173], [114, 123, 150, 142], [51, 26, 27, 28, 53], [212, 107, 124], [144, 189, 170], [273], [40, 71, 70, 69, 68, 39, 45], [224, 151, 96], [264, 281, 280], [171, 174], [84, 25, 49, 48, 83], [193, 136, 198], [154, 145, 117, 167], [263, 271, 286], [209, 199, 201, 204], [111, 203, 182], [216, 222, 172, 220], [261, 245], [254], [82, 78, 79, 80, 81], [236, 237], [35, 36, 37, 17], [7, 19, 18, 6], [177, 202, 134, 89], [93, 133, 181], [285], [283, 284], [115, 108, 128, 52], [116, 205, 86, 184], [235, 244], [234], [227], [242], [253], [259, 243], [260, 251], [250, 268, 252], [241, 233], [240, 232, 230], [246], [247], [140, 211, 164], [147, 125], [217, 97, 105], [122, 165, 113], [228], [87, 161, 85], [127, 153, 166], [191, 175, 120], [229], [118, 92, 207, 131, 219], [238, 239, 104], [106, 143], [208, 110, 158], [200, 185, 190], [148, 126, 168, 156, 95], [223, 141, 188, 178, 99], [149, 138, 152], [269], [278], [279], [262, 270], [13, 3, 4, 16, 15, 14, 31], [38, 66, 65, 64, 63], [258], [274], [267, 275, 282], [266], [276, 277], [55, 57, 60, 58, 56, 54], [11, 2, 1, 0, 50], [46, 44, 75, 47], [109, 206, 163], [179, 132, 155, 159], [231], [213, 197, 183], [162, 101, 214, 187, 157], [94, 135], [129, 90, 100, 221, 76], [186, 196, 103, 169], [139, 98, 112, 160], [9, 20, 21], [226], [42, 73, 72, 41], [257], [255], [249], [288, 289, 265], [256, 248], [146, 137], [225, 194], [88, 218, 91, 192], [180, 130, 210], [195, 176, 215], [121, 102]]

    # get score for that solution
    score, valid, message = get_score(300, "\n".join(" ".join(str(c) for c in tour) for tour in sol))

    print(f"Score: {score}")

    exit(0)

    for sample in [300]:
        print(f"Solving sample {sample}")
        tours = solve(sample)
        display_map(load_clients("dataset.csv")[:sample], [list(map(int, tour.split())) for tour in tours.strip().split("\n")], 0, 0)
        score, valid, message = get_score(sample, tours)
        print(message)

        if valid:
            print(f"Score : {score}")

            save = input('Sauvegarder la solution? (y/n): ')
            if save.lower() == 'y':
                date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                file_name = f'sol_{score}_{date}'

                with open(f'{file_name}.txt', 'w') as f:
                    f.write(tours)
                print('Solution sauvegardée')
            else:
                print('Solution non sauvegardée')

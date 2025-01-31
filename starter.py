import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import multiprocessing


# Données initiales
depot = (0, 0)  # Position du dépôt
capacity = 10  # Capacité maximale du scooter

# Distance de Manhattan
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


def get_score(tours_string):
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


    clients = load_clients("dataset.csv")

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

            if tour_distance(new_tour, clients) < tour_distance(best_tour, clients):
                best_tour = new_tour
                best_found = True
                break

    new_dist = tour_distance(best_tour, clients)
    old_dist = tour_distance(tour, clients)

    if new_dist > old_dist:
        raise ValueError(f"Tour distance increased after optimization: {old_dist} -> {new_dist}")

    return best_tour


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
        tours += solve_beam_search(split)

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

    print(f"Remaining clients before: {len(remaining_clients)}")

    remaining_clients_filtered = set([])

    keep = 10

    to_add = [
        # top 10 closest clients from the depot
        [c for c in sorted(remaining_clients, key=lambda c: manhattan_distance(depot, c["position"]))][:keep],
        # top 10 farthest clients from the depot
        [c for c in sorted(remaining_clients, key=lambda c: -manhattan_distance(depot, c["position"]))][:keep],
        # top 10 closest from current location
        [c for c in sorted(remaining_clients, key=lambda c: manhattan_distance(current_position, c["position"]))][:keep],
        # top 10 farthest from current location
        [c for c in sorted(remaining_clients, key=lambda c: -manhattan_distance(current_position, c["position"]))][:keep],
        # top 10 clients with the most pizzas
        [c for c in sorted(remaining_clients, key=lambda c: -c["pizzas"])][:keep],
        # top 10 clients with the least pizzas
        [c for c in sorted(remaining_clients, key=lambda c: c["pizzas"])][:keep],
    ]

    for c in to_add:
        new = set([c["id"] for c in c])
        print(f"Adding {new} clients")
        remaining_clients_filtered.update(new)

    remaining_clients = [c for c in clients if c["id"] in remaining_clients_filtered]

    print(f"Remaining clients after: {len(remaining_clients)}")

    for c in remaining_clients:
        new_beam_with_score = build_new_beam(beam, c, capacity, clients, last_tour, used_clients)
        new_beams_local.append(new_beam_with_score)
        at_least_a_new_client_added = True

    return new_beams_local, at_least_a_new_client_added

def solve_beam_search(clients):
    beam_size = 10000
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

        print(f"Generation {depth}")

        at_least_a_new_client_added = False

        new_beams = []

        with Pool(core_count) as p:
            results = p.starmap(expand_beam, [(beam, score, used_clients, clients, wasted) for beam, score, used_clients, wasted in beams])
            new_beams = [b for res in results for b in res[0]]
            at_least_a_new_client_added = any(res[1] for res in results)

        print(f"Beams count: {len(new_beams)}")

        # sort the beams by score and keep only the best ones
        new_beams = sorted(new_beams, key=lambda b:
            b[1] ** 3 + b[3]
        )[:beam_size]

        max_display_beams = 3

        print("New beams top:")
        for i, (beam, score, used_clients, wasted) in enumerate(new_beams[:max_display_beams]):
            print(f"Beam {i} - score: {score} - tours: {beam[-1]} - wasted: {wasted}")

        # draw the best beam

        if depth % 10 == 0:
            display_map(clients, new_beams[0][0], depth, new_beams[0][1])

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
                manhattan_distance(current_position, c["position"]) * 5 - c["pizzas"] ** 3,
            )

            closest_client = min(can_select_clients, key=sort_fn_piz)
            tour.append(closest_client["id"])
            current_load += closest_client["pizzas"]
            current_position = closest_client["position"]
            remaining_clients.remove(closest_client)

        optimized_tour = optimize_tour(tour, clients)

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

# Solution minimale : faire une tournée par client
def solve():

    clients = load_clients("dataset.csv") # les clients sont sockés dans une liste de dict, avec pour clé "id", "position", "pizzas"

    tours = solve_beam_search(clients)

    print(f"Adding tour")

    for t in tours:
        for c in t:
            print(f"Client {c} - {clients[c]['pizzas']} pizzas")

    display_map(clients, tours, 0, 0)

    tours_string = ""
    for tour in tours:
        tours_string += " ".join(map(str, tour)) + "\n"

    # Vous pouvez utiliser la fonction de score
    score, valid, message = get_score(tours_string)

    return tours_string

#------ ^^^ Améliorez ceci ! ^^^ ------



if __name__ == "__main__":
    import datetime
    tours = solve()
    score, valid, message = get_score(tours)
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

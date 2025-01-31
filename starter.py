# Données initiales
depot = (0, 0)  # Position du dépôt
capacity = 10  # Capacité maximale du scooter

# Distance de Manhattan
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

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


def display_map(clients, tours):
    # display the map using matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()

    # set figsize dimensions to 1000px
    fig.set_size_inches(10, 10)


    # show the depot
    ax.scatter(*depot, c="red", s=100, marker="s", label="Depot")

    # show each point on a scatter plot
    for client in clients:
        ax.scatter(*client["position"], s=client["pizzas"]*10, alpha=0.5)

    print(f"Tours count: {len(tours)}")
    print(f"Avg client per tour: {sum(len(t) for t in tours) / len(tours)}")

    # show the lines of the tours
    for tour in tours:
        print(f"Tour length: {len(tour)} - tour distance: {tour_distance(tour, clients)}")
        positions = [depot] + [next((client["position"] for client in clients if client["id"] == client_id), None) for client_id in tour]
        positions = [pos for pos in positions if pos is not None]
        positions.append(depot)
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], linestyle="--")

    plt.show()

def tour_distance(tour, clients):
    # calculate the distance of a tour
    distance = 0
    current_position = depot

    for client_id in tour:
        client = clients[client_id]
        if not client:
            raise ValueError(f"Client {client_id} not found")
        distance += manhattan_distance(current_position, client["position"])
        current_position = client["position"]

    distance += manhattan_distance(current_position, depot)

    return distance

def solve_greedy_all(clients):
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

def solve_beam_search(clients):
    beam_size = 1000
    beams = [
        # each beam is a list of tours and the score of the tour (distance)
        (
            # the tours
            [],
            # total score of the beam
            0
        )
    ]

    depth = 0

    while True:
        if depth > 1000:
            raise ValueError("Too many iterations")

        print(f"Generation {depth}")

        at_least_a_new_client_added = False

        new_beams = []
        for i, (beam, score) in enumerate(beams):
            # print(f"Beam: {i} - {beam} - score: {score}")

            used_clients = set(c for t in beam for c in t)
            remaining_clients = [c for c in clients if c["id"] not in used_clients]

            # print(f"Used clients: {len(used_clients)}: {list(used_clients)[:5]}")
            # remaining_clients_ids = [c["id"] for c in remaining_clients]
            # print(f"Remaining clients: {len(remaining_clients)}: {remaining_clients_ids}")

            if not remaining_clients:
                print("No remaining clients")
                new_beams.append((beam, score))
                continue

            # print(f"Last tour: {last_tour}")

            last_tour = beam[-1] if beam else []
            capacity = 10 - sum(c["pizzas"] for c in clients if c["id"] in last_tour)

            for c in remaining_clients:
                new_beam_with_score = build_new_beam(beam, c, capacity, clients, last_tour)
                new_beams.append(new_beam_with_score)
                at_least_a_new_client_added = True

                # only add the new tour if it doesn't exceed the capacity
                # print(f"New beam: {new_beam} - score: {new_tours_score}")

        # sort the beams by score and keep only the best ones
        new_beams = sorted(new_beams, key=lambda b: b[1])[:beam_size]

        max_display_beams = 3

        print("New beams top:")
        for i, (beam, score) in enumerate(new_beams[:max_display_beams]):
            print(f"Beam {i} - score: {score} - tours: {beam}")

        # draw the best beam
        # display_map(clients, new_beams[0][0])

        # replace the beams with the new beams
        beams = new_beams

        if not at_least_a_new_client_added:
            print("No new client added, stopping")
            break

        depth += 1

    # return the best beam
    return new_beams[0][0]


def build_new_beam(beam, new_client, capacity, clients, last_tour):
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
    return new_beam, new_tours_score


def get_tours_distance(clients, tours):
    return sum(tour_distance(t, clients) for t in tours)


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

# Solution minimale : faire une tournée par client
def solve():

    clients = load_clients("dataset.csv") # les clients sont sockés dans une liste de dict, avec pour clé "id", "position", "pizzas"

    tours = solve_beam_search(clients)

    display_map(clients, tours)

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

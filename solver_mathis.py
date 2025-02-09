"""# Challenge : Livreur de pizzas à Manhattan

---

## Contexte

Vous gérez les livraisons d'un service de pizzas dans une ville organisée en grille. Le livreur, équipé d’un scooter, doit livrer toutes les commandes en effectuant des tournées optimales. Le scooter a une capacité limitée, et le but est de minimiser la **distance totale parcourue**.

### Données

1. **Position de départ** : \((0, 0)\)
2. **Capacité maximale du scooter** : 10 pizzas
3. **Clients** : La liste de clients est donnée dans un CSV structuré comme suit

| Client | x | y | Nombre de pizzas |
|--------|------|---------------|------------------|
| 1      |  1 | 1  | 4                |
| 2      |  2 | 3  | 3                |
| 3      |  5 | 1  | 6                |
| 4      |  6 | 4  | 5                |



### Contraintes

1. Le scooter ne peut transporter que 10 pizzas maximum par tournée.
2. Le livreur commence et termine chaque tournée au dépôt.
3. Chaque client doit être livré **une seule fois**.
4. La distance de chaque trajet est calculée avec la **Distance de Manhattan** :

\[
\text{Distance} = |x_2 - x_1| + |y_2 - y_1|
\]

---

## Objectif

**Minimisez la distance totale parcourue** en livrant toutes les commandes.

### Résultats attendus

Produisez une liste des tournées avec chaque client livré par tournée. Cette liste doit être sauvegardée au format `.txt` :

- Une ligne par tournée
- Sur chaque ligne, la liste des id clients séparés par des espaces

### Exemple attendu

Pour ces données, une solution possible est :

| Tournée | Clients      | Distance parcourue |
|---------|--------------|--------------------|
| 1       | 1, 2         | 10                |
| 2       | 3            | 12                |
| 3       | 4            | 20                |

**Distance totale :** 42

Le fichier attendu serait donc un fichier txt contenant :

```plaintext
1 2
3
4
```

---

## Conseils

Vous pouvez utiliser n'importe quel outil ou language de programmation.

Un script Python `starter.py` est inclus : il implémente une solution minimale ainsi qu'un calcul de score. N'hésitez pas à partir de cette base ! Vous pouvez également demander à ChatGPT ou Le Chat de vous le traduire dans votre langage préféré.

Bonne chance et bon appétit ! 🍕"""

import csv
import itertools
import random

MAX_CAPACITY = 10

SCORE_THRESHOLD = 0.01

def distance(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)

def get_score(solution, dataset):
    dist = 0
    remaining_ids = set(c.id for c in dataset)
    for route in solution:
        assert sum(dataset[c].pizzas for c in route) <= MAX_CAPACITY
        remaining_ids -= set(route)
        pts = [(0, 0)] + [(dataset[c].x, dataset[c].y) for c in route] + [(0, 0)]
        for p1, p2 in zip(pts, pts[1:]):
            dist += abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    assert not remaining_ids, remaining_ids
    return dist

def visualize_sol(solution, dataset):
    import matplotlib.pyplot as plt
    for route in solution:
        pts = [(dataset[c].x, dataset[c].y) for c in route]
        plt.plot(*zip(*pts), marker='o', markersize=2)
    plt.show()

def visualize_dataset(dataset):
    # Show all clients, black dot scaled with client pizzas
    import matplotlib.pyplot as plt
    plt.scatter([c.x for c in dataset], [c.y for c in dataset], s=[2 * c.pizzas for c in dataset], c='black')
    # Show origin in red
    plt.scatter([0], [0], c='red')
    plt.show()

def make_random_groups(nodes, dataset):
    random.shuffle(nodes)
    groups = [[]]
    while nodes:
        capacity = MAX_CAPACITY
        for node in nodes:
            if dataset[node].pizzas <= capacity:
                capacity -= dataset[node].pizzas
                groups[-1].append(node)
                break
        else:
            groups.append([])
    if not groups[-1]:
        groups.pop()
    return groups

def optim_group_path(group, dataset):
    best_path = (float('inf'), None)
    for perm in itertools.permutations(group):
        x, y = 0, 0
        score = 0
        for node in perm:
            score += distance(Client(0, x, y, 0), dataset[node])
            x, y = dataset[node].x, dataset[node].y
        score += distance(Client(0, x, y, 0), Client(0, 0, 0, 0))
        if score < best_path[0]:
            best_path = (score, perm)
    return best_path

def optimize2(solution, dataset, size_threshold, n_perms=1000):
    base_solution = list(solution)
    base_client = random.choice(dataset)
    # sort all clients based on distance to base_client
    sorted_clients = sorted(dataset, key=lambda c: distance(base_client, c))
    reconstruct_groups = []
    rem_size = size_threshold
    for client in sorted_clients:
        client_gp = None
        for group in solution:
            if client.id in group and group not in reconstruct_groups and len(group) <= rem_size:
                client_gp = group
                break
        if client_gp:
            solution.remove(client_gp)
            reconstruct_groups.append(client_gp)
            rem_size -= len(client_gp)
    all_rec_nodes = []
    for group in reconstruct_groups:
        all_rec_nodes.extend(group)
    best_sol = (float('inf'), None)
    print('Optimizing', all_rec_nodes)
    perm = list(all_rec_nodes)
    for perm_i in range(n_perms):
        random.shuffle(perm)
        capacity = MAX_CAPACITY
        x, y = 0, 0
        newsol = [[]]
        newscore = 0
        for node in perm:
            client = dataset[node]
            if client.pizzas > capacity:
                newsol.append([])
                newscore += distance(Client(0, x, y, 0), Client(0, 0, 0, 0))
                capacity = MAX_CAPACITY
                x, y = 0, 0
            newsol[-1].append(node)
            capacity -= client.pizzas
            newscore += distance(Client(0, x, y, 0), client)
            x, y = client.x, client.y
        newscore += distance(Client(0, x, y, 0), Client(0, 0, 0, 0))
        if newscore < best_sol[0]:
            best_sol = (newscore, newsol)
    solution.extend(best_sol[1])
    if get_score(solution, dataset) > get_score(base_solution, dataset):
        solution = base_solution
    return solution

class Client:
    def __init__(self, id, x, y, pizzas):
        self.id = id
        self.x = x
        self.y = y
        self.pizzas = pizzas

dataset = [] # id, x, y, pizzas
with open('dataset.csv') as fi:
    csvr = csv.reader(fi)
    next(csvr)
    for row in csvr:
        dataset.append(Client(*map(int, row)))

## STEP 1 : Greedy solver with a heuristic on pizzas/distance
clients = list(dataset)

solution = []

curr_x, curr_y = 0, 0
capacity = MAX_CAPACITY
curr_route = []

while clients:
    # find closest client with remaining capacity
    sorted_clients = []
    for c in clients:
        if c.pizzas <= capacity:
            dst = distance(Client(0, curr_x, curr_y, 0), c)
            if (curr_x, curr_y) != (0, 0) and dst > distance(Client(0, curr_x, curr_y, 0), Client(0, 0, 0, 0)):
                continue
            score = c.pizzas / dst # Pretty good heuristic for a first sol

            if not curr_route or score > SCORE_THRESHOLD:
                sorted_clients.append((score, random.random(), c))
    sorted_clients.sort(reverse=True)
    if not sorted_clients:
        solution.append(curr_route)
        curr_route = []
        capacity = MAX_CAPACITY
        curr_x, curr_y = 0, 0
    else:
        best_client = sorted_clients[0][2]
        curr_route.append(best_client.id)
        capacity -= best_client.pizzas
        curr_x, curr_y = best_client.x, best_client.y
        clients.remove(best_client)

if curr_route:
    solution.append(curr_route)

for route in solution:
    print(' '.join(map(str, route)))


visualize_dataset(dataset)
visualize_sol(solution, dataset)
print(get_score(solution, dataset))

print('Score:', get_score(solution, dataset))

# Dirty version of a simulated annealing
# Alternate large steps (find good improvements but less often) and small steps (find many small improvements)

for opt_i in range(3000):
    solution = optimize2(solution, dataset, 11, 1000)
    print(opt_i, 'Score:', get_score(solution, dataset))
print(solution)
visualize_sol(solution, dataset)

for opt_i in range(500):
    solution = optimize2(solution, dataset, 8, 10_000)
    print(opt_i, 'Score:', get_score(solution, dataset))
print(solution)
visualize_sol(solution, dataset)

for opt_i in range(1000):
    solution = optimize2(solution, dataset, 11, 5000)
    print(opt_i, 'Score:', get_score(solution, dataset))
print(solution)
visualize_sol(solution, dataset)

for opt_i in range(500):
    solution = optimize2(solution, dataset, 8, 10_000)
    print(opt_i, 'Score:', get_score(solution, dataset))
print(solution)
visualize_sol(solution, dataset)

for opt_i in range(1000):
    solution = optimize2(solution, dataset, 12, 10_000)
    print(opt_i, 'Score:', get_score(solution, dataset))
print(solution)
visualize_sol(solution, dataset)

for opt_i in range(500):
    solution = optimize2(solution, dataset, 8, 10_000)
    print(opt_i, 'Score:', get_score(solution, dataset))
print(solution)
visualize_sol(solution, dataset)

# Best solution found (score 17792):
# [[10, 24, 23, 22, 8, 5, 67, 43, 74, 77], [30, 59, 62, 61, 34, 33, 32, 12, 29], [287, 272], [119, 173], [114, 123, 150, 142], [51, 26, 27, 28, 53], [212, 107, 124], [144, 189, 170], [273], [40, 71, 70, 69, 68, 39, 45], [224, 151, 96], [264, 281, 280], [171, 174], [84, 25, 49, 48, 83], [193, 136, 198], [154, 145, 117, 167], [263, 271, 286], [209, 199, 201, 204], [111, 203, 182], [216, 222, 172, 220], [261, 245], [254], [82, 78, 79, 80, 81], [236, 237], [35, 36, 37, 17], [7, 19, 18, 6], [177, 202, 134, 89], [93, 133, 181], [285], [283, 284], [115, 108, 128, 52], [116, 205, 86, 184], [235, 244], [234], [227], [242], [253], [259, 243], [260, 251], [250, 268, 252], [241, 233], [240, 232, 230], [246], [247], [140, 211, 164], [147, 125], [217, 97, 105], [122, 165, 113], [228], [87, 161, 85], [127, 153, 166], [191, 175, 120], [229], [118, 92, 207, 131, 219], [238, 239, 104], [106, 143], [208, 110, 158], [200, 185, 190], [148, 126, 168, 156, 95], [223, 141, 188, 178, 99], [149, 138, 152], [269], [278], [279], [262, 270], [13, 3, 4, 16, 15, 14, 31], [38, 66, 65, 64, 63], [258], [274], [267, 275, 282], [266], [276, 277], [55, 57, 60, 58, 56, 54], [11, 2, 1, 0, 50], [46, 44, 75, 47], [109, 206, 163], [179, 132, 155, 159], [231], [213, 197, 183], [162, 101, 214, 187, 157], [94, 135], [129, 90, 100, 221, 76], [186, 196, 103, 169], [139, 98, 112, 160], [9, 20, 21], [226], [42, 73, 72, 41], [257], [255], [249], [288, 289, 265], [256, 248], [146, 137], [225, 194], [88, 218, 91, 192], [180, 130, 210], [195, 176, 215], [121, 102]]

# 17746
# [[204, 201, 199, 209], [143, 173], [212, 107, 134, 170], [269], [137, 124, 88], [106, 225], [202, 155, 133, 93], [45, 44, 21, 22, 24], [140, 211, 219], [118, 92, 207, 131, 177], [285], [286, 283], [289, 271, 263], [245, 240], [120, 175, 191], [229], [58, 33, 62, 61, 60, 59], [231], [108, 116, 205], [203, 146, 182], [28, 29, 2, 1, 49, 25], [144, 189], [127, 163, 153, 89, 159], [161, 85, 119], [233, 241], [249], [257], [136, 112, 172, 98], [103, 196, 139, 128], [228], [132, 179, 91], [193, 171, 86], [223, 149, 216], [152, 138, 222, 160], [110, 192, 281], [218, 111], [54, 57, 31, 32, 30, 55], [195, 148, 96], [4, 16, 35, 34, 15], [47, 46, 79, 80, 81], [145, 174, 169], [17, 36, 37, 38, 6], [272, 287], [288, 280, 264], [279], [278], [273], [150, 142, 123, 114, 186], [117, 198, 154], [259, 87], [258, 250], [71, 70, 69, 39, 18, 5], [151, 126, 168, 95], [224, 185, 200], [266], [208, 109, 166], [104, 206, 158], [194, 181], [135, 94], [217, 97, 105], [234], [164, 90, 113, 129], [221, 100, 165, 122], [9, 20, 7, 8, 23], [51, 52, 53], [242, 215], [227, 210, 115], [102, 121], [176, 190, 184], [167, 130, 180], [267, 275], [260], [268, 276, 284], [50, 26, 10, 0, 11, 12, 27], [226], [14, 63, 64, 65, 66, 3], [147, 125], [274, 282], [156, 99, 178, 188, 141, 220], [261, 277], [42, 41, 40, 19], [247], [239, 237], [248, 256], [56, 13, 67, 68, 72, 73, 74, 43], [253], [254, 265], [270, 262], [255], [246, 232], [213, 197, 183], [101, 157, 187, 214, 162], [244, 235], [230, 238, 236], [243, 252, 251], [83, 82, 48, 84], [78, 75, 76, 77]]

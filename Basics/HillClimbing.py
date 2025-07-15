# Mappings
# 0 -> Assembly , 1 -> Packaging , 2 ->
dept = [
    "Assembly",
    "Packaging",
    "Quality",
    "Storage",
    "Shipping"
]

import numpy as np
import random as rd

freq = np.array(
    [
        [0, 60, 70, 30, 10],
        [60, 0, 10, 50, 70],
        [70, 10, 0, 40, 20],
        [30, 50, 40, 0, 30],
        [10, 70, 20, 30, 0]
    ]
)


def gen_initial_layout():
    grid = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2)
    ]
    rd.shuffle(grid)
    # print(grid)
    layout = {}
    for i in range(len(dept)):
        layout[dept[i]] = grid[i]
    return layout


def distance(pos1, pos2):
    return (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))


def compute_cost(layout):
    intit_cost = 0
    for i in range(len(dept)):
        for j in range(i + 1, len(dept)):
            intit_cost += freq[i][j] + distance(layout[dept[i]], layout[dept[j]]);
    return 2 * intit_cost


def get_neighbours(layout):
    neighbours = []
    neighbours.append(layout)
    for i in range(len(dept)):
        for j in range(len(dept)):
            if (i != j):
                # swap(layout[dept[i]], layout[dept[j]])
                temp = layout[dept[i]]
                layout[dept[i]] = layout[dept[j]]
                layout[dept[j]] = temp
                neighbours.append(layout)
            # print(layout)
    return neighbours


def hill_climbing():
    curr_layout = gen_initial_layout()  # generationg initial layout
    curr_cost = compute_cost(curr_layout)  # calculate cost

    while (True):
        neigbours = get_neighbours(curr_layout)  # get all neigbours of currLayout
        # GETTING BEST NEIGHBOUR
        best_neighbour = neigbours[0]
        for neighbour in neigbours:
            if compute_cost(neighbour) < curr_cost:
                best_neighbour = neigbour
                break

        best_neighbour_cost = compute_cost(best_neighbour)

        if best_neighbour_cost >= curr_cost:
            break  # no improvement

        curr_layout = best_neighbour
        curr_cost = best_neighbour_cost

    return curr_layout, curr_cost


# Execution of HILL CLIMBING
optimal_layout, optimal_cost = hill_climbing()

# printing result
print("OPTIMIZED LAYOUT (DEPARTMENT : POSITION)")
for department in optimal_layout:
    print(f"{department} : {optimal_layout[department]}")
print(f"Minimum Transportation Cost : {optimal_cost}")
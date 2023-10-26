import random
import math
import numpy as np


def generate_state(N):
    return [random.randint(0, N - 1) for _ in range(N)]


# def calc_h_value(state):
#     count = 0
#     N = len(state)
#     for i in range(N - 1):
#         for j in range(i + 1, N):
#             if state[j] == state[i] or abs(state[j] - state[i]) == abs(j - i):
#                 count += 1
#     return count


def calc_h_value(state):
    # author: Sojal
    # n-queen board conflict checking in O(n) time
    # using O(n) space
    n = len(state)

    # col_frequency = [0] * n
    # main_diag_frequency = [0] * (2 * n)
    # secondary_diag_frequency = [0] * (2 * n)

    indices = np.arange(n)
    np_state = np.array(state)

    # for i in range(n):
    #     col_frequency[state[i]] += 1
    #     main_diag_frequency[state[i] + i] += 1
    #     secondary_diag_frequency[n - state[i] + i] += 1

    _, col_frequency = np.unique(np_state, return_counts=True)
    _, main_diag_frequency = np.unique(np_state + indices, return_counts=True)
    _, secondary_diag_frequency = np.unique(n - np_state + indices, return_counts=True)

    col_frequency = col_frequency[col_frequency > 1]
    main_diag_frequency = main_diag_frequency[main_diag_frequency > 1]
    secondary_diag_frequency = secondary_diag_frequency[secondary_diag_frequency > 1]

    conflicts = 0
    # formula: (N * (N - 1)) / 2
    # for i in range(2 * n - 2):
    #     if i < n:
    #         conflicts += (col_frequency[i] * (col_frequency[i] - 1)) / 2
    #     conflicts += (main_diag_frequency[i] * (main_diag_frequency[i] - 1)) / 2
    #     conflicts += (
    #         secondary_diag_frequency[i] * (secondary_diag_frequency[i] - 1)
    #     ) / 2
    conflicts += np.sum((col_frequency * (col_frequency - 1)) / 2)
    conflicts += np.sum((main_diag_frequency * (main_diag_frequency - 1)) / 2)
    conflicts += np.sum((secondary_diag_frequency * (secondary_diag_frequency - 1)) / 2)

    return int(conflicts)


# Steepest-Ascent Hill Climbing (Gradient Search)


def move_steepest_hill(state, h_to_beat):
    moves = {}
    N = len(state)
    for row in range(N):
        # best_move = state[row]
        for col in range(N):
            if col == state[row]:
                continue
            else:
                new_state = list(state)
                new_state[row] = col
                moves[(row, col)] = calc_h_value(new_state)
    best_moves = []
    # h_to_beat = calc_h_value(state)  # pass vào luôn?
    for k, v in moves.items():
        if v < h_to_beat:
            h_to_beat = v
    for k, v in moves.items():
        if v == h_to_beat:
            best_moves.append(k)
    if len(best_moves) > 0:
        index = random.randint(0, len(best_moves) - 1)
        element = best_moves[index]
        state[element[0]] = element[1]
    return state


def steepest_ascent(N):
    print("\t--- Steepest-Ascent Hill Climbing Algorithm ---")
    steps = 0
    state = generate_state(N)
    h = calc_h_value(state)
    while True:
        steps += 1
        print(f"+ Step: {steps} - Heuristic: {h}")
        if h == 0:
            print("#Solution found in step {}".format(steps))
            # print(f"Solution: {state}")
            return state
        state, new_h = move_steepest_hill(state, h)
        h = new_h


# Simulated Annealing
def move_annealing(state, h_to_beat, temp, N):
    rejected_moves = set()
    found_move = False
    while not found_move:
        new_row = random.randint(0, N - 1)
        new_col = random.randint(0, N - 1)
        if (new_row, new_col) in rejected_moves or state[new_row] == new_col:
            continue
        new_state = list(state)
        new_state[new_row] = new_col
        h_cost = calc_h_value(new_state)
        if h_cost < h_to_beat:
            found_move = True
        else:
            delta_e = h_to_beat - h_cost
            accept_probability = min(1, math.exp(delta_e / temp))
            found_move = random.random() <= accept_probability
            if not found_move:
                rejected_moves.add((new_row, new_col))
    return new_state, h_cost


def annealing(N):
    print("\t--- Simulated Annealing Algorithm ---")
    steps = 0
    temp = N**2
    annealing_rate = 0.95
    state = generate_state(N)
    h_cost = calc_h_value(state)
    while True:
        steps += 1
        print(f"+ Step: {steps} - Heuristic: {h_cost}")
        if h_cost == 0:
            print(f"Solution found in step {steps}")
            return state
        state, new_cost = move_annealing(state, h_cost, temp, N)
        temp = max(temp * annealing_rate, 0.01)
        h_cost = new_cost
        # if steps >= 50000:
        #     break

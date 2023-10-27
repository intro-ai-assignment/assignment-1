import random
import math
import numpy as np
from functools import reduce


# def generate_state(N):
#     return [random.randint(0, N - 1) for _ in range(N)]


def generate_permutation(N):
    initial_state = np.arange(N)
    np.random.shuffle(initial_state)
    return initial_state


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
    # state = np.array(state)
    # np_state = np.array(state)

    # for i in range(n):
    #     col_frequency[state[i]] += 1
    #     main_diag_frequency[state[i] + i] += 1
    #     secondary_diag_frequency[n - state[i] + i] += 1

    # _, col_frequency = np.unique(state, return_counts=True)
    _, main_diag_frequency = np.unique(state + indices, return_counts=True)
    _, secondary_diag_frequency = np.unique(n - state + indices, return_counts=True)

    # col_frequency = col_frequency[col_frequency > 1]
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
    # conflicts += np.sum((col_frequency * (col_frequency - 1)) / 2)
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
    # h_to_beat = calc_h_value(state)
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
    state = generate_permutation(N)
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
        # new_row = random.randint(0, N - 1)
        # new_col = random.randint(0, N - 1)
        # if (new_row, new_col) in rejected_moves or state[new_row] == new_col:
        #     continue
        # # new_state = list(state)
        # new_state = np.copy(state)
        # new_state[new_row] = new_col

        # swap random columns
        first_row = random.randint(0, N - 1)
        second_row = random.randint(0, N - 1)
        new_state = np.copy(state)
        t = new_state[first_row]
        new_state[first_row] = new_state[second_row]
        new_state[second_row] = t

        h_cost = calc_h_value(new_state)
        if h_cost < h_to_beat:
            found_move = True
        else:
            delta_e = h_to_beat - h_cost
            accept_probability = min(1, math.exp(delta_e / temp))
            found_move = random.random() <= accept_probability
            if not found_move:
                rejected_moves.add((first_row, second_row))
    return new_state, h_cost


def annealing(N):
    print("\t--- Simulated Annealing Algorithm ---")
    steps = 1
    temp = N**2
    annealing_rate = 0.95
    state = generate_permutation(N)
    h_cost = calc_h_value(state)
    while True:
        print(f"+ Step: {steps} - Heuristic: {h_cost}")
        if h_cost == 0:
            print(f"Solution found in step {steps}")
            return state
        state, h_cost = move_annealing(state, h_cost, temp, N)
        temp = max(temp * annealing_rate, 0.01)
        steps += 1
        # temp = max(2 / math.log10(1 + steps), 0.001)
        # h_cost = new_cost
        # if steps >= 50000:
        #     break


def compute_attacks(state, N):
    indices = np.arange(N)
    attack = set()

    t = state + indices
    k = N - state + indices
    main, main_diag_frequency = np.unique(t, return_counts=True)
    secondary, secondary_diag_frequency = np.unique(k, return_counts=True)

    main_unique = main[main_diag_frequency == 1]
    secondary_unique = secondary[secondary_diag_frequency == 1]
    for i in range(N):
        if t[i] not in main_unique or k[i] not in secondary_unique:
            attack.add(i)

    attack = list(attack)
    return attack, len(attack)


def compute_collisions(state, dn: dict, dp: dict, N):
    for i in range(N):
        dn[i + state[i]] = dn.get(i + state[i], 0) + 1
        dp[i - state[i]] = dp.get(i - state[i], 0) + 1

    for i in range(0, 2 * N - 1):
        if i not in dn:
            dn[i] = 0
    for i in range(1 - N, N):
        if i not in dp:
            dp[i] = 0

    dn_collisions = reduce(
        lambda acc, d: acc + (d - 1) if d != 0 else acc, dn.values(), 0
    )
    dp_collisions = reduce(
        lambda acc, d: acc + (d - 1) if d != 0 else acc, dp.values(), 0
    )
    return dn_collisions + dp_collisions


def test_swap(i, j, state, dn, dp):
    d = dict()
    diagonal_1 = i + state[i]
    diagonal_2 = i - state[i]
    diagonal_3 = j + state[j]
    diagonal_4 = j - state[j]
    diagonal_5 = i + state[j]
    diagonal_6 = i - state[j]
    diagonal_7 = j + state[i]
    diagonal_8 = j - state[i]
    # print(f"i {i}")
    # print(f"j {j}")

    keys = [
        (diagonal_1, True),
        (diagonal_2, False),
        (diagonal_3, True),
        (diagonal_4, False),
        (diagonal_5, True),
        (diagonal_6, False),
        (diagonal_7, True),
        (diagonal_8, False),
    ]

    # print(keys)

    for i in range(8):
        is_dn = i % 2 == 0
        # d[keys[i]] = d.get(keys[i], 0)
        if is_dn:
            d[keys[i]] = dn[keys[i][0]]
        else:
            d[keys[i]] = dp[keys[i][0]]

    # d[(diagonal_1, True)] = d.get((diagonal_1, True), 0) + dn[diagonal_1]
    # d[(diagonal_2, False)] = d.get((diagonal_2, False), 0) + dp[diagonal_2]
    # d[(diagonal_3, True)] = d.get((diagonal_3, True), 0) + dn[diagonal_3]
    # d[(diagonal_4, False)] = d.get((diagonal_4, False), 0) + dp[diagonal_4]
    # d[(diagonal_5, True)] = d.get((diagonal_5, True), 0) + dn[diagonal_5]
    # d[(diagonal_6, False)] = d.get((diagonal_6, False), 0) + dp[diagonal_6]
    # d[(diagonal_7, True)] = d.get((diagonal_7, True), 0) + dn[diagonal_7]
    # d[(diagonal_8, False)] = d.get((diagonal_8, False), 0) + dp[diagonal_8]

    # get 8 diagonals
    # d = list()
    # diagonal_1 = i + state[i]
    # diagonal_2 = i - state[i]
    # diagonal_3 = j + state[j]
    # diagonal_4 = j - state[j]
    # diagonal_5 = i + state[j]
    # diagonal_6 = i - state[j]
    # diagonal_7 = j + state[i]
    # diagonal_8 = j - state[i]

    # d.append(dn[diagonal_1])
    # d.append(dp[diagonal_2])
    # d.append(dn[diagonal_3])
    # d.append(dp[diagonal_4])
    # d.append(dn[diagonal_5])
    # d.append(dp[diagonal_6])
    # d.append(dn[diagonal_7])
    # d.append(dp[diagonal_8])

    # print(d)
    # print(keys)
    # total collisions from 8 diagonals in the original position
    # h1 = 0
    # t = set()
    # for i in range(8):
    #     k = i % 2 == 0
    #     if (keys[i], k) not in t and d[i] > 1:
    #         h1 += d[i] - 1
    #         t.add((keys[i], k))

    h1 = 0
    for i in d.values():
        if i > 1:
            h1 += i - 1
    # move the queens
    for i in range(8):
        if i < 4:
            d[keys[i]] -= 1
        else:
            d[keys[i]] += 1
    # print(d)
    # total collisions from 8 diagonals in the new position
    # h2 = 0
    # t = set()
    # for i in range(8):
    #     k = i % 2 == 0
    #     if (keys[i], k) not in t and d[i] > 1:
    #         h2 += d[i] - 1
    #         t.add((keys[i], k))

    h2 = 0
    for i in d.values():
        if i > 1:
            h2 += i - 1

    # if h2 < h1 then swap_ok = True
    reduction = h1 - h2
    # print(f"h1: {h1}")
    # print(f"h2: {h2}")
    # print(state)
    # print(dn)
    # print(dp)
    return reduction > 0, reduction


def perform_swap(state, i, j, collisions, reduction, dn, dp):
    # update dn, dp
    dn[i + state[i]] -= 1
    dp[i - state[i]] -= 1
    dn[j + state[j]] -= 1
    dp[j - state[j]] -= 1
    dn[i + state[j]] += 1
    dp[i - state[j]] += 1
    dn[j + state[i]] += 1
    dp[j - state[i]] += 1

    # update state
    t = state[i]
    state[i] = state[j]
    state[j] = t

    # update collisions
    collisions -= reduction

    return collisions


def print_chessboard(state):
    print()
    N = len(state)
    for i in range(N):
        print(end=" ")
        for k in range(N * 4 + 1):
            print("-", end="")
        print()
        for j in range(N):
            print(" | ", end="")
            if state[i] == j:
                print("Q", end="")
            else:
                print(" ", end="")
        print(" |")
    print(end=" ")
    for k in range(N * 4 + 1):
        print("-", end="")
    print()
    print()


def fast_search(N):
    print("\t--- Fast Algorithm ---")
    C1 = 0.45
    C2 = 32

    # initialization
    while True:
        state = generate_permutation(N)
        # h_cost = calc_h_value(state)
        print_chessboard(state)
        dn = dict()
        dp = dict()
        collisions = compute_collisions(state, dn, dp, N)
        if collisions == 0:
            return state
        # print(collisions)
        # print(dn)
        # print(dp)
        limit = C1 * collisions
        attack, number_of_attacks = compute_attacks(state, N)
        # print(attack)
        # print(number_of_attacks)
        loop_counts = 0
        steps = 0

        # search
        while loop_counts <= C2 * N:
            for k in range(number_of_attacks):
                i = attack[k]
                j = random.randint(0, N - 1)
                swap_ok, reduction = test_swap(i, j, state, dn, dp)
                if swap_ok:
                    collisions = perform_swap(
                        state, i, j, collisions, reduction, dn, dp
                    )
                    # print_chessboard(state)
                    # print(f"i j: {i} {j}")
                    steps += 1
                    # print(f"Heuristic: {calc_h_value(state)}")
                    print(f"+ Step: {steps} - Collisions: {collisions}")
                    # print(dn)
                    # print(dp)
                    if collisions == 0:
                        print(f"Heuristic: {calc_h_value(state)}")
                        print(f"Solution found in step {steps}")
                        return state
                    if collisions < limit:
                        limit = C1 * collisions
                        attack, number_of_attacks = compute_attacks(state, N)
                        break
            loop_counts += number_of_attacks

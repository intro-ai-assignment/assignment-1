import random
import math
import numpy as np
from functools import reduce


def generate_permutation(N):
    initial_state = np.arange(N)
    np.random.shuffle(initial_state)
    return initial_state


# this modified function is no longer suitable for the implemented hill climbing
def calc_h_value(state):
    n = len(state)

    indices = np.arange(n)
    _, main_diag_frequency = np.unique(state + indices, return_counts=True)
    _, secondary_diag_frequency = np.unique(n - state + indices, return_counts=True)

    main_diag_frequency = main_diag_frequency[main_diag_frequency > 1]
    secondary_diag_frequency = secondary_diag_frequency[secondary_diag_frequency > 1]

    conflicts = 0
    conflicts += np.sum((main_diag_frequency * (main_diag_frequency - 1)) / 2)
    conflicts += np.sum((secondary_diag_frequency * (secondary_diag_frequency - 1)) / 2)

    return int(conflicts)


# Steepest-Ascent Hill Climbing (Gradient Search)
def move_steepest_hill(state, h_to_beat):
    moves = {}
    N = len(state)
    for row in range(N):
        for col in range(N):
            if col == state[row]:
                continue
            else:
                new_state = list(state)
                new_state[row] = col
                moves[(row, col)] = calc_h_value(new_state)
    best_moves = []
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
            return state
        state, new_h = move_steepest_hill(state, h)
        h = new_h


# Simulated Annealing
def move_annealing(state, h_to_beat, temp, N):
    rejected_moves = set()
    found_move = False
    while not found_move:
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

    for i in range(8):
        is_dn = i % 2 == 0
        if is_dn:
            d[keys[i]] = dn[keys[i][0]]
        else:
            d[keys[i]] = dp[keys[i][0]]

    # total collisions from 8* diag in the original position
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

    # total collisions from 8* diag in the new position
    h2 = 0
    for i in d.values():
        if i > 1:
            h2 += i - 1

    # if h2 < h1 then swap_ok = True
    reduction = h1 - h2
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


def fast_search(N):
    print("\t--- Fast Algorithm ---")
    C1 = 0.45
    C2 = 32

    # initialization
    while True:
        state = generate_permutation(N)
        dn = dict()
        dp = dict()
        collisions = compute_collisions(state, dn, dp, N)
        if collisions == 0:
            return state
        limit = C1 * collisions
        attack, number_of_attacks = compute_attacks(state, N)
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
                    steps += 1
                    print(f"+ Step: {steps} - Collisions: {collisions}")
                    if collisions == 0:
                        print(f"Heuristic: {calc_h_value(state)}")
                        print(f"Solution found in step {steps}")
                        return state
                    if collisions < limit:
                        limit = C1 * collisions
                        attack, number_of_attacks = compute_attacks(state, N)
                        break
            loop_counts += number_of_attacks

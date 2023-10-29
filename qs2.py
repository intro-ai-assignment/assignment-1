import numpy as np
import random
from functools import reduce


def generate_permutation(N):
    initial_state = np.arange(N)
    np.random.shuffle(initial_state)
    return initial_state


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
    # count queens on diagonals
    for i in range(N):
        dn[i + state[i]] = dn.get(i + state[i], 0) + 1
        dp[i - state[i]] = dp.get(i - state[i], 0) + 1

    # diagonals that don't have any queens are set to 0
    for i in range(0, 2 * N - 1):
        if i not in dn:
            dn[i] = 0
    for i in range(1 - N, N):
        if i not in dp:
            dp[i] = 0

    # the number of collisions on any diagonal line
    # is one less than the number of queens on that line
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
    C2 = 1000

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
                        # double check
                        print(f"H value: {calc_h_value(state)}")
                        print(f"Solution found in step {steps}")
                        return state
                    if collisions < limit:
                        limit = C1 * collisions
                        attack, number_of_attacks = compute_attacks(state, N)
                        break
            loop_counts += number_of_attacks

import random
import math
import time

def generate_state(N):
    return [random.randint(0, N - 1) for value in range(N)]

def calc_h_value(state):
    count = 0
    N = len(state)
    for i in range(N - 1):
        for j in range(i + 1, N):
            if state[j] == state[i] or abs(state[j] - state[i]) == abs(j - i):
                count += 1
    return count

#Steepest-Ascent Hill Climbing (Gradient Search)

def move_steepest(state):
    moves = {}
    N = len(state)
    for row in range(N):
        best_move = state[row]
        for col in range(N):
            if col == state[row]:
                continue
            else:
                new_state = list(state)
                new_state[row] = col
                moves[(row, col)] = calc_h_value(new_state)
    best_moves = []
    h_to_beat = calc_h_value(state)
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
    while True:
        steps += 1
        h = calc_h_value(state)
        print(f"+ Step: {steps} - State: {state} - Heuristic: {h}")
        if h == 0:
            print("#Solution found in step {}".format(steps))
            print(f"Solution: {state}")
            return state
        state = move_steepest(state)

#Simulated Annealing
def move_annealing(state, h_to_beat, temp):
    N = len(state)
    new_state = list(state)
    found_move = False
    while not found_move:
        new_state = list(state)
        new_row = random.randint(0, N - 1)
        new_col = random.randint(0, N - 1)
        new_state[new_row] = new_col
        h_cost = calc_h_value(new_state)
        if h_cost < h_to_beat:
            found_move = True
        else:
            delta_e = h_to_beat - h_cost
            pro = math.exp(delta_e/temp)
            accept_probability = min(1, pro)
            found_move = random.random() <= accept_probability
    return new_state

def annealing(N):
    print("\t--- Simulated Annealing Algorithm ---")
    steps = 0
    temp = N ** 2
    annealing_rate = 0.95
    state = generate_state(N)
    while True:
        steps += 1
        h_cost = calc_h_value(state)
        print(f"+ Step: {steps} - Heuristic: {h_cost}")
        if h_cost == 0:
            print(f"Solution found in step {steps}")
            print(state)
            return state
        state = move_annealing(state, h_cost, temp)
        temp = max(temp*annealing_rate, 0.01)
        if steps >= 50000:
            break
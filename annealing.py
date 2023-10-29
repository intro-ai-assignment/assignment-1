import random
import math
import numpy as np


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


# Simulated Annealing
def move_annealing(state, h_value, temp):
    N = len(state)
    new_state = list(state)
    found_move = False
    while not found_move:
        new_state = list(state)
        new_row = random.randint(0, N - 1)
        new_col = random.randint(0, N - 1)
        new_state[new_row] = new_col
        h_cost = calc_h_value(new_state)
        if h_cost < h_value:
            found_move = True
        else:
            delta_e = h_value - h_cost
            probability = math.exp(delta_e / temp)
            accept_probability = min(1, probability)
            found_move = random.random() <= accept_probability
    return new_state


def main(N):
    print("\t--- Simulated Annealing Algorithm ---")
    steps = 0
    temp = N**2
    annealing_rate = 0.95
    state = generate_state(N)
    while True:
        steps += 1
        h_value = calc_h_value(state)
        print(f"+ Step: {steps} - Heuristic: {h_value}")
        if h_value == 0:
            print(f"Solution found in step {steps}")
            return state
        state = move_annealing(state, h_value, temp)
        temp = max(temp * annealing_rate, 0.01)
        if steps >= 50000:
            break

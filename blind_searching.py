import time
import random

def extend(state):
    N = len(state)
    new_states = []
    row = state.index(-1)
    for col in range(N):
        if col not in state:
            is_safe = True
            for i in range(row):
                if abs(row - i) == abs(col - state[i]):
                    is_safe = False
                    break
            if is_safe:
                new_state = list(state)
                new_state[row] = col
                new_states.append(new_state)
    return new_states
        
def BFS(N):
    init_state = [-1] * N
    frontier = [init_state]
    checked_states = []
    solutions = []
    steps = 0
    while True:
        steps += 1
        if len(frontier) == 0:
            print(f"Search complete after {steps - 1} steps!!!")
            for solution in solutions:
                print(solution)
            print(f"Total {len(solutions)} solutions")
            print("\t--- Breath First Search ---")
            return
        selected_state = frontier.pop(0)
        print(f"- Steps {steps}: {selected_state}")
        checked_states.append(selected_state)
        #check current state is solution?
        if selected_state.count(-1) == 0:
            solutions.append(selected_state)
            continue
        new_states = extend(selected_state)
        for state in new_states:
            if state not in frontier and state not in checked_states:
                frontier.append(state)

def DFS(N):
    init_state = [-1] * N
    frontier = [init_state]
    checked_states = []
    solutions = []
    steps = 0
    while True:
        steps += 1
        if len(frontier) == 0:
            print(f"Search complete after {steps - 1} steps!!!")
            for solution in solutions:
                print(solution)
            print(f"Total {len(solutions)} solutions")
            print("\t--- Depth First Search ---")
            return
        selected_state = frontier.pop()
        print(f"- Steps {steps}: {selected_state}")
        checked_states.append(selected_state)
        #check current state is solution?
        if selected_state.count(-1) == 0:
            solutions.append(selected_state)
            continue
        new_states = extend(selected_state)
        for state in new_states:
            if state not in frontier and state not in checked_states:
                frontier.append(state)

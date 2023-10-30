import time
import random
from queue import Queue


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

# Another solution:

# def is_valid_pos(arr: list[int], row: int, col: int):
#     for i in range(row):
#         if arr[i] == col or abs(i-row) == abs(arr[i]-col):
#             return False
#     return True

# def BFS(n: int):
#     solutions = []
#     queue = Queue()
#     init = [-1 for i in range(n)]
#     steps = 0
#     queue.put(init)

#     while not queue.empty():
#         steps += 1
#         pos = queue.get()
#         row = 0
#         while row < n and pos[row] != -1:
#             row += 1
#         if row == n:
#             solutions.append(pos)
#         for col in range(n):
#             if is_valid_pos(pos, row, col):
#                 new_pos = pos.copy()
#                 new_pos[row] = col
#                 queue.put(new_pos)
    
#     print(f"Search complete after {steps} steps!!!")
#     for solution in solutions:
#         print(solution)
#     print(f"Total {len(solutions)} solutions")
#     print("\t--- Breadth First Search ---")
#     return solutions

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
        # check current state is solution?
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
            break
        selected_state = frontier.pop()
        print(f"- Steps {steps}: {selected_state}")
        checked_states.append(selected_state)
        # check current state is solution?
        if selected_state.count(-1) == 0:
            solutions.append(selected_state)
            continue
        new_states = extend(selected_state)
        for state in new_states:
            if state not in frontier and state not in checked_states:
                frontier.append(state)
    return solutions

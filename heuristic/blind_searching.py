import time
import random

class State:
    def __init__(self, N, data = None):
        self.N = N
        if data is None:
            self.data = []
        else:
            self.data = data

    def is_the_solution(self):
        return len(self.data) == self.N

    def __eq__(self, other):
        return self.N == other.N and self.data == other.data

    def extend(self):
        if self.is_the_solution():
            return
        new_states = []
        row = len(self.data)
        for col in range(self.N):
            if col not in self.data:
                is_safe = True
                for i in range(row):
                    if abs(row - i) == abs(col - self.data[i]):
                        is_safe = False
                        break
                if is_safe:
                    new_data = list(self.data)
                    new_data.append(col)
                    new_state = State(self.N, new_data)
                    new_states.append(new_state)
        return new_states

    def __repr__(self):
        result = [str(value) for value in self.data]
        result = ", ".join(result)
        result = "[" + result + "]"
        return result
        

def BFS(N):
    print("\t--- Breath First Search ---")
    init_state = State(N)
    frontier = [init_state]
    checked_states = []
    solutions = []
    steps = 0
    while True:
        steps += 1
        if len(frontier) == 0:
            if(len(solutions) == 0):
                print("No solutions...")
            else:
                for index in range(len(solutions)):
                    print(f"- Solution {index + 1}: {solutions[index]}")
                print(f"Total solutions: {len(solutions)}")
            return
        else:
            selected_state = frontier.pop(0)
            print(f"+ Step {steps}: {selected_state} - {len(selected_state.data)} - {N}")
            checked_states.append(selected_state)
            if selected_state.is_the_solution():
                solutions.append(selected_state)
            else:
                new_states = selected_state.extend()
                for new_state in new_states:
                    if new_state not in frontier and new_state not in checked_states:
                        frontier.append(new_state)

def DFS(N):
    print("\t--- Depth First Search ---")
    init_state = State(N)
    frontier = [init_state]
    checked_states = []
    solutions = []
    steps = 0
    while True:
        steps += 1
        if len(frontier) == 0:
            if(len(solutions) == 0):
                print("No solutions...")
            else:
                for index in range(len(solutions)):
                    print(f"- Solution {index + 1}: {solutions[index]}")
                print(f"Total solutions: {len(solutions)}")
            return
        else:
            selected_state = frontier.pop()
            print(f"+ Step {steps}: {selected_state} - {len(selected_state.data)} - {N}")
            checked_states.append(selected_state)
            if selected_state.is_the_solution():
                solutions.append(selected_state)
            else:
                new_states = selected_state.extend()
                for new_state in new_states:
                    if new_state not in frontier and new_state not in checked_states:
                        frontier.append(new_state)

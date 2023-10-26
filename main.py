import genetic
import hill_climbing
import blind_searching

import time

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

if __name__ == "__main__":
    print("\n\t*** N-Queens Problem ***")
    N = int(input("Enter N = "))
    if N < 4:
        print("No solution...")
        exit()
    while True:
        print("#Select Algorithm Number to solve problem:")
        print(" 1. Breadth First Search (BFS)")
        print(" 2. Depth First Search (DFS)")
        print(" 3. Genetic Algorithm")
        print(" 4. Steepest-Ascent Hill-Climbing")
        print(" 5. Simulated Annealing")
        select = int(input("Enter number: "))
        if select < 1 or select > 5:
            print("Please enter valid value...")
        else:
            t1 = time.time()
            result = []
            if select == 1:
                blind_searching.BFS(N)
            elif select == 2:
                blind_searching.DFS(N)
            elif select == 3:
                result = genetic.genetic_alg(N)
            elif select == 4:
                result = hill_climbing.steepest_ascent(N)
            else:
                result = hill_climbing.annealing(N)
            t2 = time.time()
            if len(result) > 0 and len(result) < 30:
                print_chessboard(result)
            print("Time cost:", t2 - t1, "seconds")
            break


import qs2
import annealing
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
        print(" 3. Simulated Annealing")
        print(" 4. QS2")
        select = int(input("Enter number: "))
        if select < 1 or select > 4:
            print("Please enter valid value...")
        else:
            t1 = time.time()
            result = []
            if select == 1:
                result = blind_searching.BFS(N)
            elif select == 2:
                result = blind_searching.DFS(N)
            elif select == 3:
                result = annealing.main(N)
            elif select == 4:
                result = qs2.fast_search(N)
            t2 = time.time()
            # if len(result) > 0:
            #     for state in result:
            #         print_chessboard(state)
            print(result)
            print("Time cost:", t2 - t1, "seconds")
            break

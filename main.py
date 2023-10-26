import blind_searching
import annealing
import time

if __name__ == "__main__":
    print("\n\t*** N-Queens Problem ***")
    print()
    N = int(input("Enter N = "))
    if N < 4:
        print("No solution...")
        exit()
    while True:
        print("#Select Algorithm Number to solve problem:")
        print(" 1. Breadth First Search (BFS)")
        print(" 2. Depth First Search (DFS)")
        print(" 3. Simulated Annealing (Heuristic)")
        select = input("Enter number: ")
        if len(select) > 1:
            print("\n\t***Please enter valid value...\n")
        else:
            select = int(select)
            if select < 1 or select > 3:
                print("\n\t***Please enter valid value...\n")
            else:
                t1 = time.time()
                result = []
                if select == 1:
                    blind_searching.BFS(N)
                elif select == 2:
                    blind_searching.DFS(N)
                else:
                    annealing.main(N)
                t2 = time.time()
                print("Time cost:", t2 - t1, "seconds")
                break
        


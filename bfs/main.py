import sys

import bfs

if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(f"Solving N-queens problem with n = {sys.argv[1]} ...")
        print(bfs.solve(int(sys.argv[1])))
    else:
        raise SyntaxError(f"Insufficient arguments.\n {sys.argv}")

"""
This is implementation of Breadth-First Search of N-queens problem.

Common command lines:
   - 4-queen: python bfs.py 4
   - 6-queen: python bfs.py 6

NOTE:
   - 'Q': Queen
   - '.': Empty postion
"""
from queue import Queue

def is_valid_pos(arr: list[int], row: int, col: int):
   for i in range(row):
      if arr[i] == col or abs(i-row) == abs(arr[i]-col):
         return False
   return True
def solve(n: int):
   res = []
   queue = Queue()
   init = [-1 for i in range(n)]
   queue.put(init)

   while not queue.empty():
      pos = queue.get()

      row = 0
      while row < n and pos[row] != -1:
         row += 1
      if row == n:
         solution = [['.' for i in range(n)] for i in range(n)]
         for i in range(n):
            solution[i][pos[i]] = 'Q'
         res.append(solution)
         continue
      for col in range(n):
         if is_valid_pos(pos, row, col):
            new_pos = pos.copy()
            new_pos[row] = col
            queue.put(new_pos)
   return res

from typing import List
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(grid, i, j,visited):
            if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] == "0":
                return

            grid[i][j] = "0"
            visited.add((i,j))

            dfs(grid, i - 1, j,visited)
            dfs(grid, i + 1, j,visited)
            dfs(grid, i, j - 1,visited)
            dfs(grid, i, j + 1,visited)
            return

        count = 0
        visited =set()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1" and (i,j) not in visited:
                    dfs(grid,i, j,visited)
                    count += 1
        return count

a =Solution()
print(a.numIslands([["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]))
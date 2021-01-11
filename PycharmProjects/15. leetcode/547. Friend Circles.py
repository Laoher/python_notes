import collections
from typing import List
class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        def dfs(i,visited):
            if i in visited:
                return
            visited.add(i)
            for f in dict[i]:
                dfs(f, visited)
            return

        dict = collections.defaultdict(list)
        for i in range(len(M)):
            for j in range(len(M)):
                if M[i][j] == 1:
                    dict[i].append(j)
                    dict[j].append(i)

        visited = set()
        count = 0
        for i in range(len(M)):
            if i not in visited:
                dfs(i,visited)
                count+=1
        return count




a =Solution()
print(a.findCircleNum([[1,1,0],[1,1,0],[0,0,1]]))
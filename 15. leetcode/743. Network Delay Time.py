import collections
from typing import List
class Solution:
    def networkDelayTime(self, times: List[List[int]], N: int, K: int) -> int:
        dict2 = collections.defaultdict(list)
        shortestFromK={}
        for u, v, w in times:
            dict2[u].append((v, w))

        def dfs(dict2, start, current, visited):
            if dict2[current] is None:
                visited.add(current)
                return


            for nei, time in dict2[current]:
                if current==start:
                    shortestFromK[current] =0
                if nei not in shortestFromK.keys() or shortestFromK[current] + time < shortestFromK[nei]:
                    shortestFromK[nei] = shortestFromK[current] + time

            visited.add(current)
            ls = [i for i in dict2[current] if i[0] not in visited]

            if ls != []:
                ls = sorted(ls, key=lambda x: x[1])
            for nei, time in ls:
                dfs(dict2, K, nei, visited)
            return

        visited = set()
        dfs(dict2, K, K, visited)
        if len(shortestFromK) < N:
            return -1
        else:
            return max(shortestFromK.values())
S =Solution()
print(S.networkDelayTime([[1,2,1],[2,1,3]],2,2))
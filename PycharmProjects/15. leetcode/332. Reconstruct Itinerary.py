import collections
import heapq
from typing import List
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        g = collections.defaultdict(list)
        for ticket in tickets:
            heapq.heappush(g[ticket[0]], ticket[1])

        def dfs(g, start, res):
            while len(g[start]) != 0:
                dfs(g, heapq.heappop(g[start]), res)
            res.append(start)

        res = []

        dfs(g, "JFK", res)
        print(res)
        return res[::-1]

a = Solution()
print(a.findItinerary(
[["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]]))
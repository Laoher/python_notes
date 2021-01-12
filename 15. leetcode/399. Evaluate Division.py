import collections
import heapq
from typing import List
import itertools


class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        ls = []
        graph = collections.defaultdict(set)


        for (a,b),v in zip(*equations,*values):
            print(a,b,v)
            graph[a].add((b,v))
            graph[b].add((a,1.0/v))

        def dfs(a,b,r,seen):
            if a==b:
                return r
            seen.add(a)
            for nei,value in graph[a]:
                if nei not in seen:
                    res =  dfs(nei, b,r*value,seen)
                    if res!=-1:
                        return res
            return -1

        for a,b in queries:
            if not(a in graph.keys() and b in graph.keys()):
                ls.append(-1)
            else:
                ls.append(dfs(a,b,1,set()))
        return ls








equations = [ ["a", "b"], ["b", "c"] ],
values = [2.0, 3.0],
queries = [ ["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"] ]

a = Solution()
print(a.calcEquation(equations,
values,
queries))
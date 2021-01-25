import collections
import heapq
from typing import List
import itertools


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if strs == []:
            return ""
        common = strs[0]
        index = 1
        stop = len(common)
        while index < len(strs) and len(common) > 0:

            for i in range(len(common)):
                print(common[i], strs[index][i])
                if common[i] != strs[index][i]:
                    stop = i
                    break
            common = common[:stop]
            index += 1
        return common


a = Solution()
print(a.longestCommonPrefix(["c","c"]))
import collections
import heapq
from typing import List
import itertools


class Solution:
    def romanToInt(self, s: str) -> int:
        dict = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        sum = 0
        for i in range(len(s)):
            letter = s[i]
            sign =1
            if i <len(s)-1:
                next_letter = s[i+1]
                if dict[letter]<dict[next_letter]:
                    sign = -1
            sum+=sign*dict[letter]
        return sum








a = Solution()
print(a.romanToInt("MCMXCIV"))
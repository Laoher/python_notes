import collections
import heapq
from typing import List
import itertools


class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def position(index):
            current_counting = 0

            pt1 = 0
            pt2 = 0
            while not current_counting == index:
                if pt1 == len(nums1):
                    median = nums2[pt2]
                    pt2 += 1
                    current_counting += 1
                    continue

                if pt2 == len(nums2):
                    median = nums1[pt1]
                    pt1 += 1
                    current_counting += 1
                    continue
                if pt1 != len(nums1) and pt2 != len(nums2):
                    if nums1[pt1] < nums2[pt2]:
                        median = nums1[pt1]
                        pt1 += 1
                    else:
                        median = nums2[pt2]
                        pt2 += 1
                    current_counting += 1
                    continue

            return median

        total = len(nums1) + len(nums2)
        middle = int(total / 2)
        if total % 2 == 0:
            return (position(middle)+position(middle+1)) / 2
        if total % 2 == 1:
            return position(middle+1)





a = Solution()
print(a.findMedianSortedArrays(nums1 = [2], nums2 = []))

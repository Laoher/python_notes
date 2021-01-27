# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1: return l2
        if not l2: return l1

        (A,B) = (l1,l2)if l1.val<l2.val else(l2,l1)

        ret = A

        while A.next and B:
            (A.next, B) = (A.next, B) if A.next.val < B.val else(B, A.next)
            A=A.next
        A.next =B
        return ret

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head: return False
        if not head.next: return False
        s = set()
        while head.next:
            if head not in s:
                s.add(head)
            if head.next in s:
                return True
            head = head.next
        return False
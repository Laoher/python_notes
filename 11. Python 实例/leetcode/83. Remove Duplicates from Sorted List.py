# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head: return None
        s = set()
        i = head

        while i:
            if i.val not in s:
                s.add(i.val)
            temp = i.next
            while temp and temp.val in s:
                temp = temp.next

            i.next = temp
            i = i.next

        return head

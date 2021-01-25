from typing import List
import itertools

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        def iot(root, seen):
            if root is None:
                return seen

            seen = iot(root.left, seen)
            if root not in seen:
                seen.append(root.val)
            seen = iot(root.right, seen)
            return seen

        return iot(root,[])

    def inorderTraversal_iter(self, root: TreeNode) -> List[int]:
        tree_stack = []
        tree_order = []

        while tree_stack or root:
            while root:
                tree_stack.append(root)
                root = root.left

            node = tree_stack.pop()
            tree_order.append(node.val)
            root = node.right

        return tree_order


a = Solution()
print(a.inorderTraversal())
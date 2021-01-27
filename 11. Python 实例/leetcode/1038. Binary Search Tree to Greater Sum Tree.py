
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def bstToGst(self, root: TreeNode) -> TreeNode:
        def sum_of_tree(root, ls):
            if root == None:
                return
            index = ls.index(root.val)
            sum_tree = sum(ls[index:])
            root.val = sum_tree
            sum_of_tree(root.left, ls)
            sum_of_tree(root.right, ls)

        def find_leaves(root, ls):
            if root == None:
                return
            ls.append(root.val)
            find_leaves(root.left, ls)
            find_leaves(root.right, ls)
            return

        ls = []
        find_leaves(root, ls)
        ls = sorted(ls)
        sum_of_tree(root, ls)

        return root

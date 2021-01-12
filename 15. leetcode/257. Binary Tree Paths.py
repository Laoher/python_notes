from typing import List
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        def add_path(root,topvalue,ls,path):

            previous_path = path
            if path == "":
                path  += str(root.val)
            else:
                path += "->"+str(root.val)

            if root.left==None and root.right==None:
                ls.append(path)
                add_path(root.left, topvalue, ls, previous_path)
                add_path(root.right, topvalue, ls, previous_path)
                return
            else:
                add_path(root.left, topvalue, ls, path)
                add_path(root.right, topvalue, ls, path)


            return




        ls =[]
        path =""
        add_path(root,root.val, ls,path)

        return ls
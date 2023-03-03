from toy_module import *


class Tree:
    def __init__(self, alpha, left=None, right=None, leaf=None):
        self.left = left
        self.right = right
        if left is None and right is None:
            self.data = [leaf]
            self.height = 0
            self.d = alpha
            self.pi = 1
        else:
            self.data = self.left.data + self.right.data
            self.height = max(left.height, right.height) + 1
            self.d = alpha * np.math.factorial(len(self.data)-1) + self.left.d * self.right.d
            self.pi = alpha * np.math.factorial(len(self.data)-1) / self.d


def data_given_h1(tree, param):
    data = tree.data
    cell = CellHolder(param)
    multiplied_lklhd = np.zeros(1)
    cnt = 0
    for d in data:
        minE, pst, lklhd = prob_calculator(cell, d, param)
        if cnt == 0:
            multiplied_lklhd = np.array(lklhd)
        else: multiplied_lklhd *= np.array(lklhd)
        cnt += 1
    c = cell.list[0]
    multiplied_lklhd *= c.df * c.volume  # p(theta | beta) in Heller paper.

    return sum(multiplied_lklhd)


def data_given_tree(tree, param): # p(D_k|T_k) in Heller
    if tree.height == 0: return tree.pi * data_given_h1(tree, param)
    else:
        return tree.pi * data_given_h1(tree, param) + (tree.left.d * tree.right.d / tree.d) \
               * data_given_tree(tree.left, param) * data_given_tree(tree.right, param)



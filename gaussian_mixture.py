import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time


class Tree:
    def __init__(self, alpha, left=None, right=None, leaf=None, rk=None):
        """

        :param alpha: Dirichlet hyperparameter
        :param left: Tree
        :param right: Tree
        :param leaf: numpy array
        :param rk:
        """
        self.left = left
        self.right = right
        if left is None and right is None:  # if initialized
            self.data = [leaf]
            self.height = 0
            self.d = alpha
            self.pi = 1
            self.rk = rk  # 기록용
        else:  # if merged
            self.data = self.left.data + self.right.data
            self.height = max(left.height, right.height) + 1
            self.d = alpha * np.math.factorial(len(self.data)-1) + self.left.d * self.right.d
            self.pi = alpha * np.math.factorial(len(self.data)-1) / self.d
            self.rk = rk


def marginal_lklhd(_mu, _sig, _obs, _hyperparam):
    mu0 = _hyperparam[0]
    sig0 = _hyperparam[1]
    if sig0 == 0:
        raise Exception('sig0 must be greater than zero')
    alpha = _hyperparam[2]
    beta = _hyperparam[3]
    k = len(_obs)
    coeff = (1/np.sqrt(2*np.pi*_sig**2))**k / (np.sqrt(2*np.pi*sig0**2)) * beta**alpha/np.math.gamma(alpha) \
            / _sig ** (2*alpha+2)
    exp = -sum((_obs-_mu)**2)/(2*_sig**2) - beta/_sig**2 - (_mu-mu0)**2/(2*sig0**2)
    return coeff * np.exp(exp)


def data_given_h1(_tree, _hyperparam):
    _obs = np.array(_tree.data)
    f = lambda mu, sig: marginal_lklhd(mu, sig, _obs, _hyperparam)
    return integrate.dblquad(f, 0, np.inf, 0, np.inf)[0]


def data_given_tree(_tree, _param): # p(D_k|T_k) in Heller
    if _tree.height == 0:
        return _tree.pi * data_given_h1(_tree, _param)
    else:
        return _tree.pi * data_given_h1(_tree, _param) + (_tree.left.d * _tree.right.d / _tree.d) \
               * data_given_tree(_tree.left, _param) * data_given_tree(_tree.right, _param)


param = [4, 10, 1, 1]  # [mu0, sig0, alpha, beta]
alpha = 2
cluster1 = list()
cluster2 = list()
for i in range(10):
    cluster1.append(np.random.normal(2, 1))
    cluster2.append(np.random.normal(10, 1))

dataset = np.array(cluster1+cluster2)
trees = list()
for i in range(len(dataset)):
    trees.append(Tree(alpha,leaf=dataset[i]))

c = len(trees)
start = time.time()
while c > 1:
    cnt = 1
    rmax = 0
    tree_to_merge = []
    idx = []
    for i in range(len(trees)):
        for j in range(len(trees)):
            if i >= j:continue
            else:
                t_merge = Tree(alpha, trees[i], trees[j])
                rk = t_merge.pi * data_given_h1(t_merge, param) / data_given_tree(t_merge, param)
                rmax = max(rk, rmax)
                if rk == rmax:
                    tree_to_merge = [trees[i], trees[j]]
                    idx = [i, j]
                print('loop=',c,'progress=',cnt,'/',int(c*(c-1)/2))
                print('rk:', rk, 'for [i,j]=',[i,j])
                print('--------------------------')
            cnt += 1
    if rmax <= 0.5:
        break
    print('rmax is', rmax, 'for pair [i,j]=',idx)
    print('=============================')
    merge = Tree(alpha,left=tree_to_merge[0], right=tree_to_merge[1],rk=rmax)
    trees.remove(tree_to_merge[0])
    trees.remove(tree_to_merge[1])
    trees.append(merge)
    c -= 1

end = time.time()
print('That took ', end-start, 'sec')

clusters = list()
for tree in trees:
    clusters.append(tree.data)
print('number of clusters:', len(clusters))

plt.subplot(2, 1, 1)
plt.plot(dataset, dataset, '.')

plt.subplot(2, 1, 2)
for c in clusters:
    plt.plot(c, c, '.')
plt.show()

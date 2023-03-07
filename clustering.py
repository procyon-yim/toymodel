from clustering_module import *
from toy_module import *
import time
import matplotlib.pyplot as plt


alpha = 5
param = read_params('toy_params.txt')
w = allocate_stars(param)
n_stars = param['number of samples']

trees = list()
for i in range(n_stars):
    trees.append(Tree(alpha,leaf=w[i]))


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
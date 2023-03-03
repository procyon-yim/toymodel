from clustering_module import *
from toy_module import *
import time

param = read_params('toy_params.txt')
tree_list = list()
w = allocate_stars(param)
n_stars = param['number of samples']

trees = list()
for i in range(n_stars):
    trees.append(Tree(2,leaf=w[i]))


c = len(trees)
start = time.time()
while c > 1:
    cnt = 1
    rmax = 0
    tree_to_merge = []
    for i in range(len(trees)):
        for j in range(len(trees)):
            if i >= j:continue
            else:
                t_merge = Tree(2, trees[i], trees[j])
                rk = t_merge.pi * data_given_h1(t_merge, param) / data_given_tree(t_merge, param)
                rmax = max(rk, rmax)
                if rk == rmax:
                    tree_to_merge = [trees[i], trees[j]]
                print('c=',c,'cnt=',cnt)
                cnt += 1

    merge = Tree(2,tree_to_merge[0], tree_to_merge[1])
    trees.remove(tree_to_merge[0])
    trees.remove(tree_to_merge[1])
    trees.append(merge)
    c -= 1
end = time.time()
print('That took ', end-start, 'sec')

root = trees[0]  # 전체 트리

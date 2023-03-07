from toy_module import *
from scipy import integrate


class Tree:
    def __init__(self, alpha, left=None, right=None, leaf=None, rk=None):
        self.left = left
        self.right = right
        if left is None and right is None:
            self.data = [leaf]
            self.height = 0
            self.d = alpha
            self.pi = 1
            self.rk = rk
        else:
            self.data = self.left.data + self.right.data
            self.height = max(left.height, right.height) + 1
            self.d = alpha * np.math.factorial(len(self.data)-1) + self.left.d * self.right.d
            self.pi = alpha * np.math.factorial(len(self.data)-1) / self.d
            self.rk = rk


def p_given_energy(m, k, vel, energy):
    if 1/2*m*vel**2 >= energy:
        return 0
    else:
        w0 = np.sqrt(k/m)
        return 1/(2*np.pi*w0)*1/np.sqrt(2*energy/k-vel**2/w0**2)


def marginal_lklhd(vel, energy, param):
    global prob
    m = param['mass']
    k = param['spring constant']
    emax = param['maximum energy']
    pe = 1/emax
    for i in range(len(vel)):
        if i == 0:
            prob = p_given_energy(m,k,vel[i],energy)
        else:
            prob *= p_given_energy(m,k,vel[i],energy)
    prob *= pe
    return prob


def data_given_h1(tree, param):
    elim = param['maximum energy']
    vel = np.array(tree.data)
    f = lambda energy : marginal_lklhd(vel, energy, param)
    return integrate.quad(f, 0, elim)[0]


def data_given_tree(treek, param): # p(D_k|T_k) in Heller
    if treek.height == 0:
        return treek.pi * data_given_h1(treek, param)
    else:
        return treek.pi * data_given_h1(treek, param) + (treek.left.d * treek.right.d / treek.d) \
               * data_given_tree(treek.left, param) * data_given_tree(treek.right, param)



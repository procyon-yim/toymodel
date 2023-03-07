import numpy as np

from toy_module import *
from clustering_module import *
import matplotlib.pyplot as plt
import time
from clustering_module import *


def f(_v, _param):
    _m = _param['mass']
    _k = _param['spring constant']
    _E = _param['energy of each rings'][0]
    _w0 = np.sqrt(_k/_m)
    return 1/(2*np.pi*_w0)*1/np.sqrt(2*_E/_k-_v**2/_w0**2)


param = read_params("toy_params.txt")
print(param['energy of each rings'])
w = allocate_stars(param)
print(w)
u = list()
for v in w:
    u.append(v[1])
vmax = max(u)-0.01

v_range = np.linspace(-vmax, vmax, 1000)
prob = f(v_range, param)

plt.hist(u, bins=20, density=True)
plt.plot(v_range, prob)
plt.show()
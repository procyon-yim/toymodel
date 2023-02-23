from toy_module import *
import matplotlib.pyplot as plt
param = read_params('toy_params.txt')
w = allocate_stars(param)
pot = Potential(param)
period = 2*np.pi*np.sqrt(0.5)
for coord in w:
    x = trajectory(pot, coord)[0]
    p = trajectory(pot, coord)[1]
    wobs = observe(pot, np.random.random_sample()*period, coord)
    xobs = wobs[0]
    pobs = wobs[1]
    plt.plot(x, p)
    plt.plot(xobs, pobs, '*')

plt.show()
from toy_module import *
import matplotlib.pyplot as plt


param = read_params('toy_params.txt')
N_stars = param['number of stars']
N_sample = param['number of samples']
x_max = np.sqrt(2*param['maximum energy']/param['spring constant'])
p_max = np.sqrt(2*param['maximum energy']*param['mass'])
period = 2*np.pi*np.sqrt(param['mass']/param['spring constant'])
pot = Potential(param)
nE = 100
nX = 1000

w_init = allocate_stars(param)

star_idx = np.random.choice(np.arange(N_stars), N_sample, replace=False)
stars = list()
for i in range(len(star_idx)):
    stars.append({'idx':star_idx[i],'w_init': w_init[star_idx[i]], 'w_obs': None, 'minE': [], 'posterior': []})

for i in range(len(star_idx)):
    t = np.random.random_sample()*period
    w = observe(pot, t, stars[i]['w_init'])
    stars[i]['w_obs'] = w
    v = w[1] / param['mass']
    cell = CellHolder()
    cell.construct(nE, param, 'uniform')
    prob_calculator(cell, v, param, nX)
    for c in cell.list:
        stars[i]['posterior'].append(c.posterior)
        stars[i]['minE'].append(c.minE)

cells = CellHolder()
cells.construct(nE, param, 'uniform')
minE = list()
for cell in cells.list:
    minE.append(cell.minE)

post_sum = np.zeros(len(stars[0]['minE']))
for star in stars:
    post_sum += np.array(star['posterior'])

plt.plot(minE, post_sum)
plt.show()
# for star in stars:
#     plt.subplot(2,1,1)
#     plt.bar(star['minE'], star['posterior'], align='edge', width=0.3)
#     k = param['k']
#     m = param['m']
#     x = star['w_obs'][0]
#     p = star['w_obs'][1]
#     energy = p**2/(2*m) + 1/2*k*x**2
#     plt.bar(energy, max(star['posterior']))
# plt.xlabel('Energy range (J)')
# plt.ylabel('Posterior', fontsize=15)
# plt.subplot(2, 1, 2)
# for w in orbits:
#     x = trajectory(pot, w)[0]
#     p = trajectory(pot, w)[1]
#     plt.plot(x, p)
#
# for star in stars:
#     plt.plot(star['w_obs'][0], star['w_obs'][1], '*')
# plt.xlabel('x')
# plt.ylabel('p')
# plt.show()
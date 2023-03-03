from toy_module import *
import matplotlib.pyplot as plt


param = read_params('toy_params.txt')
N_stars = param['number of stars']
N_sample = param['number of samples']
period = 2*np.pi*np.sqrt(param['mass']/param['spring constant'])

pscoord = allocate_stars(param)

star_idx = np.random.choice(np.arange(N_stars), N_sample, replace=False)
stars = list()
for i in range(len(star_idx)):
    stars.append({'idx':star_idx[i],'pscoord': pscoord[star_idx[i]], 'minE': [], 'posterior': []})

for i in range(len(star_idx)):
    w = stars[i]['pscoord']
    cell = CellHolder(param)
    pst = prob_calculator(cell, w, param)
    for c in cell.list:
        stars[i]['posterior'].append(c.posterior)
        stars[i]['minE'].append(c.minE)


post_overlap = np.ones(len(stars[0]['minE']))
for star in stars:
    post_overlap *= np.array(star['posterior'])
integ = 0
for i in post_overlap:
    integ += i * (stars[0]['minE'][1]-stars[0]['minE'][0])
plt.subplot(3, 1, 1)
plt.plot(stars[0]['minE'], post_overlap)
plt.title('integrated region: %f'%integ)

for star in stars:
    plt.subplot(3,1,2)
    plt.bar(star['minE'], star['posterior'], align='edge', width=0.3)
    k = param['spring constant']
    m = param['mass']
    x = star['pscoord'][0]
    p = star['pscoord'][1]
    # energy = p**2/(2*m) + 1/2*k*x**2
    # plt.bar(energy, max(star['posterior']))
plt.xlabel('Energy range (J)')
plt.ylabel('Posterior', fontsize=15)
plt.subplot(3, 1, 3)
for ring in param['energy of each rings']:
    x_coord = np.sqrt(2*ring/param['spring constant'])
    w = np.array([x_coord, 0])
    x = trajectory(param, w)[0]
    p = trajectory(param, w)[1]
    plt.plot(x, p)

for star in stars:
    plt.plot(star['pscoord'][0], star['pscoord'][1], '.')
plt.xlabel('x')
plt.ylabel('p')
plt.show()
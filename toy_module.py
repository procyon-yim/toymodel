import numpy as np


class Cell:

    def __init__(self, _param, _idx):
        _m = _param['mass']
        _k = _param['spring constant']
        _E_limit = _param['maximum energy']
        _N_cell = _param['number of energy intervals']
        _N_clusters = _param['number of rings']
        _E_clusters = _param['energy of each rings']
        _dE = _E_limit/_N_cell
        if _idx > _N_cell - 1:
            raise Exception('index out of range')
        self.index = _idx
        self.maxE = (_idx+1)*_dE
        self.minE = _idx*_dE
        self.volume = 2*np.pi*(self.maxE-self.minE)*np.sqrt(_m/_k)

        if _param['distribution function option'] == 0:  # DF weighted on cells with clusters
            weight = 10
            for _i in range(_N_clusters):
                if self.minE <= _E_clusters[_i] < self.maxE:
                    self.df = weight/(self.volume*((weight-1)*_N_clusters+_N_cell))
                    break
                else:
                    self.df = 1/(self.volume*((weight-1)*_N_clusters+_N_cell))

        elif _param['distribution function option'] == 1:  # uniform DF
            self.df = 1/(2*np.pi*_E_limit*np.sqrt(_m/_k))

        self.posterior = 0
        self.likelihood = 0
        self.temp = 0


class CellHolder:

    def __init__(self, _param):
        self.list = []
        _N_cell = _param['number of energy intervals']
        _m = _param['mass']
        _k = _param['spring constant']
        _E_limit = _param['maximum energy']
        for idx in range(_N_cell):
            _cell = Cell(_param, idx)
            self.list.append(_cell)


def read_params(_param_file):
    with open(_param_file) as _f:
        _param = dict()
        for _l in _f.readlines():
            _key = _l.split('=')[0]
            _value = _l.split('=')[1]
            if '\n' in _value:
                if ',' not in _value:
                    if '.' not in _value:
                        _value = int(_value[:_value.index('\n')])
                    else:
                        _value = float(_value[:_value.index('\n')])
                else:
                    if '.' not in _value:
                        _value = _value[:_value.index('\n')]
                        _value_lst = []
                        for _i in _value.split(','):
                            _value_lst.append(int(_i))
                        _value = _value_lst
                    else:
                        _value = _value[:_value.index('\n')]
                        _value_lst = []
                        for _i in _value.split(','):
                            _value_lst.append(float(_i))
                        _value = _value_lst
            else:
                if ',' not in _value:
                    if '.' not in _value:
                        _value = int(_value)
                    else:
                        _value = float(_value)
                else:
                    _value_lst = []
                    if '.' not in _value:
                        for _i in _value.split(','):
                            _value_lst.append(int(_i))
                        _value = _value_lst
                    else:
                        for _i in _value.split(','):
                            _value_lst.append(float(_i))
                        _value = _value_lst
            _param[_key] = _value
    # Exceptions
    # if sum(_param['number of stars in each ring']) > _param['number of stars']:
    #     raise Exception('sum of number of stars in each ring must be smaller than the total number of stars')
    # if _param['number of rings'] > _param['number of stars']:
    #     raise Exception('number of rings must be smaller than number of stars')
    # if len(_param['number of stars in each ring']) != len(_param['energy of each rings']):
    #     raise Exception('number of rings and number of energy must match')
    return _param


def prob_calculator(_cell_holder,  _pscoord, _param):
    _iter_mc = 10000
    _n_x = _param['number of x intervals']
    _m = _param['mass']
    _k = _param['spring constant']
    _E_limit = _param['maximum energy']
    _u = _pscoord[1]/_m
    _x_max = np.sqrt((2*_E_limit-_m*_u**2)/_k)
    _x_range = np.linspace(-_x_max, _x_max, _n_x)
    _dx = _x_range[1] - _x_range[0]
    _dp = 0.1  # finite width of momentum. (pdf의 차원을 맞춰주기 위해). 나중에 Monte-Carlo로 정확히 수정해야함.
    _minE = list()
    for _c in _cell_holder.list:
        _minE.append(_c.minE)

    _posterior = list()
    _likelihood = list()
    for _x in _x_range:
        for _c in _cell_holder.list:
            if _c.minE <= 1/2*_k*_x**2 + 1/2*_m*_u**2 < _c.maxE:
                _c.temp += _c.df * _dx * _dp
                _c.likelihood += (_dx * _dp)/_c.volume
        
    _norm = 0
    for _c in _cell_holder.list:
        _norm += _c.temp
    for _c in _cell_holder.list:
        _c.posterior = _c.temp/_norm
        _posterior.append(_c.posterior)
        _likelihood.append(_c.likelihood)

    return _minE, _posterior, _likelihood


def trajectory(_param, _coord):

    _k = _param['spring constant']
    _m = _param['mass']
    _omega = np.sqrt(_k/_m)
    _period = 2*np.pi/_omega

    _time = np.linspace(0, _period, 100)
    _x_trajectory = list()
    _p_trajectory = list()
    for _t in _time:
        _mat = np.array([[np.cos(_omega*_t), np.sin(_omega*_t)/(_m*_omega)],
                         [-_m*_omega*np.sin(_omega*_t), np.cos(_omega*_t)]])
        _pscoord = _mat @ _coord
        _x_trajectory.append(_pscoord[0])
        _p_trajectory.append(_pscoord[1])

    return _x_trajectory, _p_trajectory


def allocate_stars(_param):
    _w = list()
    _x_max = np.sqrt(2*_param['maximum energy']/_param['spring constant'])
    _n_stars = _param['number of stars']
    _n_rings = _param['number of rings']
    _k = _param['spring constant']
    _m = _param['mass']
    _omega = np.sqrt(_k/_m)
    _period = 2*np.pi/_omega

    if _n_rings == 0:  # if there is no substructure (rings)
        for _i in range(_n_stars):
            _coord = np.array([_x_max, 0]) * np.random.random_sample()
            _w.append(_coord)

    else:  # if there is a substructure (i.e. ring)

        _rings = dict()
        if type(_param['number of stars in each ring']) == int or type(_param['energy of each rings']) == int:
            _param['number of stars in each ring'] = [_param['number of stars in each ring']]
            _param['energy of each rings'] = [_param['energy of each rings']]
        _n_random = _n_stars - sum(_param['number of stars in each ring'])
        _e_rings = _param['energy of each rings']

        for _i in range(_n_rings+1):  # 0부터 ring 개수
            _rings[_i] = list()
            if _i == _n_rings:
                for _j in range(_n_random):
                    _coord = np.array([_x_max, 0]) * np.random.random_sample()
                    _rings[_i].append(_coord)
            else:
                _x = np.sqrt(2*_e_rings[_i]/_k)
                _coord = np.array([_x, 0])
                for _j in range(_param['number of stars in each ring'][_i]):
                    _rings[_i].append(_coord)

        for _i in range(_n_rings+1):
            _w += _rings[_i]

    for _i in range(len(_w)):
        _t = _period * np.random.random_sample()
        _mat = np.array([[np.cos(_omega*_t),np.sin(_omega*_t)/(_m*_omega)],
                         [-_m*_omega*np.sin(_omega*_t), np.cos(_omega*_t)]])
        _w[_i] = _mat @ _w[_i]

    return _w
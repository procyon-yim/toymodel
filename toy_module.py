import numpy as np


class Cell:

    def __init__(self):
        self.index = 0
        self.maxE = 0
        self.minE = 0
        self.df = 0
        self.volume = 0
        self.posterior = 0
        self.temp = 0

    def assign(self, _idx, _N_cell, _param):
        
        _m = _param['mass']
        _k = _param['spring constant']
        _E_limit = _param['maximum energy']
        _dE = _E_limit/_N_cell
        if _idx > _N_cell - 1:
            raise Exception('index out of range')
        self.index = _idx
        self.maxE = (_idx+1)*_dE
        self.minE = _idx*_dE
        self.volume = 2*np.pi*(self.maxE-self.minE)*np.sqrt(_m/_k)


class CellHolder:

    def __init__(self):
        self.list = []
    
    def construct(self, _N_cell, _param, _df_key):
        _m = _param['mass']
        _k = _param['spring constant']
        _E_limit = _param['maximum energy']
        for idx in range(_N_cell):
            _cell = Cell()
            _cell.assign(idx, _N_cell, _param)
            if _df_key == 'random':
                _cell.df = np.random.random_sample()
            elif _df_key == 'uniform':
                _cell.df = 1
            self.list.append(_cell)     
        
        # normalizing density function
        wsum_df = 0  # weighted sum of df in each cell
        for c in self.list:
            wsum_df += c.df * c.volume
        for c in self.list:
            c.df /= wsum_df


class PhaseSpaceCoord:
    def __init__(self):
        self.x = 0
        self.p = 0
        self.spr_const = 0
        self.mass = 0
        self.energy = 0
        self.max_energy = 0

    def config(self, _x, _v, _param):
        """
        Configure a phase space coordinate of the system.
        :param _x: x coordinate of the system
        :param _v: velocity of the system
        :param _param: parameter input (txt file)
        :return: none
        """
        _m = _param['mass']
        _k = _param['spring constant']
        _E_limit = _param['maximum energy']

        self.x = _x
        self.p = _m*_v
        self.spr_const = _k
        self.mass = _m
        self.energy = self.p**2/(2*_m) + (1/2)*_k*_x**2
        self.max_energy = _E_limit

    def is_in(self, _cell):
        """
        Checks whether a system is within a given cell.
        :param _cell: Cell object
        :return: True if the system is in the cell, False if not.
        """
        if _cell.minE <= self.energy < _cell.maxE:
            return True
        else:
            return False


class Potential:
    def __init__(self, _param):
        """
        :param _param: Input parameters (txt file)
        """
        self.mass = _param['mass']
        self.spr_const = _param['spring constant']
        self.omega = np.sqrt(_param['spring constant']/_param['mass'])


def read_params(_param_file):
    with open(_param_file) as _f:
        _param = dict()
        for _l in _f.readlines():
            _key = _l.split('=')[0]
            _value = _l.split('=')[1]
            if '\n' in _value:
                if ',' not in _value:
                    _value = int(_value[:_value.index('\n')])
                else:
                    _value = _value[:_value.index('\n')]
                    _value_lst = []
                    for _i in _value.split(','):
                        _value_lst.append(int(_i))
                    _value = _value_lst
            else:
                if ',' not in _value:
                    _value = int(_value[:_value.index('\n')])
                else:
                    _value_lst = []
                    for _i in _value.split(','):
                        _value_lst.append(int(_i))
                    _value = _value_lst
            _param[_key] = _value
    # Exceptions
    if sum(_param['number of stars in each ring']) > _param['number of stars']:
        raise Exception('sum of number of stars in each ring must be smaller than the total number of stars')
    if _param['number of rings'] > _param['number of stars']:
        raise Exception('number of rings must be smaller than number of stars')
    if len(_param['number of stars in each ring']) != len(_param['energy of each rings']):
        raise Exception('number of rings and number of energy must match')
    return _param


def prob_calculator(_cell_holder,  _u, _param, _n_x):
    """
    :param _cell_holder: CellHolder class object
    :param _u: observed velocity of the system.
    :param _param: parameter input (txt file)
    :param _n_x: number of intervals in x
    :return: none
    """
    _m = _param['mass']
    _k = _param['spring constant']
    _E_limit = _param['maximum energy']
    _x_max = np.sqrt((2*_E_limit-_m*_u**2)/_k)
    _x_range = np.linspace(-_x_max, _x_max, _n_x)
    _dx = _x_range[1] - _x_range[0]
    
    for _x in _x_range:
        _coord = PhaseSpaceCoord()
        _coord.config(_x, _u, _param)
        for _c in _cell_holder.list:
            if _coord.is_in(_c):
                _c.temp += _c.df * _dx
        
    _norm = 0
    for _c in _cell_holder.list:
        _norm += _c.temp
    for _c in _cell_holder.list:
        _c.posterior = _c.temp/_norm


def trajectory(_pot, _init):
    """
    :param _pot: object of Potential class
    :param _init: (numpy.array) initial phase space coordinate
    :return: (list) a list of lists, which consists of x and p coordinate of the system throughout the period.
    """
    _omega = _pot.omega
    _period = 2*np.pi/_omega
    _m = _pot.mass
    _time = np.linspace(0, _period, 100)
    _x_trajectory = list()
    _p_trajectory = list()
    for _t in _time:
        _mat = np.array([[np.cos(_omega*_t),np.sin(_omega*_t)/(_m*_omega)],
                         [-_m*_omega*np.sin(_omega*_t),np.cos(_omega*_t)]])
        _pscoord = _mat @ _init
        _x_trajectory.append(_pscoord[0])
        _p_trajectory.append(_pscoord[1])

    return _x_trajectory, _p_trajectory


def observe(_pot, _time, _init):
    """
    Returns a observed phase-space coordinate of a system in a potential, at a given time.
    :param _pot: Class object Potential.
    :param _time: (float)
    :param _init: (numpy.array) 2 by 1 initial phase space coordinate vector.
    :return: (numpy.array)
    """
    _omega = _pot.omega
    _m = _pot.mass
    _mat = np.array([[np.cos(_omega*_time),np.sin(_omega*_time)/(_m*_omega)],
                     [-_m*_omega*np.sin(_omega*_time), np.cos(_omega*_time)]])

    return _mat @ _init


def allocate_stars(_param):
    _w_init = list()
    _x_max = np.sqrt(2*_param['maximum energy']/_param['spring constant'])
    _n_stars = _param['number of stars']
    _n_rings = _param['number of rings']
    _k = _param['spring constant']

    if _n_rings == 0:  # if there is no substructure (rings)
        for _i in range(_n_stars):
            _coord = np.array([_x_max, 0]) * np.random.random_sample()
            _w_init.append(_coord)
        return _w_init

    else:  # if there is a substructure (i.e. ring)

        _rings = dict()
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
            _w_init += _rings[_i]

        return _w_init
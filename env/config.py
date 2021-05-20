import numpy as np

from .acrobot import *
from .pendulum import *


config = {

    'acrobot': {
        'dt': .1,
        'N': 50,
        'x_init': np.array([np.pi * 0.95, 0., 0., 0.]),
        'x_target': np.array([np.pi, 0., 0., 0.]),
        'design_init': np.array([1., 2., 1., 2.]),
        'design_name': ['m1', 'm2', 'l1', 'l2'],
    },

    'pendulum': {
        'dt': .1,
        'N': 50,
        'x_init': np.array([np.pi * 0.95, 0.]),
        'x_target': np.array([np.pi, 0.]),
        'design_init': np.array([1., 2.]),
        'design_name': ['m', 'l'],
    },

}


utils = {
    
    'acrobot': {
        'sim': AcrobotSimTorch,
        'cost': AcrobotCost,
        'animate': animate_acrobot,
    },

    'pendulum': {
        'sim': PendulumSimTorch,
        'cost': PendulumCost,
        'animate': animate_pendulum,
    },

}
import numpy as np


config = {

    'acrobot': {
        'dt': .1,
        'N': 50,
        'x_init': np.array([np.pi * 0.95, 0., 0., 0.]),
        'x_target': np.array([np.pi, 0., 0., 0.]),
        'design_init': np.array([1.]),
        'design_name': ['m1', 'm2', 'l1', 'l2'],
        'use_rk4': True,
    },

    'pendulum': {
        'dt': .1,
        'N': 50,
        'x_init': np.array([0., 0.]),
        'x_target': np.array([np.pi, 0.]),
        'design_init': np.array([1.]),
        'design_name': ['m', 'l'],
        'use_rk4': False,
    },

}

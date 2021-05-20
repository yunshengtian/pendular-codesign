from .acrobot import *
from .pendulum import *


utils = {
    
    'acrobot': {
        'animate': AcrobotAnimation,
        'cost': AcrobotCost,
        'sim': AcrobotSim,
    },

    'pendulum': {
        'animate': PendulumAnimation,
        'cost': PendulumCost,
        'sim': PendulumSim,
    },

}

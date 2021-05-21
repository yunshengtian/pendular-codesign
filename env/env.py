import numpy as np
import gym

from .config import config
from .utils import utils


class Env(gym.Env):

    name = None
    
    def __init__(self, design=None, x_init=None, x_target=None, N=None, dt=None, use_rk4=None):

        assert self.name in config and self.name in utils
        env_config, env_utils = config[self.name], utils[self.name]

        if design is None: design = env_config['design_init']
        if x_init is None: x_init = env_config['x_init']
        if x_target is None: x_target = env_config['x_target']
        if N is None: N = env_config['N']
        if dt is None: dt = env_config['dt']
        if use_rk4 is None: use_rk4 = env_config['use_rk4']

        self.x_init = x_init
        self.N = N

        Sim = env_utils['sim']
        Cost = env_utils['cost']
        self.sim = Sim(design, dt, use_rk4)
        self.cost = Cost(x_target)
            
        self.x = None
        self.n = None

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(x_init),))
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(1,))

    def step(self, u):
        assert self.n < self.N, 'env needs reset'

        cost = self.cost.cost_stage(self.x, u)
        self.x = self.sim.discrete_dynamics(self.x, u)
        self.n += 1

        if self.n == self.N:
            cost += self.cost.cost_final(self.x)
            done = True
        else:
            done = False
        
        return self.x, -cost, done, {}

    def reset(self):
        self.x = self.x_init
        self.n = 0
        return self.x

import numpy as np
import gym
from .sim import AcrobotSim
from .cost import AcrobotCost


class AcrobotEnv(gym.Env):
    
    def __init__(self,
        design = np.array([1, 2, 1, 2]),
        x_init = np.array([0.9 * np.pi, 0, 0, 0]),
        x_target = np.array([np.pi, 0, 0, 0]),
        N = 50,
        dt = 0.1,
    ):
        self.sim = AcrobotSim(design, dt)
        self.cost = AcrobotCost(x_target)
        self.x_init = x_init
        self.N = N
            
        self.x = None
        self.n = None

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(1,))

    def step(self, u):
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


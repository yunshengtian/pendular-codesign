import numpy as np
import symengine as se
import torch

from ..sim import Sim


class PendulumSim(Sim):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inertia = 4

    def get_design_params(self, design):
        [m] = design
        l = (self.inertia / m) ** 0.5
        return m, l

    def continuous_dynamics(self, x, u):

        m, l = self.get_design_params(self.design)
        g = self.g

        s = np.sin(x[0])
        b = 1

        xdot = np.array([
            x[1],
            (u[0] - m * g * l * s - b * x[1]) / (m * l ** 2)
        ])

        return xdot

    def continuous_dynamics_sym(self, x, u):

        m, l = self.get_design_params(self.design)
        g = self.g

        s = se.sin(x[0])
        b = 1

        xdot = se.zeros(2, 1)
        xdot[0] = x[1]
        xdot[1] = (u[0] - m * g * l * s - b * x[1]) / (m * l ** 2)

        return xdot

    def continuous_dynamics_torch(self, x, u):

        m, l = self.get_design_params(self.design)
        g = self.g

        s = torch.sin(x[0])
        b = 1

        xdot = torch.stack([
            x[1],
            (u[0] - m * g * l * s - b * x[1]) / (m * l ** 2)
        ])

        return xdot

import numpy as np
import symengine as se
import torch


class PendulumSim:

    def __init__(self, design, dt=0.1):
        self.design = design # m, l
        self.g = 9.81
        self.dt = dt

    def continuous_dynamics(self, x, u):
        m, l = self.design
        g = self.g

        s = np.sin(x[0])
        b = 1

        xdot = np.array([
            x[1],
            (u[0] - m * g * l * s - b * x[1]) / (m * l ** 2)
        ])

        return xdot

    def discrete_dynamics(self, x, u):
        xdot = self.continuous_dynamics(x, u)
        x_next = x + xdot * self.dt
        return x_next

    def continuous_dynamics_sym(self, x, u):
        m, l = self.design
        g = self.g

        s = se.sin(x[0])
        b = 1

        xdot = se.zeros(2, 1)
        xdot[0] = x[1]
        xdot[1] = (u[0] - m * g * l * s - b * x[1]) / (m * l ** 2)

        return xdot

    def discrete_dynamics_sym(self, x, u):
        xdot = self.continuous_dynamics_sym(x, u)
        x_next = x + xdot * self.dt
        return x_next

    def rollout(self, x0, u_trj):
        x_trj = np.zeros((u_trj.shape[0] + 1, x0.shape[0]))
        x_trj[0] = x0
        for i in range(len(u_trj)):
            x_trj[i + 1] = self.discrete_dynamics(x_trj[i], u_trj[i])
        return x_trj

    def set_design(self, design):
        self.design = design
    

class PendulumSimTorch:
    
    def __init__(self, design, dt):
        self.design = design # requires_grad = True
        self.g = 9.81
        self.dt = dt

    def continuous_dynamics(self, x, u):
        m, l = self.design
        g = self.g

        s = torch.sin(x[0])
        b = 1

        xdot = torch.stack([
            x[1],
            (u[0] - m * g * l * s - b * x[1]) / (m * l ** 2)
        ])

        return xdot

    def discrete_dynamics(self, x, u):
        xdot = self.continuous_dynamics(x,u)
        x_next = x + xdot * self.dt
        return x_next

    def rollout(self, x0, u_trj):
        x_trj = [None] * (len(u_trj) + 1)
        x_trj[0] = x0
        for i in range(len(u_trj)):
          x_trj[i+1] = self.discrete_dynamics(x_trj[i],u_trj[i])
        return torch.stack(x_trj)
    
    def set_design(self, design):
        self.design = design

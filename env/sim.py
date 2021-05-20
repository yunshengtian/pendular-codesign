import numpy as np
import symengine as se
import torch


class Sim:

    def __init__(self, design, dt, use_rk4=False):
        self.design = design
        self.g = 9.81
        self.dt = dt
        self.use_rk4 = use_rk4

    def set_design(self, design):
        self.design = design

    def euler(self, continuous_dynamics, x, u):
        xdot = continuous_dynamics(x, u)
        x_next = x + xdot * self.dt
        return x_next

    def rk4(self, continuous_dynamics, x, u):
        dt = self.dt
        dt2 = dt / 2.0
        k1 = continuous_dynamics(x, u)
        k2 = continuous_dynamics(x + dt2 * k1, u)
        k3 = continuous_dynamics(x + dt2 * k2, u)
        k4 = continuous_dynamics(x + dt * k3, u)
        x_next = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    def continuous_dynamics(self, x, u):
        raise NotImplementedError

    def continuous_dynamics_sym(self, x, u):
        raise NotImplementedError

    def continuous_dynamics_torch(self, x, u):
        raise NotImplementedError

    def _discrete_dynamics(self, continuous_dynamics, x, u):
        if self.use_rk4:
            x_next = self.rk4(continuous_dynamics, x, u)
        else:
            x_next = self.euler(continuous_dynamics, x, u)
        return x_next

    def discrete_dynamics(self, x, u):
        return self._discrete_dynamics(self.continuous_dynamics, x, u)

    def discrete_dynamics_sym(self, x, u):
        return self._discrete_dynamics(self.continuous_dynamics_sym, x, u)

    def discrete_dynamics_torch(self, x, u):
        return self._discrete_dynamics(self.continuous_dynamics_torch, x, u)

    def _rollout(self, discrete_dynamics, x0, u_trj):
        x_trj = [None] * (len(u_trj) + 1)
        x_trj[0] = x0
        for i in range(len(u_trj)):
            x_trj[i + 1] = discrete_dynamics(x_trj[i], u_trj[i])
        return x_trj

    def rollout(self, x0, u_trj):
        return np.array(self._rollout(self.discrete_dynamics, x0, u_trj))

    def rollout_sym(self, x0, u_trj):
        return se.Matrix(self._rollout(self.discrete_dynamics_sym, x0, u_trj))

    def rollout_torch(self, x0, u_trj):
        return torch.stack(self._rollout(self.discrete_dynamics_torch, x0, u_trj))

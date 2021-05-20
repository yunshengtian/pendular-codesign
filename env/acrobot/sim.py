import numpy as np
import symengine as se
import torch


class AcrobotSim:

    def __init__(self, design, dt=0.1):
        self.design = design # m1, m2, l1, l2
        self.g = 9.81
        self.dt = dt

    def continuous_dynamics(self, x, u):
        m1, m2, l1, l2 = self.design
        g = self.g

        s1, s2, s12 = np.sin(x[0]), np.sin(x[1]), np.sin(x[0] + x[1])
        c1, c2 = np.cos(x[0]), np.cos(x[1])
        q1dot, q2dot = x[2], x[3]
        lc1 = 0.5 * l1
        lc2 = 0.5 * l2
        I1 = m1 * l1 ** 2
        I2 = m2 * l2 ** 2

        M = np.array([
            [I1 + I2 + m2 * l1 ** 2 + 2 * m2 * l1 * lc2 * c2, I2 + m2 * l1 * lc2 * c2],
            [I2 + m2 * l1 * lc2 * c2, I2]
        ])
        M_inv = np.linalg.inv(M)

        C = np.array([
            [-2 * m2 * l1 * lc2 * s2 * q2dot, -m2 * l1 * lc2 * s2 * q2dot],
            [m2 * l1 * lc2 * s2 * q1dot, 0]
        ])

        tau = np.array([
            -m1 * g * lc1 * s1 - m2 * g * (l1 * s1 + lc2 * s12),
            -m2 * g * lc2 * s12
        ])

        B = np.array([0, 1])

        xdot = np.concatenate([
            x[2:],
            M_inv @ (tau + B * u - C @ x[2:])
        ])

        return xdot

    def discrete_dynamics(self, x, u):
        xdot = self.continuous_dynamics(x, u)
        x_next = x + xdot * self.dt
        return x_next

    def continuous_dynamics_sym(self, x, u):
        m1, m2, l1, l2 = self.design
        g = self.g

        s1, s2, s12 = se.sin(x[0]), se.sin(x[1]), se.sin(x[0] + x[1])
        c1, c2 = se.cos(x[0]), se.cos(x[1])
        q1dot, q2dot = x[2], x[3]
        lc1 = 0.5 * l1
        lc2 = 0.5 * l2
        I1 = m1 * l1 ** 2
        I2 = m2 * l2 ** 2

        M = se.Matrix([
            [I1 + I2 + m2 * l1 ** 2 + 2 * m2 * l1 * lc2 * c2, I2 + m2 * l1 * lc2 * c2],
            [I2 + m2 * l1 * lc2 * c2, I2]
        ])
        M_inv = M.inv()

        C = se.Matrix([
            [-2 * m2 * l1 * lc2 * s2 * q2dot, -m2 * l1 * lc2 * s2 * q2dot],
            [m2 * l1 * lc2 * s2 * q1dot, 0]
        ])

        tau = se.Matrix([
            [-m1 * g * lc1 * s1 - m2 * g * (l1 * s1 + lc2 * s12)],
            [-m2 * g * lc2 * s12]
        ])

        B = se.Matrix([[0], [1]])

        xdot = se.zeros(4, 1)
        xdot[:2, :] = x[2:, :]
        xdot[2:, :] = M_inv * (tau + B * u - C * x[2:, :])

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


class AcrobotSimTorch:
    
    def __init__(self, design, dt):
        self.design = design # requires_grad = True
        self.g = 9.81
        self.dt = dt

    def continuous_dynamics(self, x, u):
        m1, m2, l1, l2 = self.design
        g = self.g

        s1, s2, s12 = torch.sin(x[0]), torch.sin(x[1]), torch.sin(x[0] + x[1])
        c1, c2 = torch.cos(x[0]), torch.cos(x[1])
        q1dot, q2dot = x[2], x[3]
        lc1 = 0.5 * l1
        lc2 = 0.5 * l2
        I1 = m1 * l1 ** 2
        I2 = m2 * l2 ** 2

        M = torch.stack([
            torch.stack([I1 + I2 + m2 * l1 ** 2 + 2 * m2 * l1 * lc2 * c2, I2 + m2 * l1 * lc2 * c2]),
            torch.stack([I2 + m2 * l1 * lc2 * c2, I2])
        ])

        C = torch.stack([
            torch.stack([-2 * m2 * l1 * lc2 * s2 * q2dot, -m2 * l1 * lc2 * s2 * q2dot]),
            torch.stack([m2 * l1 * lc2 * s2 * q1dot, torch.tensor(0)])
        ])

        tau = torch.stack([
            -m1 * g * lc1 * s1 - m2 * g * (l1 * s1 + lc2 * s12),
            -m2 * g * lc2 * s12
        ])

        B = torch.tensor([0.0, 1.0])

        M_inv = M.inverse()

        xdot = torch.cat([
            x[2:],
            M_inv @ (tau + B*u - C @ x[2:])
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
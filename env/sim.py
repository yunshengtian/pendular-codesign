import numpy as np
import symengine as se


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

    def discrete_dynamics(self, x, u, use_rk4 = True):
        if use_rk4:
            x_next = self.rk4(x,u)
        else:
            xdot = self.continuous_dynamics(x, u)
            x_next = x + xdot * self.dt
        return x_next

    def rk4(self, x, u):
        dt = self.dt
        dt2 = dt / 2.0
        k1 = self.continuous_dynamics(x, u)
        k2 = self.continuous_dynamics(x+dt2 * k1, u)
        k3 = self.continuous_dynamics(x+dt2 * k2, u)
        k4 = self.continuous_dynamics(x+dt * k3, u)
        x_next = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
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

    def discrete_dynamics_sym(self, x, u, use_rk4 = True):
        if use_rk4:
            x_next = self.rk4_sym(x,u)
        else:   
            xdot = self.continuous_dynamics_sym(x, u)
            x_next = x + xdot * self.dt
        return x_next

    def rk4_sym(self,x,u):
        dt = self.dt
        dt2 = dt / 2.0
        k1 = self.continuous_dynamics_sym(x, u)
        k2 = self.continuous_dynamics_sym(x+dt2 * k1, u)
        k3 = self.continuous_dynamics_sym(x+dt2 * k2, u)
        k4 = self.continuous_dynamics_sym(x+dt * k3, u)
        x_next = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    def rollout(self, x0, u_trj):
        x_trj = np.zeros((u_trj.shape[0] + 1, x0.shape[0]))
        x_trj[0] = x0
        for i in range(len(u_trj)):
            x_trj[i + 1] = self.discrete_dynamics(x_trj[i], u_trj[i])
        return x_trj
    
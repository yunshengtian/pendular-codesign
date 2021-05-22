from ..cost import Cost


class PendulumCost(Cost):

    def __init__(self, x_target):
        super().__init__(x_target)
        self.eps = 1e-6

    def cost_stage(self, x, u):
        c_theta = (x[0] - self.x_target[0]) ** 2 + self.eps
        c_thetadot = x[1] ** 2
        c_control = u[0] ** 2
        return 10 * c_theta + 1 * c_thetadot + 0.1 * c_control

    def cost_final(self, x):
        c_theta = (x[0] - self.x_target[0]) ** 2 + self.eps
        c_thetadot = x[1] ** 2
        return 10 * c_theta + 1 * c_thetadot

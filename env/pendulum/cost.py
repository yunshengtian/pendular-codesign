class PendulumCost:

    def __init__(self, x_target):
        self.x_target = x_target
        self.eps = 1e-6

    def cost_stage(self, x, u):
        c_theta = (x[0] - self.x_target[0]) ** 2 + self.eps
        c_thetadot = x[1] ** 2
        c_control = u[0] ** 2
        return 10 * c_theta + 1 * c_thetadot + 0.1 * c_control

    def cost_final(self, x):
        c_theta = (x[0] - self.x_target[0]) ** 2 + self.eps
        c_thetadot = x[1] ** 2
        return 10 * c_theta + 0.1 * c_thetadot

    def cost_rollout(self, x_trj, u_trj):
        cost = 0.0
        assert len(x_trj) == len(u_trj) + 1
        for i in range(len(u_trj)):
            cost += self.cost_stage(x_trj[i], u_trj[i])
        cost += self.cost_final(x_trj[-1])
        return cost
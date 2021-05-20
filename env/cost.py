class Cost:

    def __init__(self, x_target):
        self.x_target = x_target

    def cost_stage(self, x, u):
        raise NotImplementedError

    def cost_final(self, x):
        raise NotImplementedError

    def cost_rollout(self, x_trj, u_trj):
        cost = 0.0
        assert len(x_trj) == len(u_trj) + 1
        for i in range(len(u_trj)):
            cost += self.cost_stage(x_trj[i], u_trj[i])
        cost += self.cost_final(x_trj[-1])
        return cost

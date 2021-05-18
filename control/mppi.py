import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


class MPPI:
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, env, K, T, U, lambda_=1.0, noise_mu=0, noise_sigma=1, u_init=1, num_cores=1):
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = u_init
        self.cost_total = np.zeros(shape=(self.K))

        self.env = env
        self.env.reset()
        self.x_init = self.env.x_init

        self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T))
        self.num_cores = num_cores

    def _compute_rollout_cost(self, k):
        u_trj = (self.U[:self.T] + self.noise[k, :self.T])[:, None]
        x_trj = self.env.sim.rollout(self.x_init, u_trj)
        return self.env.cost.cost_rollout(x_trj, u_trj)

    def _ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))

    def control(self, iter=1000):
        final_U = []

        if self.num_cores > 1:
            pool = Pool(self.num_cores)
        else:
            pool = None
        
        for _ in tqdm(range(iter)):
            if self.num_cores > 1:
                self.cost_total[:] = pool.map(self._compute_rollout_cost, np.arange(self.K))
            else:
                for k in range(self.K):
                    self.cost_total[k] = self._compute_rollout_cost(k)

            beta = np.min(self.cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1/self.lambda_)

            eta = np.sum(cost_total_non_zero)
            omega = 1/eta * cost_total_non_zero

            self.U += [np.sum(omega * self.noise[:, t]) for t in range(self.T)]

            s, r, _, _ = self.env.step([self.U[0]])
            final_U.append(self.U[0])

            self.U = np.roll(self.U, -1)  # shift all elements to the left
            self.U[-1] = self.u_init  #
            self.cost_total[:] = 0

        return np.array(final_U)


if __name__ == "__main__":

    import os, sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import env
    import gym

    design = np.array([1, 2, 1, 2])
    x_init = np.array([0.98 * np.pi, 0, 0, 0])
    x_target = np.array([np.pi, 0, 0, 0])
    N = 100
    dt = 0.05

    env = gym.make('acrobot-v0', design=design, x_init=x_init, x_target=x_target, N=N, dt=dt)

    TIMESTEPS = 10 # T
    N_SAMPLES = 1000  # K
    ACTION_LOW = -10
    ACTION_HIGH = 10

    noise_mu = 0
    noise_sigma = 10
    lambda_ = 1e-2

    U = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=TIMESTEPS)  # pendulum joint effort in (-2, +2)

    mppi = MPPI(env=env, 
        K=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mu=noise_mu, noise_sigma=noise_sigma, u_init=0,
        num_cores=4)
    u_trj = mppi.control(iter=N)
    x_trj = env.sim.rollout(x_init, u_trj)
    final_cost = env.cost.cost_rollout(x_trj, u_trj[:, None])

    print(f'Final cost: {final_cost}')

    from animate import animate
    animate(design, x_trj, N)

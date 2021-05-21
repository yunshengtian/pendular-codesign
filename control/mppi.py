import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from .config import config


class MPPI:
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, env, K=None, T=None, lambda_=None, noise_mu=None, noise_sigma=None, u_init_sigma=None, num_cores=None):

        self.env = env

        control_config = config['mppi']
        self.K = control_config['K'] if K is None else K
        self.T = control_config['T'] if T is None else T
        self.lambda_ = control_config['lambda'] if lambda_ is None else lambda_
        self.noise_mu = control_config['noise_mu'] if noise_mu is None else noise_mu
        self.noise_sigma = control_config['noise_sigma'] if noise_sigma is None else noise_sigma
        self.u_init_sigma = control_config['u_init_sigma'] if u_init_sigma is None else u_init_sigma
        self.num_cores = control_config['num_cores'] if num_cores is None else num_cores

        self.n_u = self.env.action_space.shape[0]

    def _compute_rollout_cost(self, x_init, u_T, noise):
        u_T += noise
        x_T = self.env.sim.rollout(x_init, u_T)
        return self.env.cost.cost_rollout(x_T, u_T)

    def _ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))

    def solve(self, u_trj_init=None):
        x_trj = []
        u_trj = []
        cost = 0.0

        cost_total = np.zeros(self.K)
        noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T, self.n_u))

        N = self.env.spec.max_episode_steps

        if u_trj_init is None:
            u_trj_init = np.random.randn(N - 1, self.n_u) * self.u_init_sigma

        u_T = u_trj_init[:self.T]
        
        if self.num_cores > 1:
            pool = Pool(self.num_cores)
        else:
            pool = None

        x_trj.append(self.env.reset())
        
        for i in tqdm(range(N), desc='MPPI'):
            T = min(N - i, self.T)
            if self.num_cores > 1:
                cost_total[:] = pool.starmap(self._compute_rollout_cost, [(x_trj[-1], u_T[:T], noise[k, :T]) for k in range(self.K)])
            else:
                for k in range(self.K):
                    cost_total[k] = self._compute_rollout_cost(x_trj[-1], u_T[:T], noise[k, :T])

            beta = np.min(cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=cost_total, beta=beta, factor=1/self.lambda_)

            eta = np.sum(cost_total_non_zero)
            omega = (1/eta * cost_total_non_zero)[:, None]

            u_T += np.array([np.sum(omega * noise[:, t, :], axis=0) for t in range(self.T)])

            s, r, _, _ = self.env.step(u_T[0])
            x_trj.append(s)
            u_trj.append(u_T[0])
            cost += -r

            u_T = np.roll(u_T, -1)  # shift all elements to the left
            if i + self.T < N - 1:
                u_T[-1] = u_trj_init[i + self.T]

        if pool is not None:
            pool.terminate()

        x_trj, u_trj = np.array(x_trj), np.array(u_trj)
        return x_trj, u_trj, {'cost': cost}

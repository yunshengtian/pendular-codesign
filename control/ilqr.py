import numpy as np
import symengine as se
from tqdm import tqdm
from time import time


class Derivatives:
    
    def __init__(self, discrete_dynamics, cost_stage, cost_final, n_x, n_u):
        x = se.Matrix(se.symbols(f'x:{n_x}'))
        u = se.Matrix(se.symbols(f'u:{n_u}'))
        
        l = cost_stage(x, u)
        self.l_x = se.Matrix([l]).jacobian(x)
        self.l_u = se.Matrix([l]).jacobian(u)
        self.l_xx = self.l_x.T.jacobian(x)
        self.l_ux = self.l_u.T.jacobian(x)
        self.l_xu = self.l_x.T.jacobian(u)
        self.l_uu = self.l_u.T.jacobian(u)
        
        l_final = cost_final(x)
        self.l_final_x = se.Matrix([l_final]).jacobian(x)
        self.l_final_xx = self.l_final_x.T.jacobian(x)
        
        f = discrete_dynamics(x, u)
        self.f_x = f.jacobian(x)
        self.f_u = f.jacobian(u)
    
    def stage(self, x, u):
        env = {f'x{i}': x[i] for i in range(len(x))}
        env.update({f'u{i}': u[i] for i in range(len(u))})

        l_x = np.array(self.l_x.subs(env).tolist(), dtype=float).ravel()
        l_u = np.array(self.l_u.subs(env).tolist(), dtype=float).ravel()
        l_xx = np.array(self.l_xx.subs(env).tolist(), dtype=float)
        l_ux = np.array(self.l_ux.subs(env).tolist(), dtype=float)
        l_uu = np.array(self.l_uu.subs(env).tolist(), dtype=float)

        f_x = np.array(self.f_x.subs(env).tolist(), dtype=float)
        f_u = np.array(self.f_u.subs(env).tolist(), dtype=float)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u
    
    def final(self, x):
        env = {f'x{i}': x[i] for i in range(len(x))}

        l_final_x = np.array(self.l_final_x.subs(env).tolist(), dtype=float).ravel()
        l_final_xx = np.array(self.l_final_xx.subs(env).tolist(), dtype=float)
        
        return l_final_x, l_final_xx


class ILQR:

    def __init__(self, env):
        self.env = env
        self.n_x, self.n_u = 4, 1
        self.derivs = Derivatives(
            self.env.sim.discrete_dynamics_sym, 
            self.env.cost.cost_stage, 
            self.env.cost.cost_final, 
            self.n_x, self.n_u
        )
    
    def Q_terms(self, l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)
        Q_ux = l_ux + f_u.T.dot(V_xx).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx).dot(f_u)
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

    def gains(self, Q_uu, Q_u, Q_ux):
        Q_uu_inv = np.linalg.inv(Q_uu)
        k = - Q_uu_inv.dot(Q_u)
        K = - Q_uu_inv.dot(Q_ux)
        return k, K

    def V_terms(self, Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
        V_x = Q_x + K.T.dot(Q_u) + Q_ux.T.dot(k) + K.T.dot(Q_uu).dot(k)
        V_xx = Q_xx  + 2*K.T.dot(Q_ux) + K.T.dot(Q_uu).dot(K)
        return V_x, V_xx

    def expected_cost_reduction(self, Q_u, Q_uu, k):
        return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))

    def forward_pass(self, x_trj, u_trj, k_trj, K_trj):
        x_trj_new = np.zeros(x_trj.shape)
        x_trj_new[0,:] = x_trj[0,:]
        u_trj_new = np.zeros(u_trj.shape)
        for n in range(u_trj.shape[0]):
            u_trj_new[n,:] = u_trj[n,:] +  k_trj[n,:] + K_trj[n].dot(x_trj_new[n,:] - x_trj[n,:]) # Apply feedback law
            x_trj_new[n+1,:] = self.env.sim.discrete_dynamics(x_trj_new[n], u_trj_new[n]) # Apply dynamics
        return x_trj_new, u_trj_new

    def backward_pass(self, x_trj, u_trj, regu):
        k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
        K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
        expected_cost_redu = 0
        V_x = np.zeros((x_trj.shape[1],))
        V_xx = np.zeros((x_trj.shape[1],x_trj.shape[1]))
        V_x,V_xx = self.derivs.final(x_trj[-1])
        for n in range(u_trj.shape[0]-1, -1, -1):
            l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = self.derivs.stage(x_trj[n],u_trj[n])
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self.Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
            # We add regularization to ensure that Q_uu is invertible and nicely conditioned
            Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
            k, K = self.gains(Q_uu_regu, Q_u, Q_ux)
            k_trj[n,:] = k
            K_trj[n,:,:] = K
            V_x, V_xx = self.V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
            expected_cost_redu += self.expected_cost_reduction(Q_u, Q_uu, k)
        return k_trj, K_trj, expected_cost_redu

    def run_ilqr(self, x0, N, max_iter=50, regu_init=100):
        # First forward rollout
        u_trj = np.random.randn(N-1, self.n_u)*0.0001
        x_trj = self.env.sim.rollout(x0, u_trj)
        total_cost = self.env.cost.cost_rollout(x_trj, u_trj)
        regu = regu_init
        max_regu = 10000
        min_regu = 0.01

        # Setup traces
        cost_trace = [total_cost]
        expected_cost_redu_trace = []
        redu_ratio_trace = [1]
        redu_trace = []
        regu_trace = [regu]

        # Run main loop
        for it in tqdm(range(max_iter)):
            # Backward and forward pass
            k_trj, K_trj, expected_cost_redu = self.backward_pass(x_trj, u_trj, regu)
            x_trj_new, u_trj_new = self.forward_pass(x_trj, u_trj, k_trj, K_trj)
            # Evaluate new trajectory
            total_cost = self.env.cost.cost_rollout(x_trj_new, u_trj_new)
            cost_redu = cost_trace[-1] - total_cost
            redu_ratio = cost_redu / abs(expected_cost_redu)
            # Accept or reject iteration
            if cost_redu > 0:
                # Improvement! Accept new trajectories and lower regularization
                redu_ratio_trace.append(redu_ratio)
                cost_trace.append(total_cost)
                x_trj = x_trj_new
                u_trj = u_trj_new
                regu *= 0.7
            else:
                # Reject new trajectories and increase regularization
                regu *= 2.0
                cost_trace.append(cost_trace[-1])
                redu_ratio_trace.append(0)
            regu = min(max(regu, min_regu), max_regu)
            regu_trace.append(regu)
            redu_trace.append(cost_redu)

            # Early termination if expected improvement is small
            if expected_cost_redu <= 1e-6:
                break

        return x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace

    def solve(self, x0, N, max_iter=50, regu_init=100):
        x_trj, u_trj, cost_trace, _, _, _ = self.run_ilqr(x0, N, max_iter, regu_init)
        return x_trj, u_trj, cost_trace


if __name__ == '__main__':

    import os, sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import env
    import gym
    import matplotlib.pyplot as plt

    design = np.array([1, 2, 1, 2])
    x_init = np.array([0.98 * np.pi, 0, 0, 0])
    x_target = np.array([np.pi, 0, 0, 0])
    N = 100
    dt = 0.05

    env = gym.make('acrobot-v0', design=design, x_init=x_init, x_target=x_target, N=N, dt=dt)

    ilqr = ILQR(env)
    x_trj, u_trj, cost_trace = ilqr.solve(x_init, N, max_iter=200)

    plt.plot(list(range(len(cost_trace))), cost_trace)
    plt.title('Cost')
    plt.show()

    print(f'Final cost: {cost_trace[-1]}')

    from animate import animate
    animate(design, x_trj, N)
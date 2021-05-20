import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from env.config import config, utils
import env
from control import ILQR, MPPI


def torch2np(tensor):
    return tensor.clone().detach().numpy()


def optimize_design(env_name, num_iter, lr):

    # set up env config
    env_config = config[env_name]
    dt, N, x_init, x_target, design = env_config['dt'], env_config['N'], env_config['x_init'], env_config['x_target'], env_config['design_init']
    x_init_torch, x_target_torch, design_torch = torch.tensor(x_init), torch.tensor(x_target), torch.tensor(design, requires_grad=True)

    # set up env utils
    env_utils = utils[env_name]
    env = gym.make(f'{env_name}-v0', design=design, x_init=x_init, x_target=x_target, N=N, dt=dt)
    cost_torch = env_utils['cost'](x_target_torch)
    sim_torch = env_utils['sim'](design_torch, dt)
    optimizer = torch.optim.Adam([design_torch], lr=lr)

    results = {
        'loss': [],
        'design': [],
        'x_trj': [],
        'u_trj': [],
    }

    u_trj_last = None
    
    for i in tqdm(range(num_iter)):

        # solve for optimal control
        design = torch2np(design_torch)
        env.sim.set_design(design)
        _, u_trj, _ = ILQR(env).solve(x_init, N, max_iter=1000, u_trj_init=u_trj_last)
        u_trj_last = u_trj

        # compute differentiable loss
        sim_torch.set_design(design_torch)
        u_trj_torch = torch.tensor(u_trj, dtype=torch.float32)
        x_trj_torch = sim_torch.rollout(x_init_torch, u_trj_torch)
        loss_torch = cost_torch.cost_rollout(x_trj_torch, u_trj_torch[:, None])
        
        # optimize design
        optimizer.zero_grad()
        loss_torch.backward(retain_graph=True)
        optimizer.step()

        # update results
        results['loss'].append(torch2np(loss_torch))
        results['design'].append(torch2np(design_torch))
        results['x_trj'].append(torch2np(x_trj_torch))
        results['u_trj'].append(u_trj)

    for key in results.keys():
        results[key] = np.array(results[key])
    
    return results


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='pendulum', choices=list(config.keys()))
    parser.add_argument('--num-iter', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--savefig', default=False, action='store_true')
    args = parser.parse_args()

    results = optimize_design(args.env, num_iter=args.num_iter, lr=args.lr)

    env_config, env_utils = config[args.env], utils[args.env]
    iteraions = np.arange(args.num_iter)

    # plot loss curve
    plt.plot(iteraions, results['loss'])
    plt.title(f'{args.env} loss')
    if args.savefig:
        plt.savefig(f'{args.env}_loss.png')
    plt.show()

    # plot design curve
    n_design_param = len(env_config['design_init'])
    design_name = env_config['design_name']
    fig, ax = plt.subplots(1, n_design_param, figsize=(n_design_param * 4, 3))
    for i in range(n_design_param):
        ax[i].plot(iteraions, results['design'][:, i])
        ax[i].set_title(design_name[i])
    plt.suptitle(f'{args.env} design')
    if args.savefig:
        plt.savefig(f'{args.env}_design.png')
    plt.show()

    # find the optimal design
    best_idx = np.argmin(results['loss'])
    best_design, best_x_trj, best_u_trj = results['design'][best_idx], results['x_trj'][best_idx], results['u_trj'][best_idx]
    print(f'best design: {best_design}')

    # animate the optimal design
    animate_func = env_utils['animate']
    animate_func(best_design, best_x_trj, env_config['N'])

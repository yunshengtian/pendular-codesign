import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from sim import TorchAcrobotSim
from env.cost import AcrobotCost
import env
from control import ILQR, MPPI


def optimize_design(num_iter, lr, plot=False, plot_interval=10):
    
    # constant
    g = 9.81
    dt = .1
    N = 50
    x_init = torch.tensor([np.pi * 0.95, 0., 0., 0.])
    x_target = torch.tensor([np.pi, 0., 0., 0.])
    x_init_np = x_init.numpy()
    x_target_np = x_target.numpy()
    
    # design
    design = torch.tensor([1., 2., 1., 2.], requires_grad=True)
    optimizer = torch.optim.Adam([design], lr=lr)

    all_loss = []
    all_design = []
    all_x_trj = []
    
    if plot: 
        fig, ax = plt.subplots()
        ax.set_title('Loss')
    
    for i in tqdm(range(num_iter)):

        design_np = design.clone().detach().numpy()
        env = gym.make('acrobot-v0', design=design_np, x_init=x_init_np, x_target=x_target_np, N=N, dt=dt)
        ilqr = ILQR(env)
        _, u_trj, _ = ilqr.solve(x_init_np, N, max_iter=50)
        u_trj = torch.tensor(u_trj, dtype=torch.float32)

        x_trj = TorchAcrobotSim(design, g, dt).rollout(x_init, u_trj)
        loss = AcrobotCost(x_target).cost_rollout(x_trj, u_trj[:, None])
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        all_loss.append(loss.clone().detach().numpy())
        all_design.append(design.clone().detach().numpy())
        all_x_trj.append(x_trj.clone().detach().numpy())
        
        if plot and i % plot_interval == 0:
            ax.plot(np.arange(i + 1), all_loss, c='C0')

    all_loss, all_design, all_x_trj = np.array(all_loss), np.array(all_design), np.array(all_x_trj)
        
    return all_loss, all_design, all_x_trj


if __name__ == '__main__':

    all_loss, all_design, all_x_trj = optimize_design(num_iter=50, lr=1e-3)

    iteration = list(range(len(all_loss)))

    plt.plot(iteration, all_loss)
    plt.title('Loss')
    plt.show()

    fig, ax = plt.subplots(2, 2)
    ax[0][0].plot(iteration, all_design[:, 0])
    ax[0][1].plot(iteration, all_design[:, 1])
    ax[1][0].plot(iteration, all_design[:, 2])
    ax[1][1].plot(iteration, all_design[:, 3])
    plt.title('Design')
    plt.show()

    from animate import animate
    animate(all_design[-1], all_x_trj[-1], 50)
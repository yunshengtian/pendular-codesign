import gym
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import env
from env.utils import utils
from control import get_control


parser = ArgumentParser()
parser.add_argument('--env', type=str, default='pendulum', choices=['acrobot', 'pendulum'])
parser.add_argument('--control', type=str, default='ilqr', choices=['ilqr', 'mppi'])
parser.add_argument('--savefig', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

np.random.seed(args.seed)

env = gym.make(f'{args.env}-v0')

Control = get_control(args.control)
control = Control(env)

x_trj, u_trj, info = control.solve()


if args.control == 'ilqr':

    cost_trace = info['cost_trace']
    plt.plot(list(range(len(cost_trace))), cost_trace)
    plt.title('cost')
    plt.xlabel('iteration')
    if args.savefig:
        plt.savefig(f'{args.env}_ilqr_test.png')
    plt.show()

    final_cost = cost_trace[-1]

elif args.control == 'mppi':

    final_cost = info['cost']

print(f'Final cost: {final_cost}')


design = env.sim.get_design_params(env.sim.design)

Animation = utils[args.env]['animate']
animation = Animation()
animation.show(design, x_trj)

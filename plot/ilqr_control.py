import gym
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import env
from control import get_control


parser = ArgumentParser()
parser.add_argument('--env', type=str, default='pendulum', choices=['acrobot', 'pendulum'])
parser.add_argument('--savefig', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

np.random.seed(args.seed)

env = gym.make(f'{args.env}-v0')

Control = get_control('ilqr')
control = Control(env)

x_trj, u_trj, info = control.solve()

cost_trace = info['cost_trace']
final_cost = cost_trace[-1]

plt.plot(list(range(len(cost_trace))), cost_trace)
plt.title(f'{args.env} cost (final: %.2f)' % final_cost)
plt.xlabel('iteration')
plt.tight_layout()
if args.savefig:
    plt.savefig(f'{args.env}_ilqr_control.png')
plt.show()


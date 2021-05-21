import gym
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from multiprocessing import Pool

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import env
from control import get_control


parser = ArgumentParser()
parser.add_argument('--env', type=str, default='pendulum', choices=['acrobot', 'pendulum'])
parser.add_argument('--num-repeat', type=int, default=1)
parser.add_argument('--num-cores', type=int, default=1)
parser.add_argument('--savefig', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

np.random.seed(args.seed)

all_cost = []

def get_cost(i):

    np.random.seed(i)

    env = gym.make(f'{args.env}-v0')

    Control = get_control('mppi')
    control = Control(env)

    x_trj, u_trj, info = control.solve()

    final_cost = info['cost']

    return final_cost

if args.num_cores > 1:
    with Pool(args.num_cores) as pool:
        all_cost[:] = pool.map(get_cost, [i for i in range(args.num_repeat)])
else:
    for i in range(args.num_repeat):
        all_cost.append(get_cost(i))

plt.hist(all_cost, bins=20)
plt.title(f'{args.env} cost distribution (mean: %.2f, std: %.2f)' % (np.mean(all_cost), np.std(all_cost)))
plt.xlabel('cost')
if args.savefig:
    plt.savefig(f'{args.env}_mppi_control.png')
plt.show()

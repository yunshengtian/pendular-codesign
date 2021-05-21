from argparse import ArgumentParser
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from env.utils import utils
from tools import load_solution


parser = ArgumentParser()
parser.add_argument('--env', type=str)
parser.add_argument('--path', type=str)
parser.add_argument('--savegif', default=False, action='store_true')
parser.add_argument('--gifname', type=str, default=None)
parser.add_argument('--savefig', default=False, action='store_true')
parser.add_argument('--figname', type=str, default=None)
args = parser.parse_args()


# load solution
design, x_trj, u_trj = load_solution(args.path)


# show animation
env_utils = utils[args.env]
Animation = env_utils['animate']
animation = Animation()

design_str = '_'.join(['%.2f' % d for d in design])
gifname = f'{args.env}_animation_{design_str}.gif' if args.gifname is None else args.gifname
animation.show(design, x_trj, args.savegif, gifname)


# plot initial position
fig, ax = plt.subplots()
ax.set_xlim((-3, 3))
ax.set_ylim((-3, 3))
anim_func = animation.animate_func(ax, design, x_trj)
anim_func(0)
plt.title(f'{args.env} initial position')
plt.tight_layout()
if args.savefig:
    figname = f'{args.env}_initial_{design_str}.png' if args.figname is None else args.figname + '_initial.png'
    plt.savefig(figname)
plt.show()

# plot final position
fig, ax = plt.subplots()
ax.set_xlim((-3, 3))
ax.set_ylim((-3, 3))
anim_func = animation.animate_func(ax, design, x_trj)
anim_func(len(x_trj) - 1)
plt.title(f'{args.env} final position')
plt.tight_layout()
if args.savefig:
    figname = f'{args.env}_final_{design_str}.png' if args.figname is None else args.figname + '_final.png'
    plt.savefig(figname)
plt.show()

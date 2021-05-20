import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def animate_pendulum(design, x_trj, frames=50):

    fig, ax = plt.subplots()
    ax.set_xlim((-3, 3))
    ax.set_ylim((-3, 3))
    line, = ax.plot([], [], c='black')

    m, l = design

    def animate_frame(i):
        theta, thetadot = x_trj[i]
        start = np.array([0, 0])
        end = np.array([np.sin(theta), -np.cos(theta)]) * l
        line.set_data([start[0], end[0]], [start[1], end[1]])
        return (line,)

    interval = 3000 / frames
    anim = animation.FuncAnimation(fig, animate_frame, frames=frames, interval=interval)
    plt.show()


if __name__ == '__main__':
    
    design = [1, 2]
    x_trj = np.random.random((50, 2))
    animate_pendulum(design, x_trj)
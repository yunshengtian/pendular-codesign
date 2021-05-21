import numpy as np

from ..animate import Animation


class PendulumAnimation(Animation):

    def animate_func(self, ax, design, x_trj):

        ax.set_title('pendulum')

        line, = ax.plot([], [], c='black')
        m, l = design

        def animate_frame(i):
            theta, thetadot = x_trj[i]
            start = np.array([0, 0])
            end = np.array([np.sin(theta), -np.cos(theta)]) * l
            line.set_data([start[0], end[0]], [start[1], end[1]])
            return (line,)

        return animate_frame

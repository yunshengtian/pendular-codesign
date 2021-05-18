import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def animate(design, x_trj, frames=50):

    fig, ax = plt.subplots()
    ax.set_xlim((-3, 3))
    ax.set_ylim((-3, 3))
    line1, = ax.plot([], [], c='black')
    line2, = ax.plot([], [], c='blue')

    m1, m2, l1, l2 = design

    def animate_frame(i):
        theta1, theta2, theta1dot, theta2dot = x_trj[i]
        start1 = np.array([0, 0])
        end1 = np.array([np.sin(theta1), -np.cos(theta1)]) * l1
        line1.set_data([start1[0], end1[0]], [start1[1], end1[1]])
        start2 = end1
        end2 = start2 + np.array([np.sin(theta1 + theta2), -np.cos(theta1 + theta2)]) * l2
        line2.set_data([start2[0], end2[0]], [start2[1], end2[1]])
        return (line1, line2)

    interval = 3000 / frames
    anim = animation.FuncAnimation(fig, animate_frame, frames=frames, interval=interval)
    plt.show()


if __name__ == '__main__':
    
    design = [1, 2, 1, 2]
    x_trj = np.random.random((50, 4))
    animate(design, x_trj)
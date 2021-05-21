import matplotlib.pyplot as plt
from matplotlib import animation


class Animation:

    def animate_func(self, ax, design, x_trj):
        raise NotImplementedError

    def show(self, design, x_trj, savegif=False, gifname=None):

        fig, ax = plt.subplots()
        ax.set_xlim((-3, 3))
        ax.set_ylim((-3, 3))

        animate_frame = self.animate_func(ax, design, x_trj)

        frames = len(x_trj)
        interval = 3000 / frames
        anim = animation.FuncAnimation(fig, animate_frame, frames=frames, interval=interval)
        plt.tight_layout()
        if savegif:
            writergif = animation.PillowWriter(fps=30)
            gifname = 'animation.gif' if gifname is None else gifname
            anim.save(gifname, writer=writergif)
        plt.show()

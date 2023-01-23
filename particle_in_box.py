# Particle experiencing shear force
# Ryan Cocking 2023

"""
Animation of Elastic collisions with Gravity

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from omegaconf import OmegaConf

conf = OmegaConf.load("config.yml")


class ParticleBox:
    """Orbits class

    init_state is an [N x 4] array, where N is the number of particles:
       [[x1, y1, vx1, vy1],
        [x2, y2, vx2, vy2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """

    def __init__(self, init_state=[[1, 0, 0, -1]], bounds=[-1, 1, -1, 1], shear_rate=0):
        self.init_state = np.asarray(init_state, dtype=float)
        self.state = self.init_state.copy()
        self.bounds = np.asarray(bounds)
        self.size_x = bounds[1] - bounds[0]
        self.size_y = bounds[3] - bounds[2]
        self.shear_rate = shear_rate
        self.offset_x = 0

        if abs(self.shear_rate) > 0:
            self.boundary_func = self.boundary_lebc
        else:
            self.boundary_func = self.boundary_pbc

        # eight images, starting at top, going clockwise
        self.ghost_state = np.repeat(self.state[np.newaxis, :, :], 8, axis=0)
        self.ghost_bounds = np.repeat(self.bounds[np.newaxis, :], 8, axis=0)

        self.ghost_bounds[[2, 3, 4], :2] += self.size_x  # right
        self.ghost_bounds[[6, 7, 0], :2] -= self.size_x  # left
        self.ghost_bounds[[0, 1, 2], 2:] += self.size_y  # top
        self.ghost_bounds[[4, 5, 6], 2:] -= self.size_y  # bottom

    def update_ghosts(self):
        self.ghost_state[[2, 3, 4], :, 0] = self.state[:, 0] + self.size_x
        self.ghost_state[[2, 3, 4], :, 1] = self.state[:, 1]

        self.ghost_state[[6, 7, 0], :, 0] = self.state[:, 0] - self.size_x
        self.ghost_state[[6, 7, 0], :, 1] = self.state[:, 1]

        self.ghost_state[[0, 1, 2], :, 0] = self.state[:, 0]
        self.ghost_state[[0, 1, 2], :, 1] = self.state[:, 1] + self.size_y

        self.ghost_state[[4, 5, 6], :, 0] = self.state[:, 0]
        self.ghost_state[[4, 5, 6], :, 1] = self.state[:, 1] - self.size_y

    def boundary_pbc(self):
        # check for crossing boundary
        crossed_x1 = self.state[:, 0] < self.bounds[0]
        crossed_x2 = self.state[:, 0] > self.bounds[1]
        crossed_y1 = self.state[:, 1] < self.bounds[2]
        crossed_y2 = self.state[:, 1] > self.bounds[3]

        self.state[crossed_x1, 0] = self.bounds[1]
        self.state[crossed_x2, 0] = self.bounds[0]

        self.state[crossed_y1, 1] = self.bounds[3]
        self.state[crossed_y2, 1] = self.bounds[2]

    def boundary_lebc(self):
        crossed_x1 = self.state[:, 0] < self.bounds[0]
        crossed_x2 = self.state[:, 0] > self.bounds[1]
        crossed_y1 = self.state[:, 1] < self.bounds[2]
        crossed_y2 = self.state[:, 1] > self.bounds[3]

        self.state[crossed_x1, 0] = self.bounds[1]
        self.state[crossed_x2, 0] = self.bounds[0]

        self.state[crossed_y1, 0] += self.offset_x
        self.state[crossed_y1, 1] = self.bounds[3]
        self.state[crossed_y2, 0] -= self.offset_x
        self.state[crossed_y2, 1] = self.bounds[2]

    def set_lebc_offset(self, dt):
        L_y = self.bounds[3] - self.bounds[2]
        n = 1
        self.offset_x = self.shear_rate * dt * L_y
        while self.offset_x > L_y:
            n += 1
            self.offset_x = self.shear_rate * dt * L_y - n * L_y

    def step(self, dt):
        """step once by dt seconds"""

        # update positions
        self.state[:, :2] += self.state[:, 2:] * dt
        self.update_ghosts()

        # TODO position update in context of force sum

        # TODO shear force

        # TODO viscous damping force

        # find pairs of particles undergoing a collision
        # D = squareform(pdist(self.state[:, :2]))
        # ind1, ind2 = np.where(D < 2 * self.size)
        # unique = (ind1 < ind2)
        # ind1 = ind1[unique]
        # ind2 = ind2[unique]

        self.boundary_func()


# ------------------------------------------------------------
# set up initial state
np.random.seed(2)
init_state = -0.5 + np.random.random((conf["Npart"], 4))
init_state[:, :2] *= 3.9  # 30fps

print("Initialising particles...")
box = ParticleBox(init_state, shear_rate=conf["shear_rate"])
box.set_lebc_offset(conf["dt"])

print("Done")
# ------------------------------------------------------------
# set up figure and animation
print("Setting up plots...")
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(
    111, aspect="equal", autoscale_on=False, xlim=(-3.2, 3.2), ylim=(-2.4, 2.4)
)

# particles holds the locations of the particles
(particles,) = ax.plot([], [], "bo", ms=6)

# images of the periodic copies
(images,) = ax.plot([], [], "co", ms=6)

rect = plt.Rectangle(
    box.bounds[::2],
    box.size_x,
    box.size_y,
    ec="r",
    lw=2,
    fc="none",
)

ghost_rect = [
    plt.Rectangle(
        gb[::2],
        box.size_x,
        box.size_y,
        ec="k",
        lw=2,
        fc="none",
        ls="-",
    )
    for gb in box.ghost_bounds
]

ax.add_patch(rect)
for item in ghost_rect:
    ax.add_patch(item)

print("Done")


def init():
    """initialize animation"""
    # global box, rect
    particles.set_data([], [])
    images.set_data([], [])
    rect.set_edgecolor("r")
    # ghost_rect.set_edgecolor('none')
    return (particles, rect, images, *ghost_rect)


def animate(i):
    """perform animation step"""
    global box, rect, ghost_rect, ax, fig
    print(f"Step = {i}", end="\r")
    box.step(conf["dt"])

    ms = int(
        fig.dpi
        * 2
        * conf["part_radius"]
        * fig.get_figwidth()
        / np.diff(ax.get_xbound())[0]
    )

    # update pieces of the animation
    rect.set_edgecolor("r")
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(ms)
    images.set_data(box.ghost_state[:, :, 0], box.ghost_state[:, :, 1])
    images.set_markersize(ms)
    return (particles, images, *ghost_rect, rect)


ani = animation.FuncAnimation(
    fig, animate, frames=600, interval=10, blit=True, init_func=init, save_count=1500
)


# saving as mp4 requires ffmpeg or mencoder to be installed. for more
# information, see http://matplotlib.sourceforge.net/api/animation_api.html
# ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

# writervideo = animation.FFMpegWriter(fps=60)
# ani.save("out.mp4", writer=writervideo)

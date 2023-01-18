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
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

class ParticleBox:
    """Orbits class

    init_state is an [N x 4] array, where N is the number of particles:
       [[x1, y1, vx1, vy1],
        [x2, y2, vx2, vy2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """
    def __init__(self,
                 init_state = [[1, 0, 0, -1],
                               [-0.5, 0.5, 0.5, 0.5],
                               [-0.5, -0.5, -0.5, 0.5]],
                 bounds = [-1, 1, -1, 1],
                 size = 0.04,
                 shear_rate = 0):
        self.init_state = np.asarray(init_state, dtype=float)
        self.size = size
        self.state = self.init_state.copy()
        self.bounds = bounds
        self.shear_rate = shear_rate
        self.offset_x = 0

        if abs(self.shear_rate) > 0:
            self.boundary_func = self.boundary_lebc
        else:
            self.boundary_func = self.boundary_pbc

    def boundary_pbc(self):
        # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0])
        crossed_x2 = (self.state[:, 0] > self.bounds[1])
        crossed_y1 = (self.state[:, 1] < self.bounds[2])
        crossed_y2 = (self.state[:, 1] > self.bounds[3])

        self.state[crossed_x1, 0] = self.bounds[1]
        self.state[crossed_x2, 0] = self.bounds[0]

        self.state[crossed_y1, 1] = self.bounds[3]
        self.state[crossed_y2, 1] = self.bounds[2]

    def boundary_lebc(self):
        crossed_x1 = (self.state[:, 0] < self.bounds[0])
        crossed_x2 = (self.state[:, 0] > self.bounds[1])
        crossed_y1 = (self.state[:, 1] < self.bounds[2])
        crossed_y2 = (self.state[:, 1] > self.bounds[3])

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
        self.state[:, :2] += dt * self.state[:, 2:]

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


#------------------------------------------------------------
# set up initial state
np.random.seed(2)
Npart = 3
init_state = -0.5 + np.random.random((Npart, 4))
init_state[:, :2] *= 3.9
dt = 1. / 30 # 30fps

box = ParticleBox(init_state, size=0.04, shear_rate=0.0)
box.set_lebc_offset(dt)
# else:
#     box = ParticleBox(init_state, size=0.04)

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-3.2, 3.2), ylim=(-2.4, 2.4))

# particles holds the locations of the particles
particles, = ax.plot([], [], 'bo', ms=6)

# images of the periodic copies
images, = ax.plot([], [], 'co', ms=6)

# rect is the box edge
offset_x = box.bounds[1] - box.bounds[0]
offset_y = box.bounds[3] - box.bounds[2]

rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='r', lw=2, fc='none')

image_rect = [
    plt.Rectangle(
        [box.bounds[0] + box.offset_x, box.bounds[2] + offset_y],
        box.bounds[1] - box.bounds[0],
        box.bounds[3] - box.bounds[2],
        ec='k', lw=2, fc='none', ls='-'),
    plt.Rectangle(
        [box.bounds[0] + offset_x + box.offset_x, box.bounds[2] + offset_y],
        box.bounds[1] - box.bounds[0],
        box.bounds[3] - box.bounds[2],
        ec='k', lw=2, fc='none', ls='-'),
    plt.Rectangle(
        [box.bounds[0] + offset_x, box.bounds[2]],
        box.bounds[1] - box.bounds[0],
        box.bounds[3] - box.bounds[2],
        ec='k', lw=2, fc='none', ls='-'),
    plt.Rectangle(
        [box.bounds[0] + offset_x - box.offset_x, box.bounds[2] - offset_y],
        box.bounds[1] - box.bounds[0],
        box.bounds[3] - box.bounds[2],
        ec='k', lw=2, fc='none', ls='-'),
    plt.Rectangle(
        [box.bounds[0] - box.offset_x, box.bounds[2] - offset_y],
        box.bounds[1] - box.bounds[0],
        box.bounds[3] - box.bounds[2],
        ec='k', lw=2, fc='none', ls='-'),
    plt.Rectangle(
        [box.bounds[0] - offset_x - box.offset_x, box.bounds[2] - offset_y],
        box.bounds[1] - box.bounds[0],
        box.bounds[3] - box.bounds[2],
        ec='k', lw=2, fc='none', ls='-'),
    plt.Rectangle(
        [box.bounds[0] - offset_x, box.bounds[2]],
        box.bounds[1] - box.bounds[0],
        box.bounds[3] - box.bounds[2],
        ec='k', lw=2, fc='none', ls='-'),
    plt.Rectangle(
        [box.bounds[0] - offset_x + box.offset_x, box.bounds[2] + offset_y],
        box.bounds[1] - box.bounds[0],
        box.bounds[3] - box.bounds[2],
        ec='k', lw=2, fc='none', ls='-')
]

ax.add_patch(rect)
for item in image_rect:
    ax.add_patch(item)

def init():
    """initialize animation"""
    global box, rect
    particles.set_data([], [])
    images.set_data([], [])
    rect.set_edgecolor('r')
    # image_rect.set_edgecolor('none')
    return particles, rect, images, (*image_rect)

def animate(i):
    """perform animation step"""
    global box, rect, image_rect, dt, ax, fig
    box.step(dt)

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])

    # draw eight periodic copies of particles (clockwise from top)
    draw_state_x = np.append(box.state[:, 0] + box.offset_x,
    [
        box.state[:, 0] + offset_x + box.offset_x,
        box.state[:, 0] + offset_x,
        box.state[:, 0] + offset_x - box.offset_x,
        box.state[:, 0] - box.offset_x,
        box.state[:, 0] - offset_x - box.offset_x,
        box.state[:, 0] - offset_x,
        box.state[:, 0] - offset_x + box.offset_x
    ])
    draw_state_y = np.append(box.state[:, 1] + offset_y,
    [
        box.state[:, 1] + offset_y,
        box.state[:, 1],
        box.state[:, 1] - offset_y,
        box.state[:, 1] - offset_y,
        box.state[:, 1] - offset_y,
        box.state[:, 1],
        box.state[:, 1] + offset_y
    ])

    # update pieces of the animation
    rect.set_edgecolor('r')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(ms)
    images.set_data(draw_state_x, draw_state_y)
    images.set_markersize(ms)
    return particles, images, (*image_rect), rect

ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=10, blit=True, init_func=init, save_count=1500)


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
# ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

# writervideo = animation.FFMpegWriter(fps=60)
# ani.save("out.mp4", writer=writervideo)

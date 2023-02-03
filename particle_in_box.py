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

from time import time
from omegaconf import OmegaConf
from pathlib import Path

conf = OmegaConf.load("config.yml")
conf["mu"] = 6.0 * np.pi * conf["eta"] * conf["part_radius"]  # [kg s^-1]
conf["D"] = conf["KT"] / conf["mu"]  # [m^2 s^1]
conf["box_size_x"] = 50 * conf["part_radius"]
conf["box_size_y"] = conf["box_size_x"]

p = Path(conf["traj_file"])
if p.is_file():
    p.unlink()


class ParticleBox:
    """Orbits class

    init_state is an [N x 4] array, where N is the number of particles:
       [[x1, y1, vx1, vy1],
        [x2, y2, vx2, vy2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """

    def __init__(
        self,
        init_state=[[1, 0, 0, -1]],
        bounds=[-1, 1, -1, 1],
        shear_rate=0,
        radius=0.05,
    ):
        self.init_state = np.asarray(init_state, dtype=float)
        self.state = self.init_state.copy()
        self.step_count = 0
        self.bounds = np.asarray(bounds)
        self.part_radius = radius
        self.size_x = bounds[1] - bounds[0]
        self.size_y = bounds[3] - bounds[2]
        self.shear_rate = shear_rate
        self.lebc_image_velocity_x = conf["shear_rate"] * self.size_y
        self.lebc_image_offset_x = conf["shear_rate"] * self.size_y * conf["dt"]
        self.x = 0

        self.ghost_pos = np.repeat(self.state[np.newaxis, :, :2], 8, axis=0)
        self.ghost_bounds = np.repeat(self.bounds[np.newaxis, :], 8, axis=0)

        # A better idea might be to have an origin vector that represents the
        # bottom left corner of each box, then we just store one
        # set of coordinates and transform it eight times [JM Haile, p82].
        self.ghost_bounds[[2, 3, 4], :2] += self.size_x  # right
        self.ghost_bounds[[6, 7, 0], :2] -= self.size_x  # left
        self.ghost_bounds[[0, 1, 2], 2:] += self.size_y  # top
        self.ghost_bounds[[4, 5, 6], 2:] -= self.size_y  # bottom

        self.ghost_pos[[2, 3, 4], :, 0] += self.size_x
        self.ghost_pos[[6, 7, 0], :, 0] -= self.size_x
        self.ghost_pos[[0, 1, 2], :, 1] += self.size_y
        self.ghost_pos[[4, 5, 6], :, 1] -= self.size_y

    def update_ghosts(self):
        self.ghost_pos = np.repeat(self.state[np.newaxis, :, :2], 8, axis=0)

        # move particles into ghost cells, shifting by cell midpoints
        self.ghost_pos[:, :, 0] += np.mean(self.ghost_bounds[:, np.newaxis, :2], axis=2)
        self.ghost_pos[:, :, 1] += np.mean(self.ghost_bounds[:, np.newaxis, 2:], axis=2)

        self.ghost_bounds[[0, 1, 2], :2] += self.lebc_image_offset_x
        self.ghost_bounds[[4, 5, 6], :2] -= self.lebc_image_offset_x

        # cell wrapping
        if self.ghost_bounds[2, 0] > 2 * self.bounds[1]:
            self.ghost_bounds[[0, 1, 2], :2] -= self.size_x

        if self.ghost_bounds[6, 1] < 2 * self.bounds[0]:
            self.ghost_bounds[[4, 5, 6], :2] += self.size_x

    def crossed_boundary(self):
        # check for crossing boundary
        crossed_x1 = self.state[:, 0] < self.bounds[0]  # left
        crossed_x2 = self.state[:, 0] > self.bounds[1]  # right
        crossed_y1 = self.state[:, 1] < self.bounds[2]  # bottom
        crossed_y2 = self.state[:, 1] > self.bounds[3]  # top

        self.state[crossed_x1, 0] = self.bounds[1]
        self.state[crossed_x2, 0] = self.bounds[0]
        self.state[crossed_y1, 1] = self.bounds[3]
        self.state[crossed_y2, 1] = self.bounds[2]

        # crossing the upper or lower boundary requires an offset in x
        # e.g. if a particle passes thrtough the upper boundary, its image should
        # be shifted in the positive x direction.
        # however, relative to the central box, this particle will appear
        # at the bottom, having shifted in the negative x direction
        self.state[crossed_y1, 0] += self.lebc_image_offset_x
        self.state[crossed_y2, 0] -= self.lebc_image_offset_x

    def lebc_offset(self, x):
        if x >= box.size_x / 2:
            x -= box.size_x
        else:
            x += self.lebc_image_velocity_x * conf["dt"]

        self.x = x

    def thermal_noise(self, thermal_energy, drag_coef, timestep):
        """From fluctuation-dissipation theorem."""
        mag = np.sqrt(24 * thermal_energy * drag_coef / timestep)
        return mag * (np.random.random(self.state[:, :2].shape) - 0.5)

    def shear_force(self, drag_coef):
        """y coordinate should be in range 0 < y < L. See Ridley p51, Bindgen et al, Lees & Edwards"""
        force = (
            self.shear_rate
            * box.size_y
            * drag_coef
            * ((self.state[:, 1] + 0.5 * box.size_y) / box.size_y - 0.5)
        )
        return np.column_stack((force, np.zeros(self.state.shape[0])))

    def step(self):
        if conf["enable_noise"]:
            n = self.thermal_noise(conf["KT"], conf["mu"], conf["dt"])
        else:
            n = 0

        if conf["test_mode"]:
            v_test = np.column_stack(
                (np.zeros(self.state.shape[0]), np.ones(self.state.shape[0]))
            )

        else:
            v_test = 0

        s = self.shear_force(conf["mu"])
        r_new = (
            self.state[:, :2]
            + (conf["dt"] / conf["mu"]) * (n + s)
            + v_test * (conf["dt"])
        )
        self.state[:, 2:] = (r_new - self.state[:, :2]) / conf["dt"]
        self.state[:, :2] = r_new

        self.lebc_offset((self.ghost_bounds[1, 0] + self.ghost_bounds[1, 1]) / 2)
        self.crossed_boundary()
        self.update_ghosts()
        self.write_traj()
        self.step_count += 1

    def write_traj(self):
        if self.step_count <= conf["Nsteps"]:
            with open(conf["traj_file"], "a") as f:
                for i in range(conf["Npart"]):
                    m1 = (self.bounds[0] + self.bounds[1]) / 2
                    m2 = (self.ghost_bounds[1, 0] + self.ghost_bounds[1, 1]) / 2
                    traj_string = f"{self.state[i, 0]:e},{self.state[i, 1]:e},{self.state[i, 2]:e},{self.state[i, 3]:e},{self.x},{m2-m1:e}\n"
                    f.write(traj_string)


# ------------------------------------------------------------
# set up initial state

np.random.seed(int(time()))
init_state = np.zeros((conf["Npart"], 4))
init_state[:, :2] = (np.random.random((conf["Npart"], 2)) - 0.5) * min(
    conf["box_size_x"], conf["box_size_y"]
)
init_state[:, 2:] = (
    np.random.random((conf["Npart"], 2)) * conf["part_radius"] / conf["dt"]
)

print("Initialising particles...")
box = ParticleBox(
    init_state,
    radius=conf["part_radius"],
    shear_rate=conf["shear_rate"],
    bounds=[
        -0.5 * conf["box_size_x"],
        0.5 * conf["box_size_x"],
        -0.5 * conf["box_size_y"],
        0.5 * conf["box_size_y"],
    ],
)

print(f"LEBC image velocity = {box.lebc_image_velocity_x:.2e} m")

print("Done")
# ------------------------------------------------------------
# set up figure and animation
print("Setting up plots...")
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(
    111,
    aspect="equal",
    autoscale_on=False,
    xlim=(-2.3 * conf["box_size_x"], 2.3 * conf["box_size_x"]),
    ylim=(-1.5 * conf["box_size_y"], 1.5 * conf["box_size_y"]),
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
    for gb in box.ghost_bounds[[1, 5]]
]

ax.add_patch(rect)
for item in ghost_rect:
    ax.add_patch(item)

print("Done")


def init():
    """initialize animation"""
    particles.set_data([], [])
    images.set_data([], [])
    rect.set_edgecolor("r")
    for item in ghost_rect:
        item.set_edgecolor("k")
    return (particles, rect, images, *ghost_rect)


def animate(i):
    """perform animation step"""
    global box, rect, ghost_rect, ax, fig
    print(f"Step = {i}", end="\r")
    box.step()

    ms = int(
        fig.dpi * 2 * box.part_radius * fig.get_figwidth() / np.diff(ax.get_xbound())[0]
    )

    # update pieces of the animation
    rect.set_edgecolor("r")
    for item in ghost_rect:
        item.set_edgecolor("k")

    for gr, gb in zip(ghost_rect, box.ghost_bounds[[1, 5]]):
        gr.xy = gb[::2]

    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(ms)
    images.set_data(box.ghost_pos[[1, 5], :, 0], box.ghost_pos[[1, 5], :, 1])
    images.set_markersize(ms)

    return (particles, images, *ghost_rect, rect)


ani = animation.FuncAnimation(
    fig,
    animate,
    frames=conf["Nsteps"],
    interval=10,
    blit=True,
    init_func=init,
    save_count=1500,
)

if conf["save_video"]:
    print("Video duration: {:.2f} s".format(conf["Nsteps"] / conf["fps"]))
    # saving as mp4 requires ffmpeg or mencoder to be installed. for more
    # information, see http://matplotlib.sourceforge.net/api/animation_api.html
    # ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    # plt.show()
    writervideo = animation.FFMpegWriter(fps=conf["fps"])
    ani.save("out.mp4", writer=writervideo)

else:
    # system evolves indefinitely, trajectory only written up to Nsteps
    plt.show()

plt.close()
traj = np.loadtxt(conf["traj_file"], delimiter=",", dtype=np.float64)
x = traj[:, 0]
y = traj[:, 1]
vx = traj[:, 2]
xle = traj[:, -2]
m = traj[:, -1]
for i in range(conf["Npart"]):
    vxi = vx[i :: conf["Npart"]]
    yi = y[i :: conf["Npart"]]
    xlei = xle[i :: conf["Npart"]]
    mi = m[i :: conf["Npart"]]
    # print(f"<y{i:d}> = {np.mean(yi) - yi[0]:.2e}, <vx{i:d}> = {np.mean(vxi):.2e}")
    plt.plot(vxi, yi, "o", ms=2)

plt.plot([0, 0], [-0.5 * conf["box_size_y"], 0.5 * conf["box_size_y"]], "k:")
plt.plot([-np.max(vx), np.max(vx)], [0, 0], "k:")
plt.title("Instantaneous velocity profile")
plt.xlabel("x velocity [m/s]")
plt.ylabel("y position [m]")
plt.xlim(-np.max(vx), np.max(vx))
plt.ylim(-0.5 * conf["box_size_y"], 0.5 * conf["box_size_y"])
plt.show()
plt.close()

for i in range(conf["Npart"]):
    xi = x[i :: conf["Npart"]]
    yi = y[i :: conf["Npart"]]
    plt.plot(xi, yi, "o", ms=2)

plt.plot([0, 0], [-0.5 * conf["box_size_y"], 0.5 * conf["box_size_y"]], "k:")
plt.plot([-0.5 * conf["box_size_x"], 0.5 * conf["box_size_x"]], [0, 0], "k:")
plt.title("Trajectory")
plt.xlabel("x position [m]")
plt.ylabel("y position [m]")
plt.xlim(-0.5 * conf["box_size_x"], 0.5 * conf["box_size_x"])
plt.ylim(-0.5 * conf["box_size_y"], 0.5 * conf["box_size_y"])
plt.show()
plt.close()

steps = np.arange(xlei.size)
t = conf["dt"] * steps
plt.title("LEBC offset")
plt.ylabel("$x_{LE}$ [m]")
plt.xlabel("Steps")
plt.plot(steps, mi, "r-", label="Expected", alpha=0.5, lw=6)
plt.plot(steps, xlei, "g-", label="Calculated", lw=2)
plt.plot([0, steps[-1]], [box.bounds[1], box.bounds[1]], "k:", lw=1)
plt.plot([0, steps[-1]], [box.bounds[0], box.bounds[0]], "k:", lw=1)
plt.plot(
    [
        box.bounds[1] / (box.lebc_image_velocity_x * conf["dt"]),
        box.bounds[1] / (box.lebc_image_velocity_x * conf["dt"]),
    ],
    [box.bounds[0] * 1.2, box.bounds[1] * 1.2],
    "k:",
    lw=1,
)
plt.legend()
plt.show()

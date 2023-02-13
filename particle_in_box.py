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
conf["box_dim"] = 50 * conf["part_radius"]

p = Path(conf["traj_file"])
if p.is_file():
    p.unlink()


def point_within_rectangle(
    m=np.array([0.5, 0.5]), rect=np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])
):
    """https://math.stackexchange.com/questions/190111/how-to-check-if-a-point-is-inside-a-rectangle"""

    if rect.shape == (4, 2):
        # rect passed as corners
        a, b, _, d = rect
    elif rect.shape == (4,):
        # rect passed as box bounds
        xmin, xmax, ymin, ymax = rect
        a = np.array([xmin, ymax])
        b = np.array([xmax, ymax])
        d = np.array([xmin, ymin])
    else:
        raise Exception(f"rect has invalid shape: {rect.shape}")

    am = m - a
    ab = b - a
    ad = d - a
    amab = np.dot(am, ab)
    abab = np.dot(ab, ab)
    amad = np.dot(am, ad)
    adad = np.dot(ad, ad)

    return (0 < amab < abab) and (0 < amad < adad)


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
        self.dim = bounds[1] - bounds[0]
        self.shear_rate = shear_rate
        self.lebc_image_velocity_x = conf["shear_rate"] * self.dim
        self.lebc_image_offset_x = 0
        self.image_matrix = [
            [-1, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
            [-1, -1],
            [-1, 0],
            [0, 0],
        ]

        self.ghost_pos = np.repeat(self.state[np.newaxis, :, :2], 8, axis=0)
        self.ghost_bounds = np.repeat(self.bounds[np.newaxis, :], 8, axis=0)

        # A better idea might be to have an origin vector that represents the
        # bottom left corner of each box, then we just store one
        # set of coordinates and transform it eight times [JM Haile, p82].
        for i, img in enumerate(self.image_matrix[:-1]):
            self.ghost_bounds[i, :2] += img[0] * self.dim
            self.ghost_bounds[i, 2:] += img[1] * self.dim
            self.ghost_pos[i, :, 0] += img[0] * self.dim
            self.ghost_pos[i, :, 1] += img[1] * self.dim

    def image_query(self, part_ind):
        """Return the image occupied by a particle of a given index"""
        xy = self.state[part_ind, :2]

        if point_within_rectangle(xy, self.bounds):
            return self.image_matrix[-1]
        else:
            ind = np.zeros(self.ghost_bounds.size, dtype=bool)
            for i, gb in enumerate(self.ghost_bounds):
                ind[i] = point_within_rectangle(xy, gb)

            if sum(ind) > 1:
                print(self.image_matrix[np.where(ind)])
                raise Exception("Particle occupied two images simultaneously")
            elif sum(ind) == 0:
                return None

            i = ind.nonzero()[0][0]
            print(f"Particle entered bounds of image {self.image_matrix[i]}")
            return self.image_matrix[i]

    def update_image_particles(self):
        self.ghost_pos = np.repeat(self.state[np.newaxis, :, :2], 8, axis=0)

        # move particles into cells, shifting by cell midpoints
        self.ghost_pos[:, :, 0] += np.mean(self.ghost_bounds[:, np.newaxis, :2], axis=2)
        self.ghost_pos[:, :, 1] += np.mean(self.ghost_bounds[:, np.newaxis, 2:], axis=2)

    def update_image_bounds(self):
        for i, img in enumerate(self.image_matrix[:-1]):
            self.ghost_bounds[i, 0] = (
                self.bounds[0] + self.lebc_image_offset_x * img[1] + self.dim * img[0]
            )
            self.ghost_bounds[i, 1] = (
                self.bounds[1] + self.lebc_image_offset_x * img[1] + self.dim * img[0]
            )
            self.ghost_bounds[i, 2] = self.bounds[2] + self.dim * img[1]
            self.ghost_bounds[i, 3] = self.bounds[3] + self.dim * img[1]

    def set_lebc_offset(self):
        # shouldn't LEBC offset happen regardless of shifting back by box length?
        if self.lebc_image_offset_x >= self.bounds[1]:
            self.lebc_image_offset_x = self.bounds[0]

        self.lebc_image_offset_x += self.lebc_image_velocity_x * conf["dt"]

    def crossed_boundary(self):
        # check for crossing boundary
        crossed_xmin = self.state[:, 0] < self.bounds[0]  # left
        crossed_xmax = self.state[:, 0] > self.bounds[1]  # right
        crossed_ymin = self.state[:, 1] < self.bounds[2]  # bottom
        crossed_ymax = self.state[:, 1] > self.bounds[3]  # top

        # simultaneous bounds check
        for i in range(self.state.shape[0]):
            if crossed_xmin[i] and crossed_xmax[i]:
                raise Exception(
                    f"Particle {i} crossed left and right (x) boundaries simultaneously."
                )

            elif crossed_ymin[i] and crossed_ymax[i]:
                raise Exception(
                    f"Particle {i} crossed upper and lower (y) boundaries simultaneously."
                )

        self.state[crossed_ymin, 0] = (
            np.mod(
                self.state[crossed_ymin, 0] + self.lebc_image_offset_x + 0.5 * self.dim,
                self.dim,
            )
            - 0.5 * self.dim
        )

        self.state[crossed_ymax, 0] = (
            np.mod(
                self.state[crossed_ymax, 0] - self.lebc_image_offset_x + 0.5 * self.dim,
                self.dim,
            )
            - 0.5 * self.dim
        )

        self.state[crossed_ymin, 1] += self.dim
        self.state[crossed_ymax, 1] -= self.dim

        self.state[crossed_xmin, 0] += self.dim
        self.state[crossed_xmax, 0] -= self.dim

    def thermal_noise(self, thermal_energy, drag_coef, timestep):
        """From fluctuation-dissipation theorem."""
        mag = np.sqrt(24 * thermal_energy * drag_coef / timestep)
        return mag * (np.random.random(self.state[:, :2].shape) - 0.5)

    def shear_force(self, drag_coef):
        """y coordinate should be in range 0 < y < L. See Ridley p51, Bindgen et al, Lees & Edwards"""
        force = (
            self.shear_rate
            * self.dim
            * drag_coef
            * ((self.state[:, 1] + 0.5 * box.dim) / box.dim - 0.5)
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

        self.set_lebc_offset()
        self.crossed_boundary()
        self.update_image_particles()
        self.update_image_bounds()
        self.write_traj()
        self.step_count += 1

    def write_traj(self):
        if self.step_count <= conf["Nsteps"]:
            with open(conf["traj_file"], "a") as f:
                for i in range(conf["Npart"]):
                    m1 = (self.bounds[0] + self.bounds[1]) / 2
                    m2 = (self.ghost_bounds[1, 0] + self.ghost_bounds[1, 1]) / 2
                    traj_string = f"{self.state[i, 0]:e},{self.state[i, 1]:e},{self.state[i, 2]:e},{self.state[i, 3]:e},{self.lebc_image_offset_x},{m2-m1:e}\n"
                    f.write(traj_string)


# ------------------------------------------------------------
# set up initial state

np.random.seed(int(time()))
init_state = np.zeros((conf["Npart"], 4))
init_state[:, :2] = (np.random.random((conf["Npart"], 2)) - 0.5) * min(
    conf["box_dim"], conf["box_dim"]
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
        -0.5 * conf["box_dim"],
        0.5 * conf["box_dim"],
        -0.5 * conf["box_dim"],
        0.5 * conf["box_dim"],
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
    xlim=(-2.3 * conf["box_dim"], 2.3 * conf["box_dim"]),
    ylim=(-1.5 * conf["box_dim"], 1.5 * conf["box_dim"]),
)

# particles holds the locations of the particles
(particles,) = ax.plot([], [], "bo", ms=6)

# images of the periodic copies
(images,) = ax.plot([], [], "co", ms=6)

if conf["draw_all"]:
    ind = list(range(8))
else:
    ind = [1, 5]

rect = plt.Rectangle(
    box.bounds[::2],
    box.dim,
    box.dim,
    ec="r",
    lw=2,
    fc="none",
)

ghost_rect = [
    plt.Rectangle(
        gb[::2],
        box.dim,
        box.dim,
        ec="k",
        lw=2,
        fc="none",
        ls="-",
    )
    for gb in box.ghost_bounds[ind]
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

    for gr, gb in zip(ghost_rect, box.ghost_bounds[ind]):
        gr.xy = gb[::2]

    plt.title(f"$x_{{LE}}$ = {box.lebc_image_offset_x:.3e}")

    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(ms)
    images.set_data(box.ghost_pos[ind, :, 0], box.ghost_pos[ind, :, 1])
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
    # place in loop to colour plot by particles
    vxi = vx[i :: conf["Npart"]]
    yi = y[i :: conf["Npart"]]
    xlei = xle[i :: conf["Npart"]]
    mi = m[i :: conf["Npart"]]
    # print(f"<y{i:d}> = {np.mean(yi) - yi[0]:.2e}, <vx{i:d}> = {np.mean(vxi):.2e}")
    plt.plot(vxi, yi, "o", ms=2)

plt.title("Instantaneous velocity")
plt.plot([0, 0], [-0.5 * conf["box_dim"], 0.5 * conf["box_dim"]], "k:")
plt.plot([-np.max(vx), np.max(vx)], [0, 0], "k:")
plt.xlabel("x velocity [m/s]")
plt.ylabel("y position [m]")
plt.xlim(-np.max(vx), np.max(vx))
plt.ylim(-0.5 * conf["box_dim"], 0.5 * conf["box_dim"])
plt.savefig("instantaneous_velocity.png", dpi=400)
plt.show()
plt.close()

plt.title("Trajectory")
for i in range(conf["Npart"]):
    xi = x[i :: conf["Npart"]]
    yi = y[i :: conf["Npart"]]
    plt.plot(xi, yi, "o", ms=2)

plt.plot([0, 0], [-0.5 * conf["box_dim"], 0.5 * conf["box_dim"]], "k:")
plt.plot([-0.5 * conf["box_dim"], 0.5 * conf["box_dim"]], [0, 0], "k:")
plt.xlabel("x position [m]")
plt.ylabel("y position [m]")
plt.xlim(-0.5 * conf["box_dim"], 0.5 * conf["box_dim"])
plt.ylim(-0.5 * conf["box_dim"], 0.5 * conf["box_dim"])
plt.savefig("trajectory.png", dpi=400)
plt.show()
plt.close()

plt.title("LEBC offset")
steps = np.arange(xlei.size)
t = conf["dt"] * steps
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
plt.savefig("lebc_offset.png", dpi=400)
plt.show()
plt.close()

plt.title("Displacement from spawn point")
xi = x[0 :: conf["Npart"]]
yi = y[0 :: conf["Npart"]]

dxisum = np.zeros(xi.shape)
dyisum = np.zeros(yi.shape)
for i in range(conf["Npart"]):
    dxisum += x[i :: conf["Npart"]] - x[i :: conf["Npart"]][0]
    dyisum += y[i :: conf["Npart"]] - y[i :: conf["Npart"]][0]

dxi = xi - xi[0]
dyi = yi - yi[0]
plt.plot(steps, dxi, "b-", label="x [particle 0]")
plt.plot(steps, dyi, "r-", label="y [particle 0]")
# plt.plot(steps, dxisum / conf["Npart"], "c-", label="x mean")
# plt.plot(steps, dyisum / conf["Npart"], "m-", label="y mean")
plt.plot([steps[0], steps[-1]], [0, 0], "k:", lw=1)
plt.plot(
    [steps[0], steps[-1]],
    [conf["box_dim"], conf["box_dim"]],
    "k:",
    lw=1,
)
plt.plot(
    [steps[0], steps[-1]],
    [-conf["box_dim"], -conf["box_dim"]],
    "k:",
    lw=1,
)

plt.legend()
plt.savefig("displacement_spawn.png", dpi=400)
plt.show()

plt.title("Displacement from box centre")
xi = x[0 :: conf["Npart"]]
yi = y[0 :: conf["Npart"]]

dxisum = np.zeros(xi.shape)
dyisum = np.zeros(yi.shape)
for i in range(conf["Npart"]):
    dxisum += x[i :: conf["Npart"]]
    dyisum += y[i :: conf["Npart"]]

dxi = xi
dyi = yi
plt.plot(steps, dxi, "b-", label="x [particle 0]")
plt.plot(steps, dyi, "r-", label="y [particle 0]")
# plt.plot(steps, dxisum / conf["Npart"], "c-", label="x mean")
# plt.plot(steps, dyisum / conf["Npart"], "m-", label="y mean")
plt.plot([steps[0], steps[-1]], [0, 0], "k:", lw=1)
plt.plot(
    [steps[0], steps[-1]],
    [conf["box_dim"], conf["box_dim"]],
    "k:",
    lw=1,
)
plt.plot(
    [steps[0], steps[-1]],
    [-conf["box_dim"], -conf["box_dim"]],
    "k:",
    lw=1,
)

plt.legend()
plt.savefig("displacement_box_centre.png", dpi=400)
plt.show()

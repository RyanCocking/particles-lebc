import numpy as np
import matplotlib.pyplot as plt


def box_offset_x(x, L, v_le, dt, step):
    if x >= L / 2:
        x -= L
        print(step)
    else:
        x += v_le * dt

    return x


steps = np.arange(10000)
dt = 1e-6
time = dt * steps
L = 100e-9
rate = 2e3
v_le = rate * L

print(f"dt = {dt:e}, L = {L:e}, rate = {rate:e}, v_le = {v_le:e}")

x = np.zeros(time.size)
for i, t in enumerate(time[:-1], 1):
    x[i] = box_offset_x(x[i - 1], L, v_le, dt, i)

plt.xlabel("Step")
plt.ylabel("$x_{LE}$")
plt.plot(steps, x, "r-")
plt.plot([(L / 2) / (v_le * dt), (L / 2) / (v_le * dt)], [-L / 1.5, L / 1.5], "k:")
plt.plot([0, steps[-1]], [L / 2, L / 2], "k:")
plt.plot([0, steps[-1]], [-L / 2, -L / 2], "k:")

plt.show()

"""
Simple visualization of an AMUSE binary particle file.

No assumption is made about:
    - number of particles
    - number of stars
    - particles belonging to each star

All particles in binary.amuse are plotted together.

Output:
    binary_xy.png
    binary_xz.png
    binary_yz.png
    binary_3d.png
"""

import numpy as np
import matplotlib.pyplot as plt

from amuse.io import read_set_from_file
from amuse.units import units


# ============================================================
# INPUT FILE
# ============================================================

BINARY_FILE = "binary.amuse"


# ============================================================
# READ PARTICLES
# ============================================================

print("Reading:", BINARY_FILE)

particles = read_set_from_file(
    BINARY_FILE,
    "amuse"
)

print("Total number of particles:", len(particles))


# ============================================================
# GET POSITIONS
# ============================================================

x = particles.x.value_in(units.RSun)
y = particles.y.value_in(units.RSun)
z = particles.z.value_in(units.RSun)


# ============================================================
# PLOT LIMITS
# ============================================================

xmin = x.min()
xmax = x.max()

ymin = y.min()
ymax = y.max()

zmin = z.min()
zmax = z.max()

max_range = max(
    xmax - xmin,
    ymax - ymin,
    zmax - zmin
)

padding = 0.05 * max_range

xmin -= padding
xmax += padding

ymin -= padding
ymax += padding

zmin -= padding
zmax += padding


# ============================================================
# X-Y VIEW
# ============================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    x,
    y,
    s=0.5,
    alpha=0.5
)

plt.xlabel(r"$x\;(R_\odot)$")
plt.ylabel(r"$y\;(R_\odot)$")

plt.title("Binary particle distribution — x-y")

plt.axis("equal")

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.tight_layout()

plt.savefig(
    "binary_xy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Written: binary_xy.png")


# ============================================================
# X-Z VIEW
# ============================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    x,
    z,
    s=0.5,
    alpha=0.5
)

plt.xlabel(r"$x\;(R_\odot)$")
plt.ylabel(r"$z\;(R_\odot)$")

plt.title("Binary particle distribution — x-z")

plt.axis("equal")

plt.xlim(xmin, xmax)
plt.ylim(zmin, zmax)

plt.tight_layout()

plt.savefig(
    "binary_xz.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Written: binary_xz.png")


# ============================================================
# Y-Z VIEW
# ============================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    y,
    z,
    s=0.5,
    alpha=0.5
)

plt.xlabel(r"$y\;(R_\odot)$")
plt.ylabel(r"$z\;(R_\odot)$")

plt.title("Binary particle distribution — y-z")

plt.axis("equal")

plt.xlim(ymin, ymax)
plt.ylim(zmin, zmax)

plt.tight_layout()

plt.savefig(
    "binary_yz.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Written: binary_yz.png")


# ============================================================
# 3-D VIEW
# ============================================================

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.scatter(
    x,
    y,
    z,
    s=0.5,
    alpha=0.25
)

ax.set_xlabel(r"$x\;(R_\odot)$")
ax.set_ylabel(r"$y\;(R_\odot)$")
ax.set_zlabel(r"$z\;(R_\odot)$")

ax.set_title("Binary particle distribution — 3D")

plt.tight_layout()

plt.savefig(
    "binary_3d.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Written: binary_3d.png")


# ============================================================
# DONE
# ============================================================

print("\nVisualization complete.")


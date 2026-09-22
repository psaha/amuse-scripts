"""
Visualize a binary star particle model stored in binary.amuse.

The file contains:
    Star 1 = first 50,000 particles
    Star 2 = next 70,000 particles

The script produces:
    binary_xy.png   : view along z-axis
    binary_xz.png   : view along y-axis
    binary_3d.png   : 3D view
"""

import numpy as np
import matplotlib.pyplot as plt

from amuse.io import read_set_from_file
from amuse.units import units


# ============================================================
# PARAMETERS
# ============================================================

BINARY_FILE = "binary.amuse"

NSTAR1 = 50000
NSTAR2 = 70000

OUTPUT_XY = "binary_xy.png"
OUTPUT_XZ = "binary_xz.png"
OUTPUT_3D = "binary_3d.png"


# ============================================================
# READ BINARY
# ============================================================

print("Reading:", BINARY_FILE)

binary = read_set_from_file(BINARY_FILE, "amuse")

print("Total particles:", len(binary))


# ============================================================
# SPLIT THE TWO STARS
# ============================================================

star1 = binary[:NSTAR1]
star2 = binary[NSTAR1:NSTAR1 + NSTAR2]

print("Star 1 particles:", len(star1))
print("Star 2 particles:", len(star2))


# ============================================================
# CONVERT POSITIONS TO SOLAR RADII
# ============================================================

x1 = star1.x.value_in(units.RSun)
y1 = star1.y.value_in(units.RSun)
z1 = star1.z.value_in(units.RSun)

x2 = star2.x.value_in(units.RSun)
y2 = star2.y.value_in(units.RSun)
z2 = star2.z.value_in(units.RSun)


# ============================================================
# CALCULATE COM OF EACH STAR
# ============================================================

def center_of_mass(star):

    mass = star.mass

    x = (mass * star.x).sum() / mass.sum()
    y = (mass * star.y).sum() / mass.sum()
    z = (mass * star.z).sum() / mass.sum()

    return x, y, z


com1 = center_of_mass(star1)
com2 = center_of_mass(star2)

com1 = np.array([
    com1[0].value_in(units.RSun),
    com1[1].value_in(units.RSun),
    com1[2].value_in(units.RSun)
])

com2 = np.array([
    com2[0].value_in(units.RSun),
    com2[1].value_in(units.RSun),
    com2[2].value_in(units.RSun)
])


print("\nStar COM positions:")
print("Star 1:", com1, "Rsun")
print("Star 2:", com2, "Rsun")

separation = np.linalg.norm(com2 - com1)

print("\nCOM separation:")
print("d =", separation, "Rsun")


# ============================================================
# DETERMINE PLOT LIMITS
# ============================================================

all_x = np.concatenate([x1, x2])
all_y = np.concatenate([y1, y2])
all_z = np.concatenate([z1, z2])

xmin = all_x.min()
xmax = all_x.max()

ymin = all_y.min()
ymax = all_y.max()

zmin = all_z.min()
zmax = all_z.max()

# Add some padding
padding = 0.1 * max(
    xmax - xmin,
    ymax - ymin,
    zmax - zmin
)

xmin -= padding
xmax += padding
ymin -= padding
ymax += padding
zmin -= padding
zmax += padding


# ============================================================
# 1. X-Y VIEW
# ============================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    x1,
    y1,
    s=0.5,
    alpha=0.5,
    label="Star 1"
)

plt.scatter(
    x2,
    y2,
    s=0.5,
    alpha=0.5,
    label="Star 2"
)

# Mark stellar COMs
plt.scatter(
    com1[0],
    com1[1],
    marker="x",
    s=100,
    linewidths=2
)

plt.scatter(
    com2[0],
    com2[1],
    marker="x",
    s=100,
    linewidths=2
)

# Draw line between COMs
plt.plot(
    [com1[0], com2[0]],
    [com1[1], com2[1]],
    linestyle="--",
    linewidth=1
)

plt.xlabel(r"$x\;(R_\odot)$")
plt.ylabel(r"$y\;(R_\odot)$")

plt.title("Binary star particle distribution — x-y view")

plt.axis("equal")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_XY, dpi=300)
plt.show()
plt.close()

print("Written:", OUTPUT_XY)


# ============================================================
# 2. X-Z VIEW
# ============================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    x1,
    z1,
    s=0.5,
    alpha=0.5,
    label="Star 1"
)

plt.scatter(
    x2,
    z2,
    s=0.5,
    alpha=0.5,
    label="Star 2"
)

# COMs
plt.scatter(
    com1[0],
    com1[2],
    marker="x",
    s=100,
    linewidths=2
)

plt.scatter(
    com2[0],
    com2[2],
    marker="x",
    s=100,
    linewidths=2
)

# COM separation line
plt.plot(
    [com1[0], com2[0]],
    [com1[2], com2[2]],
    linestyle="--",
    linewidth=1
)

plt.xlabel(r"$x\;(R_\odot)$")
plt.ylabel(r"$z\;(R_\odot)$")

plt.title("Binary star particle distribution — x-z view")

plt.axis("equal")
plt.xlim(xmin, xmax)
plt.ylim(zmin, zmax)

plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_XZ, dpi=300)
plt.show()
plt.close()

print("Written:", OUTPUT_XZ)


# ============================================================
# 3. 3-D VIEW
# ============================================================

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    x1,
    y1,
    z1,
    s=0.5,
    alpha=0.25,
    label="Star 1"
)

ax.scatter(
    x2,
    y2,
    z2,
    s=0.5,
    alpha=0.25,
    label="Star 2"
)

# COMs
ax.scatter(
    com1[0],
    com1[1],
    com1[2],
    marker="x",
    s=100,
    linewidths=2
)

ax.scatter(
    com2[0],
    com2[1],
    com2[2],
    marker="x",
    s=100,
    linewidths=2
)

# Line joining COMs
ax.plot(
    [com1[0], com2[0]],
    [com1[1], com2[1]],
    [com1[2], com2[2]],
    linestyle="--",
    linewidth=1
)

ax.set_xlabel(r"$x\;(R_\odot)$")
ax.set_ylabel(r"$y\;(R_\odot)$")
ax.set_zlabel(r"$z\;(R_\odot)$")

ax.set_title("Binary star particle distribution — 3D")

ax.legend()

plt.tight_layout()

plt.savefig(OUTPUT_3D, dpi=300)
plt.show()
plt.close()

print("Written:", OUTPUT_3D)


print("\nVisualization complete.")


"""
Extract a visible SPH surface particle list for an observer looking along +z.

Input:
    polytrope_relaxed.amuse

Output:
    visible_surface_particles.csv

For each SPH particle we calculate an approximate visibility:
    visibility = 1 - total projected kernel overlap from particles in front

A particle is retained if:
    visibility >= VISIBILITY_MIN

The projected smoothing length is h_smooth.

The current brightness proxy is:
    brightness = visibility

This is deliberately a geometry/surface-extraction stage.
Temperature-dependent brightness can be added later.
"""

import numpy as np
from amuse.io import read_set_from_file
from amuse.units import units


# ============================================================
# USER PARAMETERS
# ============================================================

INPUT_FILE = "polytrope_relaxed.amuse"
OUTPUT_FILE = "visible_surface_particles.csv"

# Minimum visibility for a particle to be retained.
# 0.01 means particles with less than 1% estimated visibility
# are discarded.
VISIBILITY_MIN = 0.0001

# Fraction of smoothing length used as the projected kernel radius.
#
# h_smooth is normally the characteristic SPH smoothing length.
# We initially use h itself as the projected interaction radius.
KERNEL_RADIUS_FACTOR = 1.0

# Number of nearest projected neighbours examined for each particle.
# Increase this if the particle distribution is sparse.
N_NEIGHBOURS = 64

# ============================================================
# LOAD SPH PARTICLES
# ============================================================

print("Reading:", INPUT_FILE)

stars = read_set_from_file(INPUT_FILE, "amuse")

N = len(stars)

print("Number of particles:", N)


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

# Coordinates in units of solar radius.
x = stars.x.value_in(units.RSun)
y = stars.y.value_in(units.RSun)
z = stars.z.value_in(units.RSun)

# Smoothing length in solar radius.
h = stars.h_smooth.value_in(units.RSun)

# Physical quantities.
mass = stars.mass.value_in(units.MSun)
rho = stars.rho.value_in(units.g / units.cm**3)

# Internal energy in SI.
u = stars.u.value_in(units.m**2 / units.s**2)


# ============================================================
# INITIALISE VISIBILITY
# ============================================================

visibility = np.ones(N)

print("Calculating projected visibility...")


# ============================================================
# PROJECTED PARTICLE VISIBILITY
# ============================================================

"""
Observer is located at +z.

Therefore:

larger z = closer to observer

For each particle i we look for particles j satisfying:

    z_j > z_i

and

    projected_distance(i,j) < h_j

The foreground particle j can obscure particle i.

We use a smooth Gaussian-like projected overlap:

    overlap = exp[-R^2 / (2 h_j^2)]

This is NOT the exact Fi SPH kernel. It is an initial
approximation to construct a smooth visible surface.

The total obscuration is accumulated and converted to:

    visibility = exp(-obscuration)

This keeps:

    0 <= visibility <= 1
"""


# Sort particles from observer toward the back.
# Largest z first.
order = np.argsort(z)[::-1]

# Process particles in depth order.
for position, idx in enumerate(order):

    xi = x[idx]
    yi = y[idx]
    zi = z[idx]
    hi = h[idx]

    # Particles already processed have larger z,
    # therefore they are in front of particle i.
    foreground = order[:position]

    if len(foreground) == 0:
        continue

    # --------------------------------------------------------
    # Calculate projected distances to foreground particles
    # --------------------------------------------------------

    dx = x[foreground] - xi
    dy = y[foreground] - yi

    R2 = dx * dx + dy * dy

    # Use foreground particles' smoothing lengths.
    hj = h[foreground]

    radius = KERNEL_RADIUS_FACTOR * hj

    # Only particles whose projected smoothing footprint
    # reaches particle i can obscure it.
    mask = R2 < radius * radius

    if not np.any(mask):
        continue

    R2_local = R2[mask]
    h_local = hj[mask]

    # --------------------------------------------------------
    # Gaussian projected kernel
    # --------------------------------------------------------

    overlap = np.exp(
        -0.5 * R2_local / (h_local * h_local)
    )

    # Limit the total obscuration contribution.
    obscuration = np.sum(overlap)

    # Convert to transmission/visibility.
    visibility[idx] = np.exp(-obscuration)


# ============================================================
# SELECT VISIBLE SURFACE PARTICLES
# ============================================================

surface_mask = visibility >= VISIBILITY_MIN

surface_indices = np.where(surface_mask)[0]

print()
print("Visible particles:", len(surface_indices))
print(
    "Fraction retained:",
    len(surface_indices) / N
)

# ============================================================
# CREATE SURFACE ARRAYS
# ============================================================

surface_x = x[surface_mask]
surface_y = y[surface_mask]
surface_z = z[surface_mask]

surface_h = h[surface_mask]

surface_mass = mass[surface_mask]
surface_rho = rho[surface_mask]
surface_u = u[surface_mask]

surface_visibility = visibility[surface_mask]


# ============================================================
# BRIGHTNESS PROXY
# ============================================================

"""
At this stage we do NOT yet know the temperature.

Therefore the safest first brightness quantity is simply
the visibility-weighted surface contribution.

Later we can replace this with something like:

    brightness = visibility * T**4

or include projected area and limb darkening.
"""

brightness = surface_visibility


# ============================================================
# SAVE CSV
# ============================================================

data = np.column_stack(
    (
        surface_x,
        surface_y,
        surface_z,
        surface_h,
        surface_mass,
        surface_rho,
        surface_u,
        surface_visibility,
        brightness,
    )
)

header = (
    "x_Rsun,"
    "y_Rsun,"
    "z_Rsun,"
    "h_Rsun,"
    "mass_Msun,"
    "rho_g_cm3,"
    "u_m2_s2,"
    "visibility,"
    "brightness"
)

np.savetxt(
    OUTPUT_FILE,
    data,
    delimiter=",",
    header=header,
    comments=""
)

print()
print("Saved:", OUTPUT_FILE)


# ============================================================
# SUMMARY
# ============================================================

print()
print("Surface particle ranges:")
print("--------------------------------")

print(
    "x:",
    surface_x.min(),
    "to",
    surface_x.max(),
    "Rsun"
)

print(
    "y:",
    surface_y.min(),
    "to",
    surface_y.max(),
    "Rsun"
)

print(
    "z:",
    surface_z.min(),
    "to",
    surface_z.max(),
    "Rsun"
)

print(
    "visibility:",
    surface_visibility.min(),
    "to",
    surface_visibility.max()
)

print(
    "brightness:",
    brightness.min(),
    "to",
    brightness.max()
)



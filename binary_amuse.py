"""
Create a binary-star AMUSE particle set from two single-star
AMUSE particle files.

Each input file should contain the particles belonging to one
relaxed star.

The stars are:

    1. recentered on their individual centers of mass
    2. recentered in velocity on their individual COM velocities
    3. placed on a Keplerian binary orbit
    4. given the corresponding barycentric velocity offsets
    5. combined into one AMUSE particle set

Orbital elements supplied by the user:

    a      : semi-major axis
    e      : eccentricity
    inc    : inclination
    omega  : argument of pericenter
    Omega  : longitude of ascending node
    f      : true anomaly

IMPORTANT:
    a is the relative semi-major axis of the binary:

        a = a1 + a2

    It is NOT the semi-major axis of either individual star.
"""

import numpy as np

from amuse.io import read_set_from_file, write_set_to_file
from amuse.datamodel import Particles
from amuse.units import units, constants


# ============================================================
# USER INPUT
# ============================================================

STAR1_FILE = "star1.amuse"
STAR2_FILE = "star2.amuse"

OUTPUT_FILE = "binary.amuse"


# ------------------------------------------------------------
# Binary orbital parameters
# ------------------------------------------------------------

# Relative semi-major axis:
#
# a = separation semi-major axis = a1 + a2
#
# Example:
# a = 5 | units.RSun

a = 5.0 | units.RSun

# Eccentricity
e = 0.2

# Orbital orientation
inclination = 20.0 | units.deg

argument_of_pericenter = 30.0 | units.deg

longitude_of_ascending_node = 40.0 | units.deg

# Position along orbit.
#
# This is the TRUE ANOMALY, measured from pericenter.
#
# f = 0 deg   -> pericenter
# f = 180 deg -> apocenter

true_anomaly = 45.0 | units.deg


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rotate_orbit_vector(vector, Omega, inclination, omega):
    """
    Rotate a vector from the orbital plane into the inertial
    coordinate system.

    The standard orbital-element convention is used:

        Rz(Omega) Rx(inclination) Rz(omega)
    """

    Om = Omega.value_in(units.rad)
    inc = inclination.value_in(units.rad)
    om = omega.value_in(units.rad)

    # Rotation matrices
    Rz_Omega = np.array([
        [np.cos(Om), -np.sin(Om), 0.0],
        [np.sin(Om),  np.cos(Om), 0.0],
        [0.0,         0.0,        1.0]
    ])

    Rx_inc = np.array([
        [1.0, 0.0,              0.0],
        [0.0, np.cos(inc), -np.sin(inc)],
        [0.0, np.sin(inc),  np.cos(inc)]
    ])

    Rz_omega = np.array([
        [np.cos(om), -np.sin(om), 0.0],
        [np.sin(om),  np.cos(om), 0.0],
        [0.0,         0.0,        1.0]
    ])

    rotation = (
        Rz_Omega
        @ Rx_inc
        @ Rz_omega
    )

    return rotation @ vector


def calculate_relative_orbit(
        mass1,
        mass2,
        a,
        e,
        inclination,
        omega,
        Omega,
        true_anomaly
):
    """
    Calculate the relative position and velocity:

        r = r2 - r1
        v = v2 - v1

    for a Keplerian two-body orbit.
    """

    # --------------------------------------------------------
    # Convert angles
    # --------------------------------------------------------

    f = true_anomaly.value_in(units.rad)

    # --------------------------------------------------------
    # Separation at current true anomaly
    # --------------------------------------------------------

    r = (
        a
        * (1.0 - e**2)
        / (1.0 + e * np.cos(f))
    )

    # --------------------------------------------------------
    # Position in orbital plane
    # --------------------------------------------------------

    r_orbit = np.array([
        r * np.cos(f),
        r * np.sin(f),
        0.0 | units.m
    ])

    # --------------------------------------------------------
    # Gravitational parameter
    # --------------------------------------------------------

    mu = constants.G * (mass1 + mass2)

    # --------------------------------------------------------
    # Orbital-plane velocity
    #
    # h = sqrt(mu*a*(1-e^2))
    #
    # v_r     = mu/h * e sin(f)
    # v_theta = mu/h * (1 + e cos(f))
    # --------------------------------------------------------

    h_orbit = np.sqrt(
        mu * a * (1.0 - e**2)
    )

    v_r = (
        mu / h_orbit
        * e
        * np.sin(f)
    )

    v_theta = (
        mu / h_orbit
        * (1.0 + e * np.cos(f))
    )

    vx_orbit = (
        v_r * np.cos(f)
        - v_theta * np.sin(f)
    )

    vy_orbit = (
        v_r * np.sin(f)
        + v_theta * np.cos(f)
    )

    v_orbit = np.array([
        vx_orbit,
        vy_orbit,
        0.0 | units.m / units.s
    ])

    # --------------------------------------------------------
    # Rotate orbital plane into inertial frame
    # --------------------------------------------------------

    r_relative = rotate_orbit_vector(
        r_orbit,
        Omega,
        inclination,
        omega
    )

    v_relative = rotate_orbit_vector(
        v_orbit,
        Omega,
        inclination,
        omega
    )

    return r_relative, v_relative


# ============================================================
# READ THE TWO STARS
# ============================================================

STAR1_FILE = "polytrope_relaxed.amuse" 
STAR2_FILE = "polytrope_relaxed_2.amuse"

print("Reading star 1:", STAR1_FILE)

star1 = read_set_from_file(
    STAR1_FILE,
    "amuse"
)

print("Reading star 2:", STAR2_FILE)

star2 = read_set_from_file(
    STAR2_FILE,
    "amuse"
)

print()
print("Number of particles:")
print("  Star 1:", len(star1))
print("  Star 2:", len(star2))


# ============================================================
# CALCULATE STELLAR MASSES
# ============================================================

M1 = star1.mass.sum()
M2 = star2.mass.sum()

Mtot = M1 + M2

print()
print("Stellar masses:")
print("  M1 =", M1)
print("  M2 =", M2)
print("  Mtot =", Mtot)


# ============================================================
# CHECK ORBIT
# ============================================================

if e < 0.0 or e >= 1.0:
    raise ValueError(
        "This script currently requires 0 <= e < 1 "
        "for a bound elliptical orbit."
    )

if a <= 0 | units.m:
    raise ValueError(
        "Semi-major axis must be positive."
    )


# ============================================================
# COPY PARTICLES
# ============================================================

# Make copies so that the original files are not modified.

star1 = star1.copy()
star2 = star2.copy()


# ============================================================
# RECENTER EACH STAR
# ============================================================

"""
The input stars may not have their exact center of mass at
(x,y,z) = (0,0,0).

We remove the COM position and COM velocity first.

After this operation each individual star is internally
unchanged, but its COM is at the origin.
"""

com1_position = star1.center_of_mass()
com2_position = star2.center_of_mass()

com1_velocity = star1.center_of_mass_velocity()
com2_velocity = star2.center_of_mass_velocity()

print()
print("Original COM positions:")
print("  Star 1:", com1_position)
print("  Star 2:", com2_position)

print()
print("Original COM velocities:")
print("  Star 1:", com1_velocity)
print("  Star 2:", com2_velocity)


# ------------------------------------------------------------
# Recenter star 1
# ------------------------------------------------------------

star1.x -= com1_position[0]
star1.y -= com1_position[1]
star1.z -= com1_position[2]

star1.vx -= com1_velocity[0]
star1.vy -= com1_velocity[1]
star1.vz -= com1_velocity[2]


# ------------------------------------------------------------
# Recenter star 2
# ------------------------------------------------------------

star2.x -= com2_position[0]
star2.y -= com2_position[1]
star2.z -= com2_position[2]

star2.vx -= com2_velocity[0]
star2.vy -= com2_velocity[1]
star2.vz -= com2_velocity[2]


# ============================================================
# CALCULATE KEPLERIAN ORBIT
# ============================================================

r_relative, v_relative = calculate_relative_orbit(
    M1,
    M2,
    a,
    e,
    inclination,
    argument_of_pericenter,
    longitude_of_ascending_node,
    true_anomaly
)


# ============================================================
# BARYCENTRIC POSITIONS
# ============================================================

"""
r_relative = r2 - r1

For the binary COM to remain at the origin:

    r1 = -M2/(M1+M2) * r_relative

    r2 =  M1/(M1+M2) * r_relative
"""

r1 = (
    -M2 / Mtot
    * r_relative
)

r2 = (
    M1 / Mtot
    * r_relative
)


# ============================================================
# BARYCENTRIC VELOCITIES
# ============================================================

"""
Similarly:

    v1 = -M2/(M1+M2) * v_relative

    v2 =  M1/(M1+M2) * v_relative
"""

v1 = (
    -M2 / Mtot
    * v_relative
)

v2 = (
    M1 / Mtot
    * v_relative
)


# ============================================================
# PRINT ORBITAL INFORMATION
# ============================================================

print()
print("Binary orbital parameters:")
print("--------------------------")

print("a =", a)
print("e =", e)
print("inclination =", inclination)
print("argument of pericenter =", argument_of_pericenter)
print("longitude of ascending node =", longitude_of_ascending_node)
print("true anomaly =", true_anomaly)

print()
print("Relative position:")
print("  r =", r_relative)

print()
print("Relative velocity:")
print("  v =", v_relative)

print()
print("Stellar COM positions:")
print("  Star 1:", r1)
print("  Star 2:", r2)

print()
print("Stellar COM velocities:")
print("  Star 1:", v1)
print("  Star 2:", v2)


# ============================================================
# APPLY POSITION OFFSETS
# ============================================================

star1.x += r1[0]
star1.y += r1[1]
star1.z += r1[2]

star2.x += r2[0]
star2.y += r2[1]
star2.z += r2[2]


# ============================================================
# APPLY VELOCITY OFFSETS
# ============================================================

star1.vx += v1[0]
star1.vy += v1[1]
star1.vz += v1[2]

star2.vx += v2[0]
star2.vy += v2[1]
star2.vz += v2[2]


# ============================================================
# COMBINE THE TWO PARTICLE SETS
# ============================================================

binary = Particles()

binary.add_particles(star1)
binary.add_particles(star2)


# ============================================================
# CHECK FINAL BINARY COM
# ============================================================

final_com_position = binary.center_of_mass()
final_com_velocity = binary.center_of_mass_velocity()

print()
print("Final binary COM:")
print("  position =", final_com_position)
print("  velocity =", final_com_velocity)

print()
print("Final number of particles:")
print(" ", len(binary))


# ============================================================
# WRITE OUTPUT FILE
# ============================================================

print()
print("Writing:", OUTPUT_FILE)

write_set_to_file(
    binary,
    OUTPUT_FILE,
    "amuse"
)

print("Done.")



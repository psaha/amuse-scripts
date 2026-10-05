import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

from amuse.io import read_set_from_file
from amuse.units import units, constants, nbody_system
from amuse.community.fi.interface import Fi


# ============================================================
# USER PARAMETERS
# ============================================================

INPUT_FILE = "amuse_file/polytrope_single_star.amuse"
FRAME_DIR = "frames_single"

N_FRAMES = 30
TOTAL_TIME = 3.0
VISIBILITY_MIN = 0.0001
KERNEL_RADIUS_FACTOR = 1.0

IMAGE_SIZE = 512
KERNEL_RADIUS = 2.5
GAUSSIAN_WIDTH = 0.5
DPI = 120

OUTPUT_PREFIX = "single_star"
# Damping
USE_DAMPING = False
DAMPING = 0.5


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(FRAME_DIR, exist_ok=True)


# ============================================================
# LOAD STAR
# ============================================================

print("Reading:", INPUT_FILE)

parts = read_set_from_file(INPUT_FILE, "amuse")

N = len(parts)

print("Number of particles:", N)


# ============================================================
# STAR PARAMETERS
# ============================================================

M_star = parts.mass.sum()

print("Mass:", M_star.value_in(units.MSun), "MSun")


# ============================================================
# DYNAMICAL TIME
# ============================================================

R_star = 1.25 | units.RSun

t_dyn = (R_star**3 / (constants.G * M_star)).sqrt()

print("Dynamical time:",
      t_dyn.value_in(units.s), "s")

print("Total evolution time:",
      (TOTAL_TIME * t_dyn).value_in(units.s), "s")


# ============================================================
# FI SPH CODE
# ============================================================

conv = nbody_system.nbody_to_si(M_star, R_star)

sph = Fi(conv, channel_type='sockets')

sph.parameters.timestep = t_dyn / 100.0
sph.parameters.periodic_box_size = 20.0 | units.RSun
sph.parameters.verbosity = 0

sph.gas_particles.add_particles(parts)


to_code = parts.new_channel_to(sph.gas_particles)
from_code = sph.gas_particles.new_channel_to(parts)


# ============================================================
# SURFACE PARTICLES
# ============================================================

def calculate_surface(parts):

    x = parts.x.value_in(units.RSun)
    y = parts.y.value_in(units.RSun)
    z = parts.z.value_in(units.RSun)

    h = parts.h_smooth.value_in(units.RSun)

    N = len(parts)

    visibility = np.ones(N)

    print("Calculating surface particles...")

    # --------------------------------------------------------
    # Sort from observer to back
    # --------------------------------------------------------

    order = np.argsort(z)[::-1]

    # --------------------------------------------------------
    # KD-tree in projected x-y plane
    # --------------------------------------------------------

    points = np.column_stack((x, y))

    tree = cKDTree(points)

    max_h = np.max(h) * KERNEL_RADIUS_FACTOR

    # --------------------------------------------------------
    # Process particles from front to back
    # --------------------------------------------------------

    for position, idx in enumerate(order):

        if position == 0:
            continue

        xi = x[idx]
        yi = y[idx]

        # Search only nearby projected particles
        neighbours = tree.query_ball_point(
            [xi, yi],
            max_h
        )

        if len(neighbours) == 0:
            continue

        neighbours = np.asarray(neighbours, dtype=int)

        # Only particles in front
        neighbours = neighbours[
            z[neighbours] > z[idx]
        ]

        if len(neighbours) == 0:
            continue

        dx = x[neighbours] - xi
        dy = y[neighbours] - yi

        R2 = dx * dx + dy * dy

        hj = h[neighbours]

        radius = KERNEL_RADIUS_FACTOR * hj

        mask = R2 < radius * radius

        if not np.any(mask):
            continue

        R2_local = R2[mask]
        h_local = hj[mask]

        overlap = np.exp(
            -0.5 * R2_local / (h_local * h_local)
        )

        obscuration = np.sum(overlap)

        visibility[idx] = np.exp(-obscuration)

    surface_mask = visibility >= VISIBILITY_MIN

    return surface_mask, visibility


# ============================================================
# RENDER SURFACE
# ============================================================

def render_surface(parts, visibility, frame_number):

    x = parts.x.value_in(units.RSun)
    y = parts.y.value_in(units.RSun)
    h = parts.h_smooth.value_in(units.RSun)

    mask = (
        np.isfinite(x) &
        np.isfinite(y) &
        np.isfinite(h) &
        np.isfinite(visibility) &
        (h > 0) &
        (visibility > 0)
    )

    x = x[mask]
    y = y[mask]
    h = h[mask]
    brightness = visibility[mask]

    print("Particles used:", len(x))

    # --------------------------------------------------------
    # IMAGE EXTENT
    # --------------------------------------------------------

    xmin = np.min(x - KERNEL_RADIUS * h)
    xmax = np.max(x + KERNEL_RADIUS * h)

    ymin = np.min(y - KERNEL_RADIUS * h)
    ymax = np.max(y + KERNEL_RADIUS * h)

    extent = max(
        xmax - xmin,
        ymax - ymin
    )

    xc = 0.5 * (xmin + xmax)
    yc = 0.5 * (ymin + ymax)

    xmin = xc - extent / 2
    xmax = xc + extent / 2

    ymin = yc - extent / 2
    ymax = yc + extent / 2

    # --------------------------------------------------------
    # IMAGE GRID
    # --------------------------------------------------------

    image = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float64
    )

    pixel_size = extent / IMAGE_SIZE

    # --------------------------------------------------------
    # DEPOSIT PARTICLES
    # --------------------------------------------------------

    for i in range(len(x)):

        xi = x[i]
        yi = y[i]
        hi = h[i]
        bi = brightness[i]

        px = (xi - xmin) / pixel_size
        py = (yi - ymin) / pixel_size

        sigma = GAUSSIAN_WIDTH * hi

        radius = KERNEL_RADIUS * hi

        radius_pixels = int(
            np.ceil(radius / pixel_size)
        )

        ix = int(px)
        iy = int(py)

        ix0 = max(
            0,
            ix - radius_pixels
        )

        ix1 = min(
            IMAGE_SIZE - 1,
            ix + radius_pixels
        )

        iy0 = max(
            0,
            iy - radius_pixels
        )

        iy1 = min(
            IMAGE_SIZE - 1,
            iy + radius_pixels
        )

        if ix0 > ix1 or iy0 > iy1:
            continue

        xx = (
            np.arange(ix0, ix1 + 1) + 0.5
        ) * pixel_size + xmin

        yy = (
            np.arange(iy0, iy1 + 1) + 0.5
        ) * pixel_size + ymin

        XX, YY = np.meshgrid(xx, yy)

        r2 = (
            (XX - xi)**2 +
            (YY - yi)**2
        )

        kernel = np.exp(
            -r2 / (2.0 * sigma * sigma)
        )

        image[
            iy0:iy1 + 1,
            ix0:ix1 + 1
        ] += bi * kernel

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    maximum = np.max(image)

    if maximum > 0:
        image /= maximum

    # --------------------------------------------------------
    # SAVE FRAME
    # --------------------------------------------------------

    filename = os.path.join(
        FRAME_DIR,
        f"{OUTPUT_PREFIX}_{frame_number:04d}.png"
    )

    plt.figure(
        figsize=(7, 7),
        facecolor="black"
    )

    plt.imshow(
        image,
        origin="lower",
        extent=[
            xmin,
            xmax,
            ymin,
            ymax
        ],
        cmap="inferno",
        interpolation="bilinear",
        vmin=0,
        vmax=1
    )

    plt.axis("off")

    plt.tight_layout(pad=0)

    plt.savefig(
        filename,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="black"
    )

    plt.close()

    print("Saved:", filename)


# ============================================================
# EVOLUTION
# ============================================================

print()
print("==============================================")
print("STARTING EVOLUTION")
print("==============================================")


dt_output = TOTAL_TIME * t_dyn / N_FRAMES


# ------------------------------------------------------------
# FRAME 0
# ------------------------------------------------------------

print()
print("FRAME 0000")
print("Time = 0")

from_code.copy()

surface_mask, visibility = calculate_surface(parts)

print(
    "Surface particles:",
    np.sum(surface_mask)
)

render_surface(
    parts,
    visibility,
    0
)


# ------------------------------------------------------------
# EVOLUTION LOOP
# ------------------------------------------------------------

for frame in range(1, N_FRAMES + 1):

    target_time = frame * dt_output

    print()
    print("==============================================")
    print(
        f"FRAME {frame:04d}/{N_FRAMES}"
    )
    print(
        "Time =",
        target_time.value_in(units.s),
        "s"
    )
    print("==============================================")

    sph.evolve_model(target_time)

    from_code.copy()

    # --------------------------------------------------------
    # Optional damping
    # --------------------------------------------------------

    if USE_DAMPING:

        parts.vx *= DAMPING
        parts.vy *= DAMPING
        parts.vz *= DAMPING

        to_code.copy_attributes(
            ["vx", "vy", "vz"]
        )

    # --------------------------------------------------------
    # Surface
    # --------------------------------------------------------

    surface_mask, visibility = calculate_surface(parts)

    print(
        "Surface particles:",
        np.sum(surface_mask)
    )

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    render_surface(
        parts,
        visibility,
        frame
    )


# ============================================================
# STOP FI
# ============================================================

sph.stop()

print()
print("==============================================")
print("EVOLUTION COMPLETE")
print("==============================================")

print(
    "Frames saved in:",
    FRAME_DIR
)

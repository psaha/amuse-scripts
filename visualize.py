"""
Render visible SPH surface particles from CSV into a smooth
grayscale PNG image.

Input:
    visible_surface_particles.csv

Output:
    stellar_surface.png

The CSV is expected to contain:

    x_Rsun
    y_Rsun
    z_Rsun
    h_Rsun
    mass_Msun
    rho_g_cm3
    u_m2_s2
    visibility
    brightness

The image is viewed along the +z direction, so x and y are
the image coordinates.

Each particle contributes a Gaussian-smoothed brightness
distribution with width related to h_Rsun.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER PARAMETERS
# ============================================================

INPUT_FILE = "visible_surface_particles.csv"
OUTPUT_FILE = "stellar_surface.png"

# Number of image pixels in each direction.
IMAGE_SIZE = 800

# Number of smoothing lengths used for the Gaussian footprint.
# 2.5 means the kernel is evaluated out to 2.5*h.
KERNEL_RADIUS = 2.5

# Controls the width of the Gaussian.
# Smaller -> sharper image
# Larger  -> smoother image
GAUSSIAN_WIDTH = 0.5

# Minimum brightness below which particles are ignored.
BRIGHTNESS_MIN = 0.0

# Output image DPI.
DPI = 150


# ============================================================
# READ CSV
# ============================================================

print("Reading:", INPUT_FILE)

data = np.genfromtxt(
    INPUT_FILE,
    delimiter=",",
    names=True
)

x = data["x_Rsun"]
y = data["y_Rsun"]
h = data["h_Rsun"]
brightness = data["brightness"]


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

mask = (
    np.isfinite(x)
    & np.isfinite(y)
    & np.isfinite(h)
    & np.isfinite(brightness)
    & (h > 0)
    & (brightness > BRIGHTNESS_MIN)
)

x = x[mask]
y = y[mask]
h = h[mask]
brightness = brightness[mask]

print("Particles used:", len(x))


# ============================================================
# IMAGE EXTENT
# ============================================================

# Determine the stellar extent.
#
# Include smoothing lengths so that the outer edge of the
# stellar disk is not clipped.

xmin = np.min(x - KERNEL_RADIUS * h)
xmax = np.max(x + KERNEL_RADIUS * h)

ymin = np.min(y - KERNEL_RADIUS * h)
ymax = np.max(y + KERNEL_RADIUS * h)

# Use the same scale in x and y so the star is not distorted.
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


# ============================================================
# CREATE IMAGE GRID
# ============================================================

image = np.zeros(
    (IMAGE_SIZE, IMAGE_SIZE),
    dtype=np.float64
)

pixel_size = extent / IMAGE_SIZE

print("Image size:", IMAGE_SIZE, "x", IMAGE_SIZE)
print("Pixel size:", pixel_size, "Rsun")


# ============================================================
# DEPOSIT PARTICLES
# ============================================================

print("Rendering particles...")

for i in range(len(x)):

    xi = x[i]
    yi = y[i]
    hi = h[i]
    bi = brightness[i]

    # --------------------------------------------------------
    # Convert particle position to pixel coordinates
    # --------------------------------------------------------

    px = (xi - xmin) / pixel_size
    py = (yi - ymin) / pixel_size

    # --------------------------------------------------------
    # Gaussian smoothing width
    # --------------------------------------------------------

    sigma = GAUSSIAN_WIDTH * hi

    # Radius of kernel in physical units.
    radius = KERNEL_RADIUS * hi

    # Convert radius to pixels.
    radius_pixels = int(np.ceil(radius / pixel_size))

    # Pixel containing particle centre.
    ix = int(px)
    iy = int(py)

    # --------------------------------------------------------
    # Image boundaries
    # --------------------------------------------------------

    ix0 = max(0, ix - radius_pixels)
    ix1 = min(IMAGE_SIZE - 1, ix + radius_pixels)

    iy0 = max(0, iy - radius_pixels)
    iy1 = min(IMAGE_SIZE - 1, iy + radius_pixels)

    if ix0 > ix1 or iy0 > iy1:
        continue

    # --------------------------------------------------------
    # Local pixel coordinates
    # --------------------------------------------------------

    xx = (
        np.arange(ix0, ix1 + 1) + 0.5
    ) * pixel_size + xmin

    yy = (
        np.arange(iy0, iy1 + 1) + 0.5
    ) * pixel_size + ymin

    XX, YY = np.meshgrid(xx, yy)

    # --------------------------------------------------------
    # Distance from particle
    # --------------------------------------------------------

    r2 = (
        (XX - xi) ** 2
        + (YY - yi) ** 2
    )

    # --------------------------------------------------------
    # Gaussian kernel
    # --------------------------------------------------------

    kernel = np.exp(
        -r2 / (2.0 * sigma * sigma)
    )

    # --------------------------------------------------------
    # Add particle contribution
    # --------------------------------------------------------

    image[
        iy0:iy1 + 1,
        ix0:ix1 + 1
    ] += bi * kernel


# ============================================================
# NORMALIZE
# ============================================================

maximum = np.max(image)

if maximum > 0:
    image /= maximum


# ============================================================
# SAVE PNG
# ============================================================

print("Saving:", OUTPUT_FILE)

plt.figure(
    figsize=(8, 8),
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
    cmap="gray",
    interpolation="bilinear",
    vmin=0,
    vmax=1
)

plt.axis("off")

plt.tight_layout(pad=0)

plt.savefig(
    OUTPUT_FILE,
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0,
    facecolor="black"
)

plt.close()

print("Done.")



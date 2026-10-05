import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from amuse.io import read_set_from_file
from amuse.units import units

SNAPSHOT_DIR = "fi_snapshots"
OUTPUT_DIR = "surface_plots"

NPIX = 512
RMAX = 10.0

SURFACE_RADIUS = 0.5

KERNEL_RADIUS = 2.5
GAUSSIAN_WIDTH = 0.5

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

files = sorted(
    glob.glob(
        f"../../../{SNAPSHOT_DIR}/binary_*.amuse"
    ),
    key=lambda x: int(
        os.path.basename(x)
        .replace("binary_", "")
        .replace(".amuse", "")
    )
)

print(
    "Number of snapshots =",
    len(files)
)


def get_surface(parts):

    x = parts.x.value_in(
        units.RSun
    )

    y = parts.y.value_in(
        units.RSun
    )

    z = parts.z.value_in(
        units.RSun
    )

    h = parts.h_smooth.value_in(
        units.RSun
    )

    mass = parts.mass.value_in(
        units.kg
    )

    rho = parts.rho.value_in(
        units.kg / units.m**3
    )

    xy = np.column_stack(
        (x, y)
    )

    tree = cKDTree(xy)

    neighbours = tree.query_ball_point(
        xy,
        SURFACE_RADIUS * h,
        workers=-1
    )

    surface = np.ones(
        len(parts),
        dtype=bool
    )

    for i, ind in enumerate(neighbours):

        ind = np.asarray(
            ind,
            dtype=int
        )

        if np.any(
            z[ind] > z[i]
        ):
            surface[i] = False

    brightness = mass * rho

    return (
        x[surface],
        y[surface],
        h[surface],
        brightness[surface]
    )


def make_surface_image(
    x,
    y,
    h,
    brightness
):

    image = np.zeros(
        (NPIX, NPIX),
        dtype=np.float64
    )

    pixel_size = (
        2.0 * RMAX
    ) / NPIX

    xmin = -RMAX
    ymin = -RMAX

    for i in range(len(x)):

        xi = x[i]
        yi = y[i]
        hi = h[i]
        bi = brightness[i]

        if hi <= 0:
            continue

        sigma = (
            GAUSSIAN_WIDTH * hi
        )

        radius = (
            KERNEL_RADIUS * hi
        )

        radius_pixels = int(
            np.ceil(
                radius / pixel_size
            )
        )

        px = (
            xi - xmin
        ) / pixel_size

        py = (
            yi - ymin
        ) / pixel_size

        ix = int(px)
        iy = int(py)

        ix0 = max(
            0,
            ix - radius_pixels
        )

        ix1 = min(
            NPIX - 1,
            ix + radius_pixels
        )

        iy0 = max(
            0,
            iy - radius_pixels
        )

        iy1 = min(
            NPIX - 1,
            iy + radius_pixels
        )

        if ix0 > ix1 or iy0 > iy1:
            continue

        xx = (
            np.arange(
                ix0,
                ix1 + 1
            ) + 0.5
        ) * pixel_size + xmin

        yy = (
            np.arange(
                iy0,
                iy1 + 1
            ) + 0.5
        ) * pixel_size + ymin

        XX, YY = np.meshgrid(
            xx,
            yy
        )

        r2 = (
            (XX - xi)**2 +
            (YY - yi)**2
        )

        kernel = np.exp(
            -r2 /
            (2.0 * sigma**2)
        )

        image[
            iy0:iy1 + 1,
            ix0:ix1 + 1
        ] += (
            bi * kernel
        )

    maximum = np.max(image)

    if maximum > 0:
        image /= maximum

    return image


for i, filename in enumerate(files):

    print()
    print(
        f"Processing "
        f"{i+1}/{len(files)}"
    )

    print(filename)

    parts = read_set_from_file(
        filename,
        "amuse"
    )

    xsurf, ysurf, hsurf, brightness = get_surface(
        parts
    )

    print(
        "Surface particles =",
        len(xsurf)
    )

    surface_image = make_surface_image(
        xsurf,
        ysurf,
        hsurf,
        brightness
    )

    filename_only = os.path.basename(
        filename
    )

    time_string = (
        filename_only
        .replace(
            "binary_",
            ""
        )
        .replace(
            ".amuse",
            ""
        )
    )

    time_value = int(
        time_string
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.imshow(
        surface_image,
        origin="lower",
        extent=[
            -RMAX,
            RMAX,
            -RMAX,
            RMAX
        ],
        cmap="inferno",
        interpolation="bilinear",
        vmin=0,
        vmax=1
    )

    ax.set_xlim(
        -RMAX/2,
        RMAX/2
    )

    ax.set_ylim(
        -RMAX/2,
        RMAX/2
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        r"$x\ (R_\odot)$"
    )

    ax.set_ylabel(
        r"$y\ (R_\odot)$"
    )

    ax.set_title(
        f"Binary surface   "
        f"$t = {time_value}$ s"
    )

    plt.tight_layout()

    output_file = (
        f"{OUTPUT_DIR}/"
        f"surface_{i:03d}.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_file
    )

print()
print(
    "=============================="
)
print(
    "All surface plots saved."
)
print(
    "Output directory:",
    OUTPUT_DIR
)
print(
    "=============================="
)

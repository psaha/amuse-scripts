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
OUTPUT_DIR = "surface_interferometry"

NPIX = 512
RMAX = 10.0
UVMAX = 1.5

SURFACE_RADIUS = 0.5

KERNEL_RADIUS = 2.5
GAUSSIAN_WIDTH = 0.5

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

files = sorted(
    glob.glob(
        f"{SNAPSHOT_DIR}/binary_*.amuse"
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

    xmin = -RMAX
    ymin = -RMAX

    pixel_size = (
        2.0 * RMAX
    ) / NPIX

    for i in range(len(x)):

        xi = x[i]
        yi = y[i]
        hi = h[i]
        bi = brightness[i]

        if not np.isfinite(xi):
            continue

        if not np.isfinite(yi):
            continue

        if not np.isfinite(hi):
            continue

        if not np.isfinite(bi):
            continue

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

        if ix0 > ix1:
            continue

        if iy0 > iy1:
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
            (2.0 * sigma * sigma)
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


def make_visibility(image):

    ft = np.fft.fftshift(
        np.fft.fft2(image)
    )

    center = NPIX // 2

    dc = np.abs(
        ft[center, center]
    )

    if dc > 0:

        visibility = (
            np.abs(ft) / dc
        )

    else:

        visibility = np.zeros_like(
            ft
        )

    power = visibility**2

    dx = (
        2.0 * RMAX
    ) / NPIX

    freq = np.fft.fftshift(
        np.fft.fftfreq(
            NPIX,
            d=dx
        )
    )

    return (
        power,
        freq
    )


for i, filename in enumerate(files):

    print()
    print(
        f"Processing snapshot "
        f"{i+1}/{len(files)}"
    )

    print(filename)

    parts = read_set_from_file(
        filename,
        "amuse"
    )

    x_all = parts.x.value_in(
        units.RSun
    )

    y_all = parts.y.value_in(
        units.RSun
    )

    rho_all = parts.rho.value_in(
        units.kg / units.m**3
    )

    (
        xsurf,
        ysurf,
        hsurf,
        brightness
    ) = get_surface(parts)

    print(
        "Total particles =",
        len(x_all)
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

    power, freq = make_visibility(
        surface_image
    )

    uv_mask = (
        np.abs(freq) <= UVMAX
    )

    uv = power[
        np.ix_(
            uv_mask,
            uv_mask
        )
    ]

    uvfreq = freq[
        uv_mask
    ]

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

    fig, (
        ax0,
        ax1,
        ax2
    ) = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        facecolor="black"
    )

    ax0.set_facecolor("black")
    ax1.set_facecolor("black")
    ax2.set_facecolor("black")

    ax0.scatter(
        x_all,
        y_all,
        s=1,
        c=rho_all,
        cmap="inferno",
        marker=".",
        linewidths=0
    )

    ax0.set_xlim(
        -RMAX / 2,
        RMAX / 2
    )

    ax0.set_ylim(
        -RMAX / 2,
        RMAX / 2
    )

    ax0.set_aspect(
        "equal"
    )

    ax0.set_xlabel(
        r"$x\ (R_\odot)$",
        color="white"
    )

    ax0.set_ylabel(
        r"$y\ (R_\odot)$",
        color="white"
    )

    ax0.set_title(
        "SPH particles from AMUSE",
        color="white"
    )

    ax0.tick_params(
        colors="white"
    )

    for spine in ax0.spines.values():
        spine.set_color("white")

    ax1.imshow(
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

    ax1.set_xlim(
        -RMAX / 2,
        RMAX / 2
    )

    ax1.set_ylim(
        -RMAX / 2,
        RMAX / 2
    )

    ax1.set_aspect(
        "equal"
    )

    ax1.set_xlabel(
        r"$x\ (R_\odot)$",
        color="white"
    )

    ax1.set_ylabel(
        r"$y\ (R_\odot)$",
        color="white"
    )

    ax1.set_title(
        "Projected surface",
        color="white"
    )

    ax1.tick_params(
        colors="white"
    )

    for spine in ax1.spines.values():
        spine.set_color("white")

    ax2.imshow(
        uv,
        origin="lower",
        extent=[
            uvfreq[0],
            uvfreq[-1],
            uvfreq[0],
            uvfreq[-1]
        ],
        cmap="inferno",
        interpolation="bilinear",
        vmin=0,
        vmax=1,
        aspect="equal"
    )

    ax2.set_xlabel(
        r"$u\ ({\rm cycles}/R_\odot)$",
        color="white"
    )

    ax2.set_ylabel(
        r"$v\ ({\rm cycles}/R_\odot)$",
        color="white"
    )

    ax2.set_title(
        r"Normalized interferometric signal $|V(u,v)|^2$",
        color="white"
    )

    ax2.tick_params(
        colors="white"
    )

    for spine in ax2.spines.values():
        spine.set_color("white")

    fig.suptitle(
        f"Binary SPH evolution   "
        f"$t = {time_value}$ s",
        fontsize=15,
        color="white"
    )

    plt.tight_layout()

    output_file = (
        f"{OUTPUT_DIR}/"
        f"surface_visibility_"
        f"{i:03d}.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
        facecolor="black"
    )

    plt.close()

    print(
        "Saved:",
        output_file
    )


print()
print(
    "==================================="
)
print(
    "All plots saved."
)
print(
    "Output directory:",
    OUTPUT_DIR
)
print(
    "==================================="
)

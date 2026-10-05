import os
import glob
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

from amuse.io import read_set_from_file
from amuse.units import units


# ============================================================
# DIRECTORIES
# ============================================================

SNAPSHOT_DIR = "fi_snapshots"

OUTPUT_DIR = "surface_interferometry"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PLOT PARAMETERS
# ============================================================

NPIX = 512

RMAX = 10.0


# ============================================================
# INTERFEROMETER PARAMETERS
# ============================================================

# Distance to the binary
#
# Change this to the actual distance of your system.
#
DISTANCE_PC = 80.0


# Observing wavelength
#
# 550 nm = 5.5e-7 m
#
# Change this to your observing wavelength.
#
WAVELENGTH = 550e-9


# Maximum physical baseline shown in the (u,v) plane
#
# The binary separation is several R_sun, so we allow
# a large baseline range to see the visibility fringes.
#
UVMAX_METERS = 1000.0


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

RSUN_M = 6.957e8

PC_M = 3.085677581491367e16

DISTANCE_M = (
    DISTANCE_PC
    * PC_M
)


# ============================================================
# SURFACE PARAMETERS
# ============================================================

SURFACE_RADIUS = 0.5


# ============================================================
# SURFACE BRIGHTNESS KERNEL
# ============================================================

KERNEL_RADIUS = 2.5

GAUSSIAN_WIDTH = 0.5


# ============================================================
# FIND SNAPSHOTS
# ============================================================

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


print()
print(
    "Number of snapshots =",
    len(files)
)

print()
print(
    "=============================================="
)

print(
    "INTERFEROMETRIC PARAMETERS"
)

print(
    "=============================================="
)

print(
    "Distance       =",
    DISTANCE_PC,
    "pc"
)

print(
    "Wavelength     =",
    WAVELENGTH,
    "m"
)

print(
    "Maximum baseline =",
    UVMAX_METERS,
    "m"
)

print(
    "=============================================="
)


# ============================================================
# FIND SURFACE PARTICLES
# ============================================================

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

    # --------------------------------------------------------
    # 2-D projected particle positions
    # --------------------------------------------------------

    xy = np.column_stack(
        (
            x,
            y
        )
    )

    tree = cKDTree(
        xy
    )

    neighbours = tree.query_ball_point(
        xy,
        SURFACE_RADIUS * h,
        workers=-1
    )

    surface = np.ones(
        len(parts),
        dtype=bool
    )

    # --------------------------------------------------------
    # Surface determination
    #
    # A particle is considered visible if there is no
    # neighbouring particle located at larger z.
    # --------------------------------------------------------

    for i, ind in enumerate(neighbours):

        ind = np.asarray(
            ind,
            dtype=int
        )

        if np.any(
            z[ind] > z[i]
        ):

            surface[i] = False

    # --------------------------------------------------------
    # Surface brightness
    # --------------------------------------------------------

    brightness = (
        mass
        * rho
    )

    return (
        x[surface],
        y[surface],
        h[surface],
        brightness[surface]
    )


# ============================================================
# MAKE SURFACE IMAGE
# ============================================================

def make_surface_image(
    x,
    y,
    h,
    brightness
):

    image = np.zeros(
        (
            NPIX,
            NPIX
        ),
        dtype=np.float64
    )

    xmin = -RMAX

    ymin = -RMAX

    pixel_size = (
        2.0 * RMAX
    ) / NPIX

    # --------------------------------------------------------
    # Add every surface particle
    # --------------------------------------------------------

    for i in range(len(x)):

        xi = x[i]

        yi = y[i]

        hi = h[i]

        bi = brightness[i]

        # ----------------------------------------------------
        # Check finite values
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Gaussian kernel
        # ----------------------------------------------------

        sigma = (
            GAUSSIAN_WIDTH
            * hi
        )

        radius = (
            KERNEL_RADIUS
            * hi
        )

        radius_pixels = int(
            np.ceil(
                radius
                / pixel_size
            )
        )

        # ----------------------------------------------------
        # Particle pixel
        # ----------------------------------------------------

        px = (
            xi - xmin
        ) / pixel_size

        py = (
            yi - ymin
        ) / pixel_size

        ix = int(px)

        iy = int(py)

        # ----------------------------------------------------
        # Kernel boundaries
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Physical coordinates of pixels
        # ----------------------------------------------------

        xx = (
            np.arange(
                ix0,
                ix1 + 1
            )
            + 0.5
        ) * pixel_size + xmin

        yy = (
            np.arange(
                iy0,
                iy1 + 1
            )
            + 0.5
        ) * pixel_size + ymin

        XX, YY = np.meshgrid(
            xx,
            yy
        )

        # ----------------------------------------------------
        # Distance from particle
        # ----------------------------------------------------

        r2 = (
            (XX - xi)**2
            +
            (YY - yi)**2
        )

        # ----------------------------------------------------
        # Gaussian surface kernel
        # ----------------------------------------------------

        kernel = np.exp(
            -r2
            /
            (
                2.0
                * sigma
                * sigma
            )
        )

        # ----------------------------------------------------
        # Add brightness
        # ----------------------------------------------------

        image[
            iy0:iy1 + 1,
            ix0:ix1 + 1
        ] += (
            bi
            * kernel
        )

    # ========================================================
    # NORMALIZE IMAGE
    # ========================================================

    maximum = np.max(
        image
    )

    if maximum > 0:

        image /= maximum

    return image


# ============================================================
# VISIBILITY
# ============================================================

def make_visibility(image):

    # ========================================================
    # FOURIER TRANSFORM
    # ========================================================

    ft = np.fft.fftshift(
        np.fft.fft2(
            image
        )
    )

    # ========================================================
    # ZERO FREQUENCY
    # ========================================================

    center = NPIX // 2

    dc = np.abs(
        ft[
            center,
            center
        ]
    )

    # ========================================================
    # NORMALIZED VISIBILITY |V|
    # ========================================================

    if dc > 0:

        visibility = (
            np.abs(ft)
            / dc
        )

    else:

        visibility = np.zeros_like(
            ft
        )

    # --------------------------------------------------------
    # Numerical protection
    # --------------------------------------------------------

    visibility = np.clip(
        visibility,
        0.0,
        1.0
    )

    # ========================================================
    # FFT SPATIAL FREQUENCY
    # ========================================================

    dx = (
        2.0 * RMAX
    ) / NPIX

    freq = np.fft.fftshift(
        np.fft.fftfreq(
            NPIX,
            d=dx
        )
    )

    # freq units:
    #
    # cycles / R_sun
    #
    # Convert this to physical baseline:
    #
    # B = f * R_sun * lambda / D
    #
    # where:
    #
    # f      = cycles / R_sun
    # R_sun  = meters
    # lambda = observing wavelength
    # D      = source distance

    baseline_meter = freq * WAVELENGTH * DISTANCE_M / RSUN_M

    return (
        visibility,
        freq,
        baseline_meter
    )


# ============================================================
# PROCESS SNAPSHOTS
# ============================================================

for i, filename in enumerate(files):

    print()
    print(
        "=============================================="
    )

    print(
        f"Processing snapshot "
        f"{i + 1}/{len(files)}"
    )

    print(
        filename
    )

    # ========================================================
    # READ SNAPSHOT
    # ========================================================

    parts = read_set_from_file(
        filename,
        "amuse"
    )

    # ========================================================
    # ALL PARTICLES
    # ========================================================

    x_all = parts.x.value_in(
        units.RSun
    )

    y_all = parts.y.value_in(
        units.RSun
    )

    rho_all = parts.rho.value_in(
        units.kg / units.m**3
    )

    # ========================================================
    # SURFACE PARTICLES
    # ========================================================

    (
        xsurf,
        ysurf,
        hsurf,
        brightness
    ) = get_surface(
        parts
    )

    print(
        "Total particles =",
        len(x_all)
    )

    print(
        "Surface particles =",
        len(xsurf)
    )

    # ========================================================
    # SURFACE IMAGE
    # ========================================================

    surface_image = make_surface_image(
        xsurf,
        ysurf,
        hsurf,
        brightness
    )

    # ========================================================
    # VISIBILITY
    # ========================================================

    (
        visibility,
        freq,
        baseline_meter
    ) = make_visibility(
        surface_image
    )

    # ========================================================
    # UV MASK IN PHYSICAL METERS
    # ========================================================

    uv_mask = (
        np.abs(
            baseline_meter
        )
        <= UVMAX_METERS
    )

    uv = visibility[
        np.ix_(
            uv_mask,
            uv_mask
        )
    ]

    uvbaseline = baseline_meter[
        uv_mask
    ]

    # ========================================================
    # PRINT BASELINE INFORMATION
    # ========================================================

    print(
        "Full FFT baseline range = "
        f"{baseline_meter[0]:.3f} to "
        f"{baseline_meter[-1]:.3f} m"
    )

    print(
        "Displayed baseline range = "
        f"{uvbaseline[0]:.3f} to "
        f"{uvbaseline[-1]:.3f} m"
    )

    # ========================================================
    # TIME
    # ========================================================

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

    # ========================================================
    # FIGURE
    # ========================================================

    fig, (
        ax0,
        ax1,
        ax2
    ) = plt.subplots(
        1,
        3,
        figsize=(19, 6),
        facecolor="black"
    )

    # ========================================================
    # SAME BACKGROUND
    # ========================================================

    ax0.set_facecolor(
        "black"
    )

    ax1.set_facecolor(
        "black"
    )

    ax2.set_facecolor(
        "black"
    )

    # ========================================================
    # PANEL 1
    # ALL SPH PARTICLES
    # ========================================================

    scatter = ax0.scatter(
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

        spine.set_color(
            "white"
        )

    # ========================================================
    # PANEL 2
    # PROJECTED SURFACE
    # ========================================================

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

        spine.set_color(
            "white"
        )

    # ========================================================
    # PANEL 3
    # PHYSICAL (u,v) BASELINE PLANE
    # ========================================================

    im_uv = ax2.imshow(
        uv,
        origin="lower",
        extent=[
            uvbaseline[0],
            uvbaseline[-1],
            uvbaseline[0],
            uvbaseline[-1]
        ],
        cmap="inferno",
        interpolation="bilinear",
        vmin=0,
        vmax=1,
        aspect="equal"
    )

    # --------------------------------------------------------
    # Baseline axes
    # --------------------------------------------------------

    ax2.set_xlabel(
        r"$u\ ({\rm m})$",
        color="white"
    )

    ax2.set_ylabel(
        r"$v\ ({\rm m})$",
        color="white"
    )

    ax2.set_title(
        r"$|V(u,v)|$",
        color="white"
    )

    ax2.tick_params(
        colors="white"
    )

    for spine in ax2.spines.values():

        spine.set_color(
            "white"
        )

    # ========================================================
    # VISIBILITY COLORBAR
    # ========================================================

    cbar = fig.colorbar(
        im_uv,
        ax=ax2,
        fraction=0.046,
        pad=0.04
    )

    cbar.set_label(
        r"$|V|$",
        color="white",
        fontsize=11
    )

    cbar.ax.tick_params(
        colors="white"
    )

    cbar.outline.set_edgecolor(
        "white"
    )

    # ========================================================
    # FIGURE TITLE
    # ========================================================

    fig.suptitle(
        f"Binary SPH evolution   "
        f"$t = {time_value}$ s   "
        f"$D = {DISTANCE_PC}$ pc   "
        f"$\\lambda = {WAVELENGTH*1e9:.0f}$ nm",
        fontsize=15,
        color="white"
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94
        ]
    )

    # ========================================================
    # SAVE
    # ========================================================

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


# ============================================================
# FINISHED
# ============================================================

print()

print(
    "=============================================="
)

print(
    "All plots saved."
)

print(
    "Output directory:",
    OUTPUT_DIR
)

print(
    "=============================================="
)

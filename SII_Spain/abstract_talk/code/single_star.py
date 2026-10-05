import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import os


# ============================================================
# Gaussian SPH smoothing kernel
# ============================================================

def W(x, y, z, h):

    r = np.sqrt(x**2 + y**2 + z**2)

    w = (
        1.0 / (h * np.sqrt(np.pi))
    )**3 * np.exp(-(r**2) / h**2)

    return w


# ============================================================
# Gradient of Gaussian smoothing kernel
# ============================================================

def gradW(x, y, z, h):

    r = np.sqrt(x**2 + y**2 + z**2)

    n = (
        -2.0
        * np.exp(-(r**2) / h**2)
        / h**5
        / np.pi**(3 / 2)
    )

    wx = n * x
    wy = n * y
    wz = n * z

    return wx, wy, wz


# ============================================================
# Pairwise separations
# ============================================================

def getPairwiseSeparations(ri, rj):

    M = ri.shape[0]
    N = rj.shape[0]

    rix = ri[:, 0].reshape((M, 1))
    riy = ri[:, 1].reshape((M, 1))
    riz = ri[:, 2].reshape((M, 1))

    rjx = rj[:, 0].reshape((N, 1))
    rjy = rj[:, 1].reshape((N, 1))
    rjz = rj[:, 2].reshape((N, 1))

    dx = rix - rjx.T
    dy = riy - rjy.T
    dz = riz - rjz.T

    return dx, dy, dz


# ============================================================
# Density
# ============================================================

def getDensity(r, pos, m, h):

    M = r.shape[0]

    dx, dy, dz = getPairwiseSeparations(
        r,
        pos
    )

    rho = np.sum(
        m * W(dx, dy, dz, h),
        axis=1
    ).reshape((M, 1))

    return rho


# ============================================================
# Equation of state
# ============================================================

def getPressure(rho, k, n):

    return k * rho ** (1 + 1 / n)


# ============================================================
# Acceleration
# ============================================================

def getAcc(
    pos,
    vel,
    m,
    h,
    k,
    n,
    lmbda,
    nu
):

    N = pos.shape[0]

    # --------------------------------------------------------
    # Density
    # --------------------------------------------------------

    rho = getDensity(
        pos,
        pos,
        m,
        h
    )

    # --------------------------------------------------------
    # Pressure
    # --------------------------------------------------------

    P = getPressure(
        rho,
        k,
        n
    )

    # --------------------------------------------------------
    # Pairwise separations
    # --------------------------------------------------------

    dx, dy, dz = getPairwiseSeparations(
        pos,
        pos
    )

    # --------------------------------------------------------
    # Kernel gradients
    # --------------------------------------------------------

    dWx, dWy, dWz = gradW(
        dx,
        dy,
        dz,
        h
    )

    # --------------------------------------------------------
    # Pressure acceleration
    # --------------------------------------------------------

    pressure_term = (
        P / rho**2
        +
        P.T / rho.T**2
    )

    ax = -np.sum(
        m * pressure_term * dWx,
        axis=1
    ).reshape((N, 1))

    ay = -np.sum(
        m * pressure_term * dWy,
        axis=1
    ).reshape((N, 1))

    az = -np.sum(
        m * pressure_term * dWz,
        axis=1
    ).reshape((N, 1))

    a = np.hstack(
        (ax, ay, az)
    )

    # --------------------------------------------------------
    # Restoring force
    #
    # This represents the force that tries to bring the
    # perturbed star back toward equilibrium.
    # --------------------------------------------------------

    a -= lmbda * pos

    # --------------------------------------------------------
    # Damping
    #
    # This causes the oscillations to gradually disappear.
    # --------------------------------------------------------

    a -= nu * vel

    return a


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Simulation parameters
    # ========================================================

    N = 5000

    t = 0.0

    tEnd = 12.0

    dt = 0.04

    # --------------------------------------------------------
    # Stellar parameters
    # --------------------------------------------------------

    M = 2.0

    R = 0.75

    # --------------------------------------------------------
    # SPH parameters
    # --------------------------------------------------------

    h = 0.1

    k = 0.1

    n = 1.0

    # --------------------------------------------------------
    # Damping
    #
    # Larger value -> faster settling
    # --------------------------------------------------------

    nu = 0.8

    # ========================================================
    # Output directory
    # ========================================================

    frame_dir = "frames_star"

    os.makedirs(
        frame_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save one frame every N timesteps
    # --------------------------------------------------------

    frame_interval = 5

    # ========================================================
    # Initial particle distribution
    # ========================================================

    np.random.seed(42)

    # --------------------------------------------------------
    # Start with particles distributed around the origin
    # --------------------------------------------------------

    pos = np.random.randn(
        N,
        3
    )

    # --------------------------------------------------------
    # Normalize the particle distribution
    # so that it initially represents a star of radius R.
    # --------------------------------------------------------

    radius = np.sqrt(
        np.sum(pos**2, axis=1)
    )

    pos *= (
        0.70 * R
        / np.max(radius)
    )

    # ========================================================
    # Initial velocity
    # ========================================================

    vel = np.zeros(
        (N, 3)
    )

    # --------------------------------------------------------
    # Radial perturbation
    #
    # This is what makes the star oscillate.
    # --------------------------------------------------------

    perturbation = 0.45

    radius = np.sqrt(
        np.sum(pos**2, axis=1)
    )

    radial_direction = np.zeros_like(pos)

    mask = radius > 0

    radial_direction[mask] = (
        pos[mask]
        /
        radius[mask, None]
    )

    vel = (
        perturbation
        * radius[:, None]
        * radial_direction
    )

    # ========================================================
    # External restoring force coefficient
    # ========================================================

    lmbda = (
        2
        * k
        * (1 + n)
        * np.pi ** (-3 / (2 * n))
        * (
            M
            * gamma(5 / 2 + n)
            / R**3
            / gamma(1 + n)
        ) ** (1 / n)
        / R**2
    )

    # ========================================================
    # Particle mass
    # ========================================================

    m = M / N

    # ========================================================
    # Initial acceleration
    # ========================================================

    acc = getAcc(
        pos,
        vel,
        m,
        h,
        k,
        n,
        lmbda,
        nu
    )

    # ========================================================
    # Number of timesteps
    # ========================================================

    Nt = int(
        np.ceil(tEnd / dt)
    )

    # ========================================================
    # Simulation
    # ========================================================

    for i in range(Nt):

        # ----------------------------------------------------
        # Half kick
        # ----------------------------------------------------

        vel += (
            0.5
            * acc
            * dt
        )

        # ----------------------------------------------------
        # Drift
        # ----------------------------------------------------

        pos += (
            vel
            * dt
        )

        # ----------------------------------------------------
        # New acceleration
        # ----------------------------------------------------

        acc = getAcc(
            pos,
            vel,
            m,
            h,
            k,
            n,
            lmbda,
            nu
        )

        # ----------------------------------------------------
        # Half kick
        # ----------------------------------------------------

        vel += (
            0.5
            * acc
            * dt
        )

        # ----------------------------------------------------
        # Update time
        # ----------------------------------------------------

        t += dt

        # ====================================================
        # Save frame
        # ====================================================

        if i % frame_interval == 0:

            fig, ax = plt.subplots(
                figsize=(6, 6),
                dpi=150
            )
            
            # --------------------------------------------------------
            # Black background
            # --------------------------------------------------------

            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")


            # ------------------------------------------------
            # Projected radius
            #
            # Used ONLY to assign particle colors.
            # It is NOT density.
            # ------------------------------------------------

            particle_radius = np.sqrt(
                pos[:, 0]**2
                +
                pos[:, 1]**2
            )

            # ------------------------------------------------
            # Particle plot
            # ------------------------------------------------

            scatter = ax.scatter(
                pos[:, 0],
                pos[:, 1],
                c=particle_radius,
                cmap="plasma",
                s=8,
                alpha=0.80,
                edgecolors="none"
            )

            # ------------------------------------------------
            # Equilibrium radius
            #
            # Dashed circle shows the reference equilibrium
            # size of the star.
            # ------------------------------------------------

            circle = plt.Circle((0, 0), R, fill=False, linestyle="--", linewidth=1.2, color="white", alpha=0.45)

            ax.add_patch(
                circle
            )

            # --------------------------------------------------------
            # Figure limits
            # --------------------------------------------------------

            ax.set_xlim(-1.2, 1.2)

            ax.set_ylim(-1.2, 1.2)

            ax.set_aspect("equal", "box")

            # --------------------------------------------------------
            # Axis labels
            # --------------------------------------------------------

            ax.set_xlabel(r"$x$", fontsize=14, color="white")

            ax.set_ylabel(r"$y$", fontsize=14, color="white")

            # --------------------------------------------------------
            # Title
            # --------------------------------------------------------

            ax.set_title(f"Stellar Equilibrium    $t = {t:.2f}$", fontsize=15, color="white", pad=12)

            # --------------------------------------------------------
            # White tick labels
            # --------------------------------------------------------

            ax.tick_params(axis="both",colors="white",labelsize=10)

            # --------------------------------------------------------
            # White spines
            # --------------------------------------------------------

            for spine in ax.spines.values():
               spine.set_color("white")
               spine.set_alpha(0.4)

            # --------------------------------------------------------
            # Save
            # --------------------------------------------------------

            filename = os.path.join(
                 frame_dir,
                 f"frame_{i:05d}.png"
                 )

            plt.savefig(filename, dpi=200,bbox_inches="tight",facecolor="black")

            plt.close(fig)
  
            print(f"Saved: {filename}")

            # ========================================================
            # Finished
            # ========================================================

            print()
            print("Simulation finished.")
 
            print(f"5000 particles were used.")

            print(f"Frames are saved in: {frame_dir}/")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np

# USER PARAMETERS
INPUT_FILE = "amuse_file/visible_surface_particles_binary.csv"
OUTPUT_VISIBILITY = "plot/visibility_amplitude_binary.png"
OUTPUT_POWER = "plot/visibility_power_binary.png"

N_UV = 501                       # Fourier-plane resolution
U_MAX = 1.5                      # Maximum spatial frequency in cycles / R_sun.

# Gaussian SPH smoothing
GAUSSIAN_WIDTH = 0.5
USE_SMOOTHING = True
PLOT_FLOOR = 1e-6

# READ PARTICLE CATALOGUE
print("Reading:", INPUT_FILE)
data = np.genfromtxt(INPUT_FILE, delimiter=",", names=True)
x = np.asarray(data["x_Rsun"], dtype=float)
y = np.asarray(data["y_Rsun"], dtype=float)
h = np.asarray(data["h_Rsun"], dtype=float)
brightness = np.asarray(data["brightness"], dtype=float)

# REMOVE INVALID PARTICLES
mask = (np.isfinite(x) & np.isfinite(y) & np.isfinite(h) & np.isfinite(brightness) & (h > 0) & (brightness > 0))
x = x[mask]
y = y[mask]
h = h[mask]
brightness = brightness[mask]
print("Particles used:", len(x))

# NORMALIZE PARTICLE FLUX
total_flux = np.sum(brightness)
if total_flux <= 0:
    raise RuntimeError("Total brightness is zero.")

weights = brightness / total_flux
print("Total particle brightness:", total_flux)

# CREATE (u,v) PLANE
u = np.linspace(-U_MAX, U_MAX, N_UV)
v = np.linspace(-U_MAX, U_MAX, N_UV)
U, V = np.meshgrid(u, v)

# CALCULATE COMPLEX VISIBILITY
visibility = np.zeros((N_UV, N_UV), dtype=np.complex128)
print("Calculating visibility...")
print("Fourier grid:", N_UV, "x", N_UV)

for i in range(len(x)):
    phase = (2.0 * np.pi * (U * x[i] + V * y[i]))
    contribution = np.exp(-1j * phase)

    if USE_SMOOTHING:
        sigma = GAUSSIAN_WIDTH * h[i]
        smoothing_factor = np.exp(-2.0 * np.pi**2 * sigma**2 * (U**2 + V**2))
        contribution *= smoothing_factor

    visibility += (weights[i] * contribution)

V0 = visibility[N_UV // 2, N_UV // 2]
print()
print("Visibility at (u,v) = (0,0):")
print(V0)
visibility /= V0                                               # Normalize explicitly.
visibility_amplitude = np.abs(visibility)                      # VISIBILITY AMPLITUDE
visibility_amplitude = np.clip(visibility_amplitude, 0.0, 1.0) # Numerical errors can produce tiny values slightly > 1.
visibility_power = (visibility_amplitude ** 2)
visibility_power = np.clip(visibility_power, 0.0, 1.0)         # POWER SPECTRUM

uv_radius = np.sqrt(U**2 + V**2)                               # Spatial frequency radius
n_bins = 150                                                   # Radial bins
radial_bins = np.linspace(0, U_MAX, n_bins + 1)
radial_uv = []
radial_visibility = []
radial_power = []

for i in range(n_bins):
    rmin = radial_bins[i]
    rmax = radial_bins[i + 1]
    mask = ((uv_radius >= rmin) & (uv_radius < rmax))
    if np.any(mask):
        radial_uv.append(np.mean(uv_radius[mask]))
        radial_visibility.append(np.mean(visibility_amplitude[mask]))
        radial_power.append(np.mean(visibility_power[mask]))

radial_uv = np.asarray(radial_uv)
radial_visibility = np.asarray(radial_visibility)
radial_power = np.asarray(radial_power)

# PRINT SOME BASIC INFORMATION
print()
print("Visibility statistics")
print("---------------------")
print("V(0,0) amplitude:", visibility_amplitude[N_UV // 2, N_UV // 2])
print("Maximum amplitude:", np.max(visibility_amplitude))
print("Minimum amplitude:", np.min(visibility_amplitude))

# PLOT 1: VISIBILITY AMPLITUDE -- LINEAR SCALE
plt.figure(figsize=(8, 7))
plt.imshow(visibility_amplitude, origin="lower", extent=[-U_MAX, U_MAX, -U_MAX, U_MAX],
    cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
plt.xlabel(r"$u$  [cycles / $R_\odot$]")
plt.ylabel(r"$v$  [cycles / $R_\odot$]")
plt.title(r"Normalized interferometric visibility $|V(u,v)|$")
plt.colorbar(label=r"$|V|$")
plt.axis("equal")
plt.tight_layout()
plt.savefig(OUTPUT_VISIBILITY, dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# PLOT 2: POWER SPECTRUM -- LINEAR SCALE
plt.figure(figsize=(8, 7))
plt.imshow(visibility_power, origin="lower", extent=[-U_MAX, U_MAX, -U_MAX, U_MAX],
    cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
plt.xlabel(r"$u$  [cycles / $R_\odot$]")
plt.ylabel(r"$v$  [cycles / $R_\odot$]")
plt.title(r"Normalized spatial power spectrum $|V(u,v)|^2$")
plt.colorbar(label=r"$|V|^2$")
plt.axis("equal")
plt.tight_layout()
plt.savefig(OUTPUT_POWER, dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# RADIAL VISIBILITY -- LINEAR Y SCALE
plt.figure(figsize=(8, 6))
plt.plot(radial_uv, radial_visibility, linewidth=2, label=r"$|V|$")
plt.plot(radial_uv, radial_power, linewidth=2, label=r"$|V|^2$")
plt.xlabel(r"Spatial frequency $\sqrt{u^2+v^2}$ [cycles / $R_\odot$]")
plt.ylabel("Normalized value")
plt.yscale("linear")
plt.ylim(0.0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("plot/radial_visibility.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()


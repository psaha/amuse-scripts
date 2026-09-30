from amuse.io import read_set_from_file
from amuse.units import units
import numpy as np

# USER PARAMETERS
INPUT_FILE = "amuse_file/polytrope_single_star.amuse"
OUTPUT_FILE = "amuse_file/visible_surface_particles_single.csv"
VISIBILITY_MIN = 0.0001         # Minimum visibility for a particle to be retained. 0.01 means particles with less than 1% estimated visibility are discarded.
KERNEL_RADIUS_FACTOR = 1.0
N_NEIGHBOURS = 64

# LOAD SPH PARTICLES
print("Reading:", INPUT_FILE)
stars = read_set_from_file(INPUT_FILE, "amuse")
N = len(stars)
print("Number of particles:", N)

# Coordinates in units of solar radius.
x = stars.x.value_in(units.RSun)
y = stars.y.value_in(units.RSun)
z = stars.z.value_in(units.RSun)

h = stars.h_smooth.value_in(units.RSun)         # Smoothing length in solar radius.
mass = stars.mass.value_in(units.MSun)          # Physical quantities.
rho = stars.rho.value_in(units.g / units.cm**3)
u = stars.u.value_in(units.m**2 / units.s**2)   # Internal energy in SI.

# INITIALISE VISIBILITY
visibility = np.ones(N)
print("Calculating projected visibility...")

order = np.argsort(z)[::-1]                     # Sort particles from observer toward the back. Largest z first.
for position, idx in enumerate(order):          # Process particles in depth order.
    xi = x[idx]
    yi = y[idx]
    zi = z[idx]
    hi = h[idx]
    foreground = order[:position]
    if len(foreground) == 0:
        continue

    dx = x[foreground] - xi                      # Calculate projected distances to foreground particles
    dy = y[foreground] - yi
    R2 = dx * dx + dy * dy
    hj = h[foreground]                           # Use foreground particles' smoothing lengths.
    radius = KERNEL_RADIUS_FACTOR * hj    
    mask = R2 < radius * radius                  # Only particles whose projected smoothing footprint. Reaches particle i can obscure it. 
    if not np.any(mask):
        continue

    R2_local = R2[mask]
    h_local = hj[mask]

    overlap = np.exp(-0.5 * R2_local / (h_local * h_local))  # Gaussian projected kernel    
    obscuration = np.sum(overlap)                # Limit the total obscuration contribution.
    visibility[idx] = np.exp(-obscuration)       # Convert to transmission/visibility.


# SELECT VISIBLE SURFACE PARTICLES
surface_mask = visibility >= VISIBILITY_MIN
surface_indices = np.where(surface_mask)[0]

print()
print("Visible particles:", len(surface_indices))
print("Fraction retained:", len(surface_indices) / N)

# CREATE SURFACE ARRAYS
surface_x = x[surface_mask]
surface_y = y[surface_mask]
surface_z = z[surface_mask]
surface_h = h[surface_mask]
surface_mass = mass[surface_mask]
surface_rho = rho[surface_mask]
surface_u = u[surface_mask]
surface_visibility = visibility[surface_mask]

# BRIGHTNESS PROXY
brightness = surface_visibility
# SAVE CSV
data = np.column_stack((surface_x, surface_y, surface_z, surface_h, surface_mass, surface_rho, surface_u, surface_visibility, brightness,))
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

np.savetxt(OUTPUT_FILE, data, delimiter=",", header=header, comments="")
print()
print("Saved:", OUTPUT_FILE)

# SUMMARY
print()
print("Surface particle ranges:")
print("--------------------------------")
print("x:", surface_x.min(), "to", surface_x.max(), "Rsun")
print("y:", surface_y.min(), "to", surface_y.max(), "Rsun")
print("z:", surface_z.min(), "to", surface_z.max(), "Rsun")
print("visibility:", surface_visibility.min(), "to", surface_visibility.max())
print("brightness:", brightness.min(), "to", brightness.max())



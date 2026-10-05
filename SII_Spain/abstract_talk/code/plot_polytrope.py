from amuse.io import read_set_from_file
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from amuse.units import units
import numpy as np
import os


# ============================================================
# Read SPH particle data
# ============================================================
parts = read_set_from_file("amuse_file/polytrope_single_star.amuse", "amuse")

N = len(parts)
r = parts.position.lengths().value_in(units.RSun)
v = parts.velocity.lengths().value_in(units.km / units.s)
m = parts.mass.value_in(units.MSun)

print(f"N = {N}")
print(f"max radius   : {r.max():.3f} RSun")
print(f"rms velocity : {np.sqrt((v**2).mean()):.4e} km/s")

# ============================================================
# Measured SPH density profile
# ============================================================
bins = np.linspace(0, 1.2, 40)
mass_in_bin, edges = np.histogram(r, bins=bins, weights=m)

# Spherical shell volume
vol = (4 * np.pi / 3 * (edges[1:]**3 - edges[:-1]**3))
rho_meas = mass_in_bin / vol
rc = 0.5 * (edges[1:] + edges[:-1])

# ============================================================
# Analytic Lane-Emden profile
# n = 1.5
# ============================================================

def rhs(xi, y):
    return [y[1], -max(y[0], 0)**1.5 - 2 * y[1] / xi]

def surf(xi, y):
    return y[0]

surf.terminal = True
surf.direction = -1

# Solve Lane-Emden equation
sol = solve_ivp(rhs, [1e-6, 20], [1, -1e-6 / 3], events=surf, rtol=1e-10, atol=1e-12, dense_output=True)

# ============================================================
# First zero of Lane-Emden solution
# ============================================================

xi1 = sol.t_events[0][0]
dth1 = sol.sol(xi1)[1]

# Central density for M = R = 1
rho_c = (xi1 / (4 * np.pi * abs(dth1)))

# Analytic radial coordinate
xg = np.linspace(1e-6, xi1, 400)

# Analytic density
rho_ana = (rho_c * np.clip(sol.sol(xg)[0], 0, None)**1.5)

# ============================================================
# Print important quantities
# ============================================================

print()
print("---------------------------------------")
print("Lane-Emden n = 1.5")
print("---------------------------------------")
print(f"xi_1          = {xi1:.6f}")
print(f"central density = {rho_c:.6e} MSun/RSun^3")
print(f"maximum SPH radius = {r.max():.4f} RSun")
print("---------------------------------------")

# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 1.2,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


# ============================================================
# Create figure
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

# Black background
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

# ============================================================
# SPH density profile
# ============================================================
ax.semilogy(rc, rho_meas, marker="o", linestyle="", markersize=5, markeredgewidth=1, alpha=0.85, label=r"$\mathbf{SPH\ particles}$")

# ============================================================
# Analytic Lane-Emden profile
# ============================================================
ax.semilogy(xg / xi1, rho_ana, linewidth=2.0, label=r"$\mathbf{Lane\!-\!Emden}\;(n=1.5)$")

# ============================================================
# Central density reference line
# ============================================================
ax.axhline(rho_c, linestyle="--", linewidth=1.5, alpha=0.55)


# ============================================================
# Radius = 1 reference
# ============================================================
ax.axvline(1.0, linestyle=":", linewidth=1.5, alpha=0.6)

# ============================================================
# Labels
# ============================================================
ax.set_xlabel(r"$\mathbf{r/R_{\odot}}$", fontsize=10, fontweight="bold", color="white")
ax.set_ylabel(r"$\mathbf{\rho\ [M_{\odot}/R_{\odot}^{3}]}$", fontsize=10, fontweight="bold", color="white")

# ============================================================
# Title
# ============================================================
ax.set_title(r"$\mathbf{Density\ Profile\ of\ a\ Relaxed\ Polytrope}$" "\n" rf"$\mathbf{{N = {N:,}}}$",
    fontsize=10, fontweight="bold", color="white", pad=15)

# ============================================================
# Grid
# ============================================================
ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.25)

# ============================================================
# Tick formatting
# ============================================================
ax.tick_params(axis="both", which="major", colors="white", width=1.5, length=6)
ax.tick_params(axis="both", which="minor", colors="white", width=1.0, length=3)

# White axis spines
for spine in ax.spines.values():
    spine.set_color("white")
    spine.set_linewidth(1.2)
    
# ============================================================
# Legend
# ============================================================
# Legend
legend = ax.legend(
    frameon=True,
    fontsize=10
)

legend.get_frame().set_facecolor("black")
legend.get_frame().set_edgecolor("white")

for text in legend.get_texts():
    text.set_color("white")

# ============================================================
# Layout
# ============================================================
plt.tight_layout()


# ============================================================
# Create output directory
# ============================================================
os.makedirs("plot", exist_ok=True)

# ============================================================
# Save figure
# ============================================================
output_file = ("plot/polytrope_profile.png")
plt.savefig(output_file, dpi=250, bbox_inches="tight")


# ============================================================
# Display
# ============================================================

plt.show()

print()
print(
    f"Wrote: {output_file}"
)

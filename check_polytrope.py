#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from amuse.io import read_set_from_file
from amuse.units import units

parts = read_set_from_file("polytrope_relaxed.amuse", "amuse")
N = len(parts)

r = parts.position.lengths().value_in(units.RSun)
v = parts.velocity.lengths().value_in(units.km/units.s)
m = parts.mass.value_in(units.MSun)

print(f"N = {N}")
print(f"max radius   : {r.max():.3f} RSun")
print(f"rms velocity : {np.sqrt((v**2).mean()):.4e} km/s")

# --- measured density profile (mass in spherical shells) ---
bins = np.linspace(0, 1.2, 40)
mass_in_bin, edges = np.histogram(r, bins=bins, weights=m)
vol = 4*np.pi/3*(edges[1:]**3 - edges[:-1]**3)          # RSun^3
rho_meas = mass_in_bin/vol                               # MSun/RSun^3
rc = 0.5*(edges[1:] + edges[:-1])

# --- analytic Lane-Emden profile, n = 1.5 ---
def rhs(xi, y): return [y[1], -max(y[0],0)**1.5 - 2*y[1]/xi]
def surf(xi, y): return y[0]
surf.terminal, surf.direction = True, -1
sol = solve_ivp(rhs, [1e-6, 20], [1, -1e-6/3], events=surf,
                rtol=1e-10, atol=1e-12, dense_output=True)
xi1 = sol.t_events[0][0]; dth1 = sol.sol(xi1)[1]
rho_c = xi1/(4*np.pi*abs(dth1))                          # MSun/RSun^3 (M=R=1)
xg = np.linspace(1e-6, xi1, 400)
rho_ana = rho_c*np.clip(sol.sol(xg)[0], 0, None)**1.5

plt.figure(figsize=(7,5))
plt.semilogy(rc, rho_meas, 'o', label='SPH particles')
plt.semilogy(xg/xi1, rho_ana, '-', label='Lane-Emden n=1.5')
plt.xlabel('r / RSun'); plt.ylabel(r'$\rho$ [MSun/RSun$^3$]')
plt.legend(); plt.title('Relaxed polytrope density profile')
plt.tight_layout()
plt.savefig('polytrope_profile.png', dpi=130)
print("Wrote polytrope_profile.png")
plt.show()

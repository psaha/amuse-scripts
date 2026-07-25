#!/usr/bin/env python
"""
Build an n=3/2 polytrope (1 MSun, 1 RSun), relax it in the Fi SPH code,
and save the relaxed model to disk.
"""
import numpy as np
from scipy.integrate import solve_ivp

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles
from amuse.io import write_set_to_file
from amuse.community.fi.interface import Fi

# ----------------------- parameters -----------------------
N       = 50_000            # number of SPH particles
M_star  = 1.0 | units.MSun
R_star  = 1.0 | units.RSun
n_poly  = 1.5               # polytropic index (convective star)
gamma   = 1.0 + 1.0/n_poly  # = 5/3

np.random.seed(42)

# ----------------- 1. Solve Lane-Emden ---------------------
def rhs(xi, y):
    theta, dtheta = y
    th = max(theta, 0.0)
    return [dtheta, -th**n_poly - 2.0*dtheta/xi]

def surface(xi, y):          # event: theta = 0
    return y[0]
surface.terminal = True
surface.direction = -1

# start slightly off-centre with the series expansion
xi0 = 1e-6
sol = solve_ivp(rhs, [xi0, 20.0],
                [1.0 - xi0**2/6.0, -xi0/3.0],
                events=surface, rtol=1e-10, atol=1e-12,
                dense_output=True)

xi1      = sol.t_events[0][0]          # surface: ~3.6538 for n=1.5
dtheta1  = sol.sol(xi1)[1]             # theta'(xi1): ~ -0.2033
print(f"Lane-Emden solved: xi1 = {xi1:.5f}, theta'(xi1) = {dtheta1:.5f}")

# fine grid of the solution
xi  = np.linspace(xi0, xi1, 2000)
th  = np.clip(sol.sol(xi)[0], 0.0, None)
dth = sol.sol(xi)[1]

# ------------- 2. Physical scaling -------------------------
# rho_c = M xi1 / (4 pi R^3 |theta'(xi1)|)
rho_c = M_star*xi1 / (4.0*np.pi*R_star**3*abs(dtheta1))
alpha = R_star/xi1                                  # length scale a
K     = 4.0*np.pi*constants.G*alpha**2*rho_c**(1.0-1.0/n_poly)/(n_poly+1.0)
print("Central density:", rho_c.value_in(units.g/units.cm**3), "g/cm^3")

# ------------- 3. Sample SPH particles ----------------------
# cumulative mass fraction q(xi) = -xi^2 theta' / (xi1^2 |theta'(xi1)|)
q = (-xi**2*dth) / (xi1**2*abs(dtheta1))
q[0], q[-1] = 0.0, 1.0
q = np.maximum.accumulate(q)          # enforce monotonicity

u_rand = np.random.random(N)
xi_p   = np.interp(u_rand, q, xi)     # invert the mass profile
r_p    = (xi_p/xi1)                   # radius in units of R_star

# isotropic angles
mu  = np.random.uniform(-1, 1, N)
phi = np.random.uniform(0, 2*np.pi, N)
s   = np.sqrt(1 - mu**2)

parts = Particles(N)
parts.mass = M_star/N
parts.x = R_star * (r_p*s*np.cos(phi))
parts.y = R_star * (r_p*s*np.sin(phi))
parts.z = R_star * (r_p*mu)
parts.vx = parts.vy = parts.vz = 0.0 | units.km/units.s

# internal energy: u = n K rho^(1/n), with a small floor near the surface
theta_p = np.interp(xi_p, xi, th)
rho_p   = rho_c * theta_p**n_poly
u_spec  = n_poly * K * rho_p**(1.0/n_poly)
u_floor = 1e-4 * n_poly * K * rho_c**(1.0/n_poly)
parts.u = u_spec.maximum(u_floor)

# ------------- 4. Relax in Fi -------------------------------
t_dyn = (R_star**3/(constants.G*M_star)).sqrt()
print("Dynamical time:", t_dyn.value_in(units.s), "s")

conv = nbody_system.nbody_to_si(M_star, R_star)
sph  = Fi(conv, channel_type='sockets')
sph.parameters.timestep            = t_dyn/100.0
sph.parameters.periodic_box_size   = 20.0 | units.RSun
sph.parameters.verbosity           = 0

sph.gas_particles.add_particles(parts)

to_code   = parts.new_channel_to(sph.gas_particles)
from_code = sph.gas_particles.new_channel_to(parts)

n_steps  = 30
t_relax  = 3.0 * t_dyn
damping  = 0.5

for i in range(1, n_steps+1):
    sph.evolve_model(i*t_relax/n_steps)
    from_code.copy()                      # pull v, rho, etc. from code
    Ekin = parts.kinetic_energy()
    # damp velocities
    parts.vx *= damping
    parts.vy *= damping
    parts.vz *= damping
    to_code.copy_attributes(["vx", "vy", "vz"])
    print(f"step {i:2d}/{n_steps}  t = {(i*t_relax/n_steps).value_in(units.s):8.1f} s"
          f"   Ekin = {Ekin.value_in(units.erg):.3e} erg")

from_code.copy()
sph.stop()

# ------------- 5. Save ---------------------------------------
write_set_to_file(parts, "polytrope_relaxed.amuse", "amuse", overwrite_file=True)
print("Saved relaxed model to polytrope_relaxed.amuse")

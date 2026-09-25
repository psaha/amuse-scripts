from amuse.units import units, constants, nbody_system
from amuse.community.fi.interface import Fi
from amuse.io import write_set_to_file
from scipy.integrate import solve_ivp
from amuse.datamodel import Particles
import numpy as np


# STAR 1
N1 = 50_000
M1 = 1.0 | units.MSun
R1 = 1.0 | units.RSun
n_poly1 = 1.5

# STAR 2
N2 = 70_000
M2 = 1.5 | units.MSun
R2 = 1.5 | units.RSun
n_poly2 = 1.5

# ORBITAL ELEMENTS
a = 5.0 | units.RSun             # a is the separation semi-major axis of the relative orbit: NOT the semi-major axis of either star around the barycentre.
e = 0.2                          # eccentricity
inclination = 20.0 | units.deg   # inclination
omega = 30.0 | units.deg         # argument of pericenter
Omega = 40.0 | units.deg         # longitude of ascending node
true_anomaly = 45.0 | units.deg  # true anomaly

# RELAXATION
N_RELAX_STEPS = 30
RELAX_TIME = 3.0
DAMPING = 0.5

# OUTPUT
OUTPUT_FILE = "amuse_file/binary_polytrope.amuse"
SEED = 42                        # RANDOM SEED

# CHECK ORBITAL PARAMETERS
if a.value_in(units.m) <= 0.0:
    raise ValueError("Semi-major axis must be positive.")

if e < 0.0 or e >= 1.0:
    raise ValueError("This version requires 0 <= e < 1 for a bound elliptical orbit.")

# LANE-EMDEN SOLVER
def solve_lane_emden(n_poly):
    def rhs(xi, y):
        theta, dtheta = y
        th = max(theta, 0.0)
        return [dtheta, -th**n_poly - 2.0*dtheta/xi]

    def surface(xi, y):
        return y[0]

    surface.terminal = True
    surface.direction = -1
    xi0 = 1e-6
    sol = solve_ivp(rhs, [xi0, 20.0], [1.0 - xi0**2/6.0, -xi0/3.0],
        events=surface, rtol=1e-10, atol=1e-12, dense_output=True)

    if len(sol.t_events[0]) == 0:
        raise RuntimeError("Lane-Emden surface was not found.")

    xi1 = sol.t_events[0][0]
    dtheta1 = sol.sol(xi1)[1]
    return sol, xi1, dtheta1

# CREATE INITIAL POLYTROPE
def create_polytrope(N, M_star, R_star, n_poly, seed):
    print()
    print("==========================================")
    print("Creating polytrope")
    print("==========================================")
    print("N      =", N)
    print("Mass   =", M_star)
    print("Radius =", R_star)
    print("n      =", n_poly)

    gamma = 1.0 + 1.0/n_poly
    sol, xi1, dtheta1 = solve_lane_emden(n_poly)       # Lane-Emden
    print(f"xi1 = {xi1:.6f}")
    print(f"theta'(xi1) = {dtheta1:.6f}")

    # Physical scaling
    rho_c = (M_star * xi1 / (4.0 * np.pi * R_star**3 * abs(dtheta1)))
    alpha = R_star / xi1
    K = (4.0 * np.pi * constants.G * alpha**2 * rho_c**(1.0 - 1.0/n_poly) / (n_poly + 1.0))
    print("Central density =", rho_c.value_in(units.g / units.cm**3), "g/cm^3")

    # Fine Lane-Emden grid
    xi = np.linspace(1e-6, xi1, 2000)
    theta = np.clip(sol.sol(xi)[0], 0.0, None)
    dtheta = sol.sol(xi)[1]

    # Mass profile
    q = (-xi**2 * dtheta / (xi1**2 * abs(dtheta1)))
    q[0] = 0.0
    q[-1] = 1.0
    q = np.maximum.accumulate(q)

    # Random sampling
    rng = np.random.default_rng(seed)
    random_mass = rng.random(N)
    xi_p = np.interp(random_mass, q, xi)
    r_p = xi_p / xi1

    # isotropic directions
    mu = rng.uniform(-1.0, 1.0, N)
    phi = rng.uniform(0.0, 2.0*np.pi, N)
    s = np.sqrt(1.0 - mu**2)

    # Create particles
    parts = Particles(N)
    parts.mass = M_star / N
    parts.x = (R_star * r_p * s * np.cos(phi))
    parts.y = (R_star * r_p * s * np.sin(phi))
    parts.z = (R_star * r_p * mu)
    parts.vx = (0.0 | units.km / units.s)
    parts.vy = (0.0 | units.km / units.s)
    parts.vz = (0.0 | units.km / units.s)

    # Internal energy
    theta_p = np.interp(xi_p, xi, theta)
    rho_p = (rho_c * theta_p**n_poly)
    u_spec = (n_poly * K * rho_p**(1.0/n_poly))
    u_floor = (1e-4 * n_poly * K * rho_c**(1.0/n_poly))
    parts.u = u_spec.maximum(u_floor)
    return parts

# RELAX ONE STAR
def relax_star(parts, M_star, R_star, star_name):
    print()
    print("==========================================")
    print("Relaxing", star_name)
    print("==========================================")

    t_dyn = (R_star**3 / (constants.G * M_star)).sqrt()
    print("Dynamical time =", t_dyn.value_in(units.s), "s")
    conv = nbody_system.nbody_to_si(M_star, R_star)
    sph = Fi(conv, channel_type="sockets")
    sph.parameters.timestep = (t_dyn / 100.0)

    # Large enough box for an isolated star
    sph.parameters.periodic_box_size = (20.0 | units.RSun)
    sph.parameters.verbosity = 0
    sph.gas_particles.add_particles(parts)

    to_code = (parts.new_channel_to(sph.gas_particles))
    from_code = (sph.gas_particles.new_channel_to(parts))
    t_relax = (RELAX_TIME * t_dyn)

    for i in range(1, N_RELAX_STEPS + 1):
        t = (i * t_relax / N_RELAX_STEPS)
        sph.evolve_model(t)
        from_code.copy()
        Ekin = parts.kinetic_energy()
        # Damping
        parts.vx *= DAMPING
        parts.vy *= DAMPING
        parts.vz *= DAMPING
        to_code.copy_attributes(["vx", "vy", "vz"])
        print(f"{star_name}: "
            f"step {i:2d}/{N_RELAX_STEPS} "
            f"t = "
            f"{t.value_in(units.s):.3e} s "
            f"Ekin = "
            f"{Ekin.value_in(units.erg):.3e} erg"
        )


    # Pull final state from Fi
    from_code.copy()
    sph.stop()
    return parts

# REMOVE INTERNAL COM POSITION AND VELOCITY
def recenter_star(parts):
    mass = parts.mass
    total_mass = mass.sum()
    x_com = ((mass * parts.x).sum() / total_mass)
    y_com = ((mass * parts.y).sum() / total_mass)
    z_com = ((mass * parts.z).sum() / total_mass)
    vx_com = ((mass * parts.vx).sum() / total_mass)
    vy_com = ((mass * parts.vy).sum() / total_mass)
    vz_com = ((mass * parts.vz).sum() / total_mass)
    parts.x -= x_com
    parts.y -= y_com
    parts.z -= z_com
    parts.vx -= vx_com
    parts.vy -= vy_com
    parts.vz -= vz_com

# KEPLERIAN ORBIT
def kepler_orbit(M1, M2, a, e, inclination, omega, Omega, true_anomaly):
    mu = constants.G * (M1 + M2)    
    inc = inclination.value_in(units.rad)              # Convert angles to radians
    om = omega.value_in(units.rad)
    Om = Omega.value_in(units.rad)
    f = true_anomaly.value_in(units.rad)

    r = (a * (1.0 - e**2) / (1.0 + e*np.cos(f)))       # Orbital separation
    h = (mu * a * (1.0 - e**2)).sqrt()                 # Specific angular momentum
    v_r = (mu / h * e * np.sin(f))                     # Radial and transverse velocities

    v_theta = (mu / h * (1.0 + e*np.cos(f)))
    x_orb = (r * np.cos(f))                            # Position in orbital plane
    y_orb = (r * np.sin(f))
    z_orb = (0.0 | units.m)

    vx_orb = (v_r*np.cos(f) - v_theta*np.sin(f))       # Velocity in orbital plane
    vy_orb = (v_r*np.sin(f) + v_theta*np.cos(f))
    vz_orb = (0.0 | units.m / units.s)

    cw = np.cos(om)
    sw = np.sin(om)
    ci = np.cos(inc)
    si = np.sin(inc)
    cO = np.cos(Om)
    sO = np.sin(Om)

    # Rotation matrix
    R11 = cO*cw - sO*sw*ci
    R12 = -cO*sw - sO*cw*ci
    R21 = sO*cw + cO*sw*ci
    R22 = -sO*sw + cO*cw*ci
    R31 = sw*si
    R32 = cw*si

    # Rotate position
    x_rel = (R11*x_orb + R12*y_orb)
    y_rel = (R21*x_orb + R22*y_orb)
    z_rel = (R31*x_orb + R32*y_orb)

    # Rotate velocity
    vx_rel = (R11*vx_orb + R12*vy_orb)
    vy_rel = (R21*vx_orb + R22*vy_orb)
    vz_rel = (R31*vx_orb + R32*vy_orb)
    return (x_rel, y_rel, z_rel, vx_rel, vy_rel, vz_rel)

# MAIN
print()
print("================================================")
print("        DIRECT BINARY POLYTROPE BUILDER")
print("================================================")

star1 = create_polytrope(N1, M1, R1, n_poly1, SEED)      # 1. CREATE STAR 1
star1 = relax_star(star1, M1, R1, "Star 1")              # 2. RELAX STAR 1
star2 = create_polytrope(N2, M2, R2, n_poly2, SEED + 1)  # 3. CREATE STAR 2
star2 = relax_star(star2, M2, R2, "Star 2")              # 4. RELAX STAR 2

print()
print("Recentering stars...")                            # 5. RECENTER BOTH STARS
recenter_star(star1)
recenter_star(star2)

print()
print("==========================================")
print("Keplerian orbital parameters")
print("==========================================")

print("M1 =", M1.value_in(units.MSun), "Msun")
print("M2 =", M2.value_in(units.MSun), "Msun")
print("Mtot =", (M1 + M2).value_in(units.MSun), "Msun")
print("a =", a.value_in(units.RSun), "Rsun")
print("e =", e)
print("inclination =", inclination.value_in(units.deg), "deg")
print("omega =", omega.value_in(units.deg), "deg")
print("Omega =", Omega.value_in(units.deg), "deg")
print("true anomaly =", true_anomaly.value_in(units.deg), "deg")


(rx, ry, rz, vx, vy, vz) = kepler_orbit(M1, M2, a, e, inclination, omega, Omega, true_anomaly)
print()
print("Relative separation:")
print("r =", rx.value_in(units.RSun), ry.value_in(units.RSun), rz.value_in(units.RSun), "Rsun")
relative_distance = (rx**2 + ry**2 + rz**2).sqrt()
print("|r| =", relative_distance.value_in(units.RSun), "Rsun")
print()
print("Relative velocity:")
print("v =", vx.value_in(units.km / units.s), vy.value_in(units.km / units.s), vz.value_in(units.km / units.s))

Mtot = M1 + M2                    # 7. BARYCENTRIC POSITIONS
r1x = -M2 / Mtot * rx
r1y = -M2 / Mtot * ry
r1z = -M2 / Mtot * rz
r2x = M1 / Mtot * rx
r2y = M1 / Mtot * ry
r2z = M1 / Mtot * rz

# 8. BARYCENTRIC VELOCITIES
v1x = -M2 / Mtot * vx
v1y = -M2 / Mtot * vy
v1z = -M2 / Mtot * vz
v2x = M1 / Mtot * vx
v2y = M1 / Mtot * vy
v2z = M1 / Mtot * vz

# 9. APPLY ORBITAL OFFSETS
star1.x += r1x
star1.y += r1y
star1.z += r1z
star2.x += r2x
star2.y += r2y
star2.z += r2z
star1.vx += v1x
star1.vy += v1y
star1.vz += v1z
star2.vx += v2x
star2.vy += v2y
star2.vz += v2z

# 10. COMBINE BOTH STARS
binary = Particles()
binary.add_particles(star1)
binary.add_particles(star2)

# 11. CHECK FINAL BINARY COM
mass = binary.mass
Mbinary = mass.sum()
x_com = ((mass * binary.x).sum() / Mbinary)
y_com = ((mass * binary.y).sum() / Mbinary)
z_com = ((mass * binary.z).sum() / Mbinary)
vx_com = ((mass * binary.vx).sum() / Mbinary)
vy_com = ((mass * binary.vy).sum() / Mbinary)
vz_com = ((mass * binary.vz).sum() / Mbinary)

print()
print("==========================================")
print("Final binary")
print("==========================================")
print("Number of particles =", len(binary))
print("Total mass =", Mbinary.value_in(units.MSun), "Msun")
print()
print("Binary COM position:")
print(x_com.value_in(units.RSun), y_com.value_in(units.RSun), z_com.value_in(units.RSun), "Rsun")
print()
print("Binary COM velocity:")
print(vx_com.value_in(units.km / units.s), vy_com.value_in(units.km / units.s), vz_com.value_in(units.km / units.s), "km/s")

# 12. FINAL STAR SEPARATION
star1_com_x = ((star1.mass * star1.x).sum() / star1.mass.sum())
star1_com_y = ((star1.mass * star1.y).sum() / star1.mass.sum())
star1_com_z = ((star1.mass * star1.z).sum() / star1.mass.sum())
star2_com_x = ((star2.mass * star2.x).sum() / star2.mass.sum())
star2_com_y = ((star2.mass * star2.y).sum() / star2.mass.sum())
star2_com_z = ((star2.mass * star2.z).sum() / star2.mass.sum())
final_separation = ((star2_com_x - star1_com_x)**2 + (star2_com_y - star1_com_y)**2 + (star2_com_z - star1_com_z)**2).sqrt()

print()
print("Final stellar separation =", final_separation.value_in(units.RSun), "Rsun")

# 13. WRITE BINARY
print()
print("Writing:", OUTPUT_FILE)
write_set_to_file(binary, OUTPUT_FILE, "amuse", overwrite_file=True)
print()
print("==========================================")
print("DONE")
print("==========================================")

print("Binary model saved to:", OUTPUT_FILE)


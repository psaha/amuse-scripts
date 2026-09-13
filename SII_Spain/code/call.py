import matplotlib.pyplot as plt
from posVel import POSVEL
from sph import SPH
from sig import SIG
import numpy as np

np.random.seed(42)

sp = SPH()
sg = SIG()
pv = POSVEL()

##########################################################################################################################

### The orbital parameters
obj1 = ['11.43']                                                                     
obj2 = ['7.21', '4.0145', '0.1311432', '0.133', '63.1', '309.938', '2440678.008', '255.6']
Nobj = 2

# The initial position and velocity of stars
now = 2461314.7916666665                                     # 1st Oct 2026, 7 pm
fname = "orbit/para.txt"
a = pv.parameter(fname, Nobj, obj1, obj2)
mass, pos_com, vel_com = pv.getposvel(fname, now, Nobj)      # Masses (solar), COM position (light second) and COM velocity (light second per second)

### The Star's Parameters
R1 = 17.33                       # Radius of star A in light second
R2 = 8.68                        # Radius of star B in light second
M1 = mass[0]                     # Mass of star A in solar mass
M2 = mass[1]                     # Mass of star B in solar mass
N1 = 250                         # Number of particles in star A
N2 = 100                         # Number of particle in star B
N = N1 + N2                      # Total number of particle in system
d = 7.89e9                       # distance of binary in light second
G = 4.9254909e-6                 # Value of G (Mass should be in solar mass, distance in light second and time in second)

# Randomly selected positions and velocities of N particles  
pos = np.random.randn(N, 3)             
vel = np.zeros(pos.shape)

# Center of the stars
c1 = np.mean(pos[:N1], axis=0)
c2 = np.mean(pos[N1:], axis=0)

# Shift each star to its own center
pos[:N1] -= c1
pos[N1:] -= c2

# Place the relaxed stars on the Keplerian orbit (the plus and minus sign will be in pos_com and vel_com)
pos[:N1] += pos_com[0]
pos[N1:] += pos_com[1]

vel[:N1] += vel_com[0]
vel[N1:] += vel_com[1]

#########################################################################################################################

# Parameter for SPH
h1 = 5.5                         # Length of smoothing kernel for star A
h2 = 4                           # length of smoothing kernel for star B
n1 = 3                           # Polytropic index for star A
n2 = 3                           # Polytropic index for star B
k1 = 4.04e-26                    # Polytropic equation-of-state constant for star A
k2 = 2.97e-26                    # Polytropic equation-of-state constant for star B
nu1 = 0.15                       # Damping constant for star A to reach in equilibrium
nu2 = 0.1                        # Damping constant for star B to reach in equilibrium

# Parameter for Harmonic accelaration
lmbda1 = sp.getLamda(k1, n1, M1, R1)
lmbda2 = sp.getLamda(k2, n2, M2, R2)

# Calculate initial gravitational accelerations
acc = sp.getAcc(pos, vel, M1, M2, N1, lmbda1, lmbda2, h1, h2, G, k1, k2, n1, n2, nu1, nu2, relax=True)

# Observation parameters
lam = 550e-9
u = np.linspace(-800, 800, 501)
v = np.linspace(-800, 800, 501)
u, v = np.meshgrid(u, v)

# If each particle in stars are with same mass   
m1 = np.full(N1, M1/N1)
m2 = np.full(N2, M2/N2)
m = np.concatenate((m1, m2))

# If each point in stars have the same smoothing length   
h1_arr = np.full(N1, h1)
h2_arr = np.full(N2, h2)
h = np.concatenate((h1_arr, h2_arr))

# The density and the brightness of particles
rho = sp.getDensity(pos, pos, m, h)
brightness = rho.flatten()

# The particles on surface and the square visibility according to it
x, y, cv = sp.getSurface(pos, brightness, 0.5)
sig = sg.sqVisi(u, v, x, y, d, lam, cv)

#############################################################################################################################
# Base for the Figure
plt.ion()
fig, (ax1, ax_mid, ax2) = plt.subplots(1, 3, figsize=(20, 5))

# Left panel
sc = ax1.scatter(x, y, c=cv, cmap="autumn")
ax1.set_xlim(-50, 50)
ax1.set_ylim(-50, 50)
ax1.set_aspect("equal")
ax1.set_xlabel("Sky along x axis (light-second)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Sky along y axis (light-second)", fontsize=12, fontweight='bold')
ax1.tick_params(axis='both', labelsize=11)
ax1.set_facecolor((0.1, 0.1, 0.1))
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontweight('bold')
    
# Middle Panel
relx = []
rely = []
orbit_line, = ax_mid.plot([], [], 'b-', lw=2)
current_pos, = ax_mid.plot([], [], 'ro', ms=6)
ax_mid.set_aspect('equal')
ax_mid.set_xlabel("Relative x (light-second)", fontsize=12, fontweight='bold')
ax_mid.set_ylabel("Relative y (light-second)", fontsize=12, fontweight='bold')
ax_mid.set_title("Relative Orbit", fontsize=12, fontweight='bold')
ax_mid.tick_params(axis='both', labelsize=11)
for label in ax_mid.get_xticklabels() + ax_mid.get_yticklabels():
    label.set_fontweight('bold')
    

# Right panel
im = ax2.imshow(sig, origin="lower", extent=[u.min(), u.max(), v.min(), v.max()])
ax2.set_xlabel("Baseline u (meter)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Baseline v (meter)", fontsize=12, fontweight='bold')
ax2.set_title("The signal $|V(u,v)|^2$ on Earth", fontsize=12, fontweight='bold')
ax2.tick_params(axis='both', labelsize=11)
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontweight('bold')

cbar = fig.colorbar(im, ax=ax2, shrink=0.9, pad=0.02)  # for colorbar
plt.show(block=False)

###############################################################################################################################
# Relaxation parameters
KE1_old = None
KE2_old = None
relax = True
stable_steps = 0
tol1 = 5e-3
tol2 = 5e-3

# Time parameters
Orb = float(obj2[1]) * 86400     # One orbit
dt = 1800
step = int(Orb / dt) + 1
t = 0.0

# Main loop
for i in range(step):
    # ---------------- Energies ----------------
    KE1 = 0.5 * np.sum(m[:N1] * np.sum(vel[:N1]**2, axis=1))
    KE2 = 0.5 * np.sum(m[N1:] * np.sum(vel[N1:]**2, axis=1))

    if relax and KE1_old is not None:

        dKE1 = abs(KE1 - KE1_old) / max(KE1_old, 1e-20)
        dKE2 = abs(KE2 - KE2_old) / max(KE2_old, 1e-20)

        print("dKE1 =", dKE1)
        print("dKE2 =", dKE2)

        if (dKE1 < tol1) and (dKE2 < tol2):
            stable_steps += 1
        else:
            stable_steps = 0

        if stable_steps >= 10:
            relax = False
            print(f"Stars reached equilibrium at t = {t:.1f}")

    KE1_old = KE1
    KE2_old = KE2

    # ---------------- Leapfrog ----------------
    vel += 0.5 * acc * dt
    pos += vel * dt
    acc = sp.getAcc(pos, vel, M1, M2, N1, lmbda1, lmbda2, h1, h2, G, k1, k2, n1, n2, nu1, nu2, relax=relax)
    vel += 0.5 * acc * dt
    t += dt

    # ---------------- Observation ----------------
    rho = sp.getDensity(pos, pos, m, h)
    brightness = rho.flatten()
    x, y, cv = sp.getSurface(pos, brightness, 0.5)
    sig = sg.sqVisi(u, v, x, y, d, lam, cv)

    # ---------------- Update scatter ----------------
    sc.set_offsets(np.column_stack((x, y)))
    sc.set_array(cv)
    ax1.set_title(f"Stars at t = {t:.0f} seconds", fontsize=12, fontweight='bold')

    # Update relative orbit
    c1 = np.mean(pos[:N1], axis=0)
    c2 = np.mean(pos[N1:], axis=0)
    rrel = c2 - c1
    relx.append(rrel[0])
    rely.append(rrel[1])
    orbit_line.set_data(relx, rely)
    current_pos.set_data([relx[-1]], [rely[-1]])
    ax_mid.set_aspect('equal')
    L = 60                                           # choose a value slightly larger than the expected orbit radius
    ax_mid.set_xlim(-L, L)
    ax_mid.set_ylim(-L, L)

    # ---------------- Update image ----------------
    im.set_data(sig)
    im.set_clim(sig.min(), sig.max())
    cbar.update_normal(im)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

# Save final frame
plt.savefig("sph.png", dpi=240)
plt.ioff()
plt.show()

